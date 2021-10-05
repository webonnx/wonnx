pub mod boilerplate;
pub mod compute;
use std::error;
pub mod compiler;
pub mod dimensions;
pub mod onnx;
pub mod resource;
pub mod utils;
use protobuf::{self, Message};
use std::collections::HashMap;
use std::time::Instant;
use tera::Tera;
// Change the alias to `Box<error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;
/// Creates a new session connected to the GPU.
///
/// Generate a session that will translate the onnx format into WGSL instructions.
///
/// # Examples
///
/// Basic usage:
///
/// ```ignore
/// let mut session = Session::from_path("path/to/model.onnx").await.unwrap();
/// ```
pub struct Session {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub model: onnx::ModelProto,
    pub inner_infos: HashMap<String, InnerInfo>,
    pub tera: Tera,
}

impl Session {
    pub async fn from_path(path: &str) -> Result<Session> {
        let promise = resource::request_device_queue();

        let tera = match Tera::new("templates/**/*.wgsl") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        };
        let (device, queue) = promise.await;

        let model = onnx::ModelProto::parse_from_bytes(
            &std::fs::read(path).expect("ONNX Model path not found."),
        )
        .expect("Could not deserialize the Model");

        let inner_infos = Session::load_initializers(&device, &model).unwrap();

        Ok(Session {
            device,
            queue,
            model,
            inner_infos,
            tera,
        })
    }

    pub async fn from_model(model: onnx::ModelProto) -> Result<Session> {
        let promise = resource::request_device_queue();

        let tera = match Tera::new("templates/**/*.wgsl") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        };
        let (device, queue) = promise.await;

        let inner_infos = Session::load_initializers(&device, &model).unwrap();

        Ok(Session {
            device,
            queue,
            model,
            inner_infos,
            tera,
        })
    }

    pub fn load_initializers(
        device: &wgpu::Device,
        model: &onnx::ModelProto,
    ) -> Result<HashMap<std::string::String, InnerInfo>> {
        let mut inner_infos = HashMap::new();
        let initializers = model.get_graph().get_initializer();
        let graph = model.get_graph();
        for initializer in initializers.iter() {
            let input = initializer.get_name();

            let initiated_data = initializers
                .iter()
                .find(|x| x.get_name() == input)
                .unwrap_or_else(|| panic!("Did not find initializer for input: {}", input));

            let initiated_data_dims = initiated_data.get_dims().to_vec();
            inner_infos.insert(
                input.to_string(),
                InnerInfo {
                    buffer: resource::create_buffer_init(
                        device,
                        initiated_data.get_float_data(),
                        input,
                    ),
                    dims: initiated_data_dims.clone(),
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }

        let inputs = graph.get_input();

        for node in graph.get_node().iter() {
            dimensions::generate_buffer(node, inputs, device, &mut inner_infos, initializers);
        }

        Ok(inner_infos)
    }
}

pub async fn run(
    session: &mut Session,
    input_data: HashMap<String, (&[f32], &[i64])>,
) -> Option<Vec<f32>> {
    let time_start = Instant::now();
    let device = &session.device;
    let inner_infos = &mut session.inner_infos;
    for (input, (data, dims)) in input_data.iter() {
        inner_infos.insert(
            input.to_string(),
            InnerInfo {
                buffer: resource::create_buffer_init(device, data, input),
                dims: dims.to_vec(),
                inner_type: crate::compute::InnerType::ArrayVector,
            },
        );
    }

    let graph = session.model.get_graph();
    let outputs = graph.get_output();
    let queue = &session.queue;
    let tera = &session.tera;

    let mut previous_node = &graph.get_node()[0];
    for node in graph.get_node().iter() {
        let previous_node_op_type = previous_node.get_op_type();
        let node_op_type = node.get_op_type();

        if previous_node_op_type == "Conv" && node_op_type == "Relu" {
            let mut tmp_node = crate::onnx::NodeProto::new();
            tmp_node.set_op_type("ConvRelu".to_string());
            tmp_node.set_name("ConvRelu".to_string());
            tmp_node.set_input(protobuf::RepeatedField::from(
                previous_node.get_input().to_vec(),
            ));
            tmp_node.set_attribute(protobuf::RepeatedField::from(previous_node.get_attribute()));
            tmp_node.set_output(protobuf::RepeatedField::from(node.get_output().to_vec()));

            compute::wrapper(device, queue, graph, &tmp_node, inner_infos, tera).unwrap();
        } else if previous_node_op_type == "Conv" && node_op_type != "Relu" {
            compute::wrapper(device, queue, graph, previous_node, inner_infos, tera).unwrap();
            compute::wrapper(device, queue, graph, node, inner_infos, tera).unwrap();
        } else if node_op_type == "Conv" {
        } else if ["Dropout"].contains(&node_op_type) {
            let mut tmp_node = crate::onnx::NodeProto::new();
            tmp_node.set_op_type(previous_node_op_type.to_string());
            tmp_node.set_name("Some node".to_string());
            tmp_node.set_input(protobuf::RepeatedField::from(
                previous_node.get_input().to_vec(),
            ));
            tmp_node.set_attribute(protobuf::RepeatedField::from(previous_node.get_attribute()));
            tmp_node.set_output(protobuf::RepeatedField::from(node.get_output().to_vec()));

            compute::wrapper(device, queue, graph, &tmp_node, inner_infos, tera).unwrap();
        } else {
            compute::wrapper(device, queue, graph, previous_node, inner_infos, tera).unwrap();
        }

        previous_node = node;
    }

    let buffer_slice = inner_infos
        .get(outputs[0].get_name())
        .unwrap()
        .buffer
        .slice(..);
    // TODO: Define behavior for multi output.
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);
    // // OUTPUT

    buffer_future.await.expect("failed to run compute on gpu!");
    // Gets contents of buffer
    let data = buffer_slice.get_mapped_range();
    // Since contents are got in bytes, this converts these bytes back to f32
    let result = bytemuck::cast_slice(&data).to_vec();

    println!("time: post_wait: {:#?}", time_start.elapsed());

    Some(result)
}
pub struct InnerInfo {
    buffer: wgpu::Buffer,
    dims: Vec<i64>,
    inner_type: compute::InnerType,
}

pub fn get_attribute<'a>(
    attribute: &'a str,
    defaults: Option<&'a onnx::AttributeProto>,
    node: &'a onnx::NodeProto,
) -> &'a onnx::AttributeProto {
    match defaults {
        Some(default) => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .unwrap_or(default),
        None => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .unwrap_or_else(|| {
                panic!(
                    "Did not find required attribute: {}, for node: {}",
                    attribute,
                    node.get_name()
                )
            }),
    }
}
