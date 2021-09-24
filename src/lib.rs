pub mod boilerplate;
pub mod compute;
pub mod ir;
use std::error;
pub mod onnx;
pub mod op;
pub mod resource;
use log::debug;
use protobuf::{self, Message};
use std::collections::HashMap;
// Change the alias to `Box<error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

/// Creates a new session connectedd to the GPU.
///
/// Generate a session that will translate the onnx format into WGSL instructions.
///
/// # Examples
///
/// Basic usage:
///
/// ```ignore
/// let session = Session::from_path("path/to/model.onnx").await.unwrap();
/// ```
pub struct Session {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub model: onnx::ModelProto,
}

impl Session {
    pub async fn from_path(path: &str) -> Result<Session> {
        let (device, queue) = resource::request_device_queue().await;

        let model = onnx::ModelProto::parse_from_bytes(
            &std::fs::read(path).expect("ONNX Model path not found."),
        )
        .expect("Could not deserialize the Model");

        debug!("model: {:#?}", model);

        Ok(Session {
            device,
            queue,
            model,
        })
    }

    pub async fn from_model(model: onnx::ModelProto) -> Result<Session> {
        let (device, queue) = resource::request_device_queue().await;

        debug!("model: {:#?}", model);

        Ok(Session {
            device,
            queue,
            model,
        })
    }

    pub async fn run(&self, input_data: HashMap<&str, (&[f32], &[i64])>) -> Option<Vec<f32>> {
        let graph = self.model.get_graph();
        let device = &self.device;
        let queue = &self.queue;

        let mut inner_infos = HashMap::new();

        let inputs = graph.get_input();
        let value_infos = graph.get_value_info();
        let outputs = graph.get_output();

        for input in inputs.iter() {
            let name = input.get_name();
            let (data, dim) = input_data.get(name).expect(
                format!("Input: {name} was not found in user HashMap.", name = name,).as_str(),
            );
            inner_infos.insert(
                name,
                InnerInfo {
                    attribute_info: &input,
                    buffer: resource::create_buffer_init(&device, data),
                    dims: Some(dim.to_vec()),
                },
            );
        }

        for output in outputs.iter() {
            inner_infos.insert(
                output.get_name(),
                InnerInfo {
                    attribute_info: &output,
                    buffer: resource::create_buffer(&device, resource::size(output) as _),
                    dims: None,
                },
            );
        }

        for value_info in value_infos.iter() {
            inner_infos.insert(
                value_info.get_name(),
                InnerInfo {
                    attribute_info: &value_info,
                    buffer: resource::create_buffer(&device, resource::size(value_info) as _),
                    dims: None,
                },
            );
        }

        infer_shape(&mut inner_infos, graph);

        compute::wrapper(device, queue, graph, &inner_infos).unwrap();

        // TODO: Define behavior for multi output.
        let buffer_slice = inner_infos
            .get(outputs[0].get_name())
            .unwrap()
            .buffer
            .slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        // OUTPUT

        if let Ok(()) = buffer_future.await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result = bytemuck::cast_slice(&data).to_vec();

            //            drop(data);

            Some(result)
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

pub struct InnerInfo<'a> {
    attribute_info: &'a onnx::ValueInfoProto,
    buffer: wgpu::Buffer,
    dims: Option<Vec<i64>>,
}

pub fn infer_shape(inner_infos: &mut HashMap<&str, InnerInfo>, graph: &onnx::GraphProto) {
    let nodes = graph.get_node();

    for node in nodes.iter() {
        let input = node.get_input();
        let output = node.get_output();

        let input_dims = inner_infos.get(input[0].as_str()).unwrap().dims.clone();

        match node.get_op_type() {
            "Abs" | "Add" | "Relu" => {
                let mut output_info = inner_infos.get_mut(output[0].as_str()).unwrap();
                output_info.dims = input_dims;
            }
            "Transpose" => {
                let mut output_info = inner_infos.get_mut(output[0].as_str()).unwrap();
                let dims = input_dims.unwrap();
                output_info.dims = Some(vec![dims[1], dims[0]]);
            }

            _ => unimplemented!(),
        }
    }
}
