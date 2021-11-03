pub mod compute;
use std::error;
pub mod compiler;
pub mod onnx;
pub mod resource;
pub mod utils;
use log::debug;
use protobuf::{self, Message};
use std::collections::HashMap;
use std::time::Instant;
use tera::Tera;
use utils::{get_dimension, len};
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

        let value_info = graph.get_value_info();

        let output_info = &graph.get_output();

        let mut kernel_3_inputs = vec![];

        // Pad convolution layer that has shape [3, 3] with 4 bytes.
        for node in model.get_graph().get_node() {
            let mut pads_default = onnx::AttributeProto::new();
            pads_default.set_ints(vec![0, 0, 0, 0]);

            let mut strides_default = onnx::AttributeProto::new();
            strides_default.set_ints(vec![1, 1]);

            if node.get_op_type() == "Conv"
                && utils::get_attribute("kernel_shape", None, node).get_ints() == [3, 3]
                && utils::get_attribute("pads", Some(&pads_default), node).get_ints()
                    == [1, 1, 1, 1]
                && utils::get_attribute("strides", Some(&strides_default), node).get_ints()
                    == [1, 1]
            {
                let string = node.get_input()[1].as_str();
                kernel_3_inputs.push(string);
            }

            let outputs = node.get_output();

            if let Some(output_dims) = get_dimension(&value_info, &outputs[0]) {
                inner_infos.insert(
                    outputs[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            len(&output_dims) as _,
                            outputs[0].as_str(),
                        ),
                        dims: output_dims,
                    },
                );
            } else if let Some(output_dims) = get_dimension(&output_info, &outputs[0]) {
                inner_infos.insert(
                    outputs[0].clone(),
                    InnerInfo {
                        buffer: resource::output_buffer(
                            device,
                            len(&output_dims) as _,
                            outputs[0].as_str(),
                        ),
                        dims: output_dims,
                    },
                );
            } else {
                panic!("output dims was not provided. You can use python's onnx-simplifier to generate implied dimensions.")
            }
        }

        for initializer in initializers {
            let input = initializer.get_name();

            let dims = initializer.get_dims().to_vec();
            let data = initializer.get_float_data();
            let mut raw_data = if !data.is_empty() {
                bytemuck::cast_slice(&data)
            } else {
                initializer.get_raw_data()
            };

            let n = raw_data.len() / 12;

            let mut padded_data = vec![];

            // Reformat the data for 3-kernel as it is not memory optimized.
            if kernel_3_inputs.contains(&input) {
                for i in 0..n {
                    padded_data.extend_from_slice(&raw_data[12 * i..12 * (i + 1)]);
                    padded_data.extend_from_slice(&[0; 4]);
                }
                raw_data = padded_data.as_slice();
            }

            if !raw_data.is_empty() {
                inner_infos.insert(
                    input.to_string(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, raw_data, input),
                        dims,
                    },
                );
            } else {
                debug!("Not inserting input: {} with shape: {:?}", input, dims);
            };
        }

        Ok(inner_infos)
    }
}

#[cfg_attr(doc, aquamarine::aquamarine)]
/// ```mermaid
/// graph LR
///     s([Source]) --> a[[aquamarine]]
///     r[[rustdoc]] --> f([Docs w/ Mermaid!])
///     subgraph rustc[Rust Compiler]
///     a -. inject mermaid.js .-> r
///     end
/// ```
pub async fn run(
    session: &mut Session,
    input_data: HashMap<String, (&[f32], &[i64])>,
) -> Result<Vec<f32>> {
    let device = &session.device;
    let inner_infos = &mut session.inner_infos;
    for (input, (data, dims)) in input_data {
        inner_infos.insert(
            input.to_string(),
            InnerInfo {
                buffer: resource::create_buffer_init(device, data, &input),
                dims: dims.to_vec(),
            },
        );
    }
    let time_start = Instant::now();

    let graph = session.model.get_graph();
    let outputs = graph.get_output();
    let queue = &session.queue;
    let tera = &session.tera;

    compute::compute(device, queue, graph, inner_infos, tera).unwrap();

    let buffer_slice = inner_infos
        .get(outputs[0].get_name())
        //.get(&"squeezenet0_conv3_weight".to_string())
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
    Ok(result)
}

pub struct InnerInfo {
    buffer: wgpu::Buffer,
    dims: Vec<i64>,
}
