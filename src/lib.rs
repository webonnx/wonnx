use std::error;
pub mod compiler;
pub mod onnx;
pub mod optimisation;
pub mod resource;
pub mod sequencer;
pub mod utils;

#[macro_use]
extern crate lazy_static;

use optimisation::EncoderBuilder;
use protobuf::{self, Message};
use std::collections::HashMap;

use crate::resource::resize;
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
// +----------------+
// |ONNX Path       |
// +----------------+
//         v
// +----------------+
// |ONNX Model      |
// +----------------+
//         v
// +----------------+
// |Session         |                        Optimisation on       Node -> WGSL Shader
// +----------------+                        Mutiple node
//         v
// +----------------+   +----------------+   +----------------+   +----------------+
// |load (once)     | > |Load params     | > |Sequencer       | > |Compiler        |
// +----------------+   +----------------+   +----------------+   +----------------+
//         v
// +----------------+
// |run             |  (Can be run multiple times)
// +----------------+
//
//
pub struct Session {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub builders: Vec<EncoderBuilder>,
    pub outputs: Vec<String>,
}

impl Session {
    // Read an ONNX model from a path and create a session.
    pub async fn from_path(path: &str) -> Result<Session> {
        let model = onnx::ModelProto::parse_from_bytes(
            &std::fs::read(path).expect("ONNX Model path not found."),
        )
        .expect("Could not deserialize the Model");

        Session::from_model(model).await
    }

    // Create a Session given an ONNX model.
    pub async fn from_model(model: onnx::ModelProto) -> Result<Session> {
        let promise = resource::request_device_queue();
        let (device, queue) = promise.await;

        // Find the version of the ONNX operator set this model is using (this is useful because some operators' specifications change over time).
        // Note, if any other op set than the ONNX operator set is referenced, we cannot run the model.
        // See https://github.com/onnx/onnx/blob/master/docs/Versioning.md#operator-sets
        let mut onnx_opset_version = None;
        for opset_import in model.get_opset_import() {
            match opset_import.get_domain() {
                "" => {
                    // This is a reference to the ONNX specification op set
                    if let Some(onnx_version) = onnx_opset_version {
                        panic!(
                            "two onnx operator set versions specified (first is {}, second is {})",
                            onnx_version,
                            opset_import.get_version()
                        );
                    } else {
                        onnx_opset_version = Some(opset_import.get_version());
                    }
                }
                some_other_opset => {
                    panic!(
                        "the model references other op sets than the ONNX specified opset ({})",
                        some_other_opset
                    );
                }
            }
        }

        let onnx_opset_version =
            onnx_opset_version.expect("no ONNX opset was referenced by the model!");
        let graph = model.get_graph();
        let (buffers, builders) = optimisation::load(graph, &device, onnx_opset_version).unwrap();

        let outputs = graph
            .get_output()
            .iter()
            .map(|info| (info.get_name().to_string()))
            .collect();

        // The data is loaded after the first submit
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        queue.submit(Some(encoder.finish()));

        Ok(Session {
            device,
            queue,
            buffers,
            builders,
            outputs,
        })
    }

    // Run use the element loaded into the session to produce the inference.
    // It copy input data to the buffers.
    // Run the command encoder.
    // Copy the output into an exit buffer that can be deleted.
    pub async fn run(&self, inputs: HashMap<String, &[f32]>) -> Result<HashMap<String, Vec<f32>>> {
        let device = &self.device;
        let queue = &self.queue;
        let builders = &self.builders;
        let inner_infos = &self.buffers;
        let outputs = &self.outputs;

        // Copy input data
        for (input, data) in inputs {
            queue.write_buffer(
                inner_infos.get(&input).unwrap_or_else(|| {
                    panic!(
                        "Invalid input: {}, try to use netron.app to see the correct input name",
                        input
                    )
                }),
                0,
                bytemuck::cast_slice(&resize(data.to_vec())),
            )
        }

        // Run the command encoder generated during the load
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        for builder in builders {
            let (x, y, z) = builder.threads;

            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&builder.pipeline);
            for (index, bind_group) in builder.bind_groups.iter().enumerate() {
                cpass.set_bind_group(index as u32, bind_group, &[]);
            }
            cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
        }

        queue.submit(Some(encoder.finish()));

        let mut results = HashMap::new();

        for output in outputs {
            // Copy the output data into the staging buffer.
            let buffer = inner_infos.get(output.as_str()).unwrap();

            let buffer_slice = buffer.slice(..);
            // TODO: Define behavior for multi output.
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

            device.poll(wgpu::Maintain::Wait);

            buffer_future.await.expect("failed to run compute on gpu!");
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            results.insert(output.clone(), bytemuck::cast_slice(&data).to_vec());
            drop(data);
            buffer.unmap();
        }

        Ok(results)
    }
}
