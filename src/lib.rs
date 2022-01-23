pub mod compiler;
pub mod onnx;
pub mod optimisation;
pub mod resource;
pub mod sequencer;
pub mod utils;

#[macro_use]
extern crate lazy_static;

use compiler::CompileError;
use onnx::ValueInfoProto;
use optimisation::{OptimizationError, OptimizedModel};
use protobuf::{self, Message, ProtobufError};
use std::collections::HashMap;
use std::result::Result;

use crate::resource::resize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WonnxError {
    #[error("error compiling model: {0}")]
    CompileError(#[from] CompileError),

    #[error("error executing the model: {0}")]
    SessionError(#[from] SessionError),
}

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
    pub optimized_model: OptimizedModel,
    pub outputs: Vec<ValueInfoProto>,
}

#[derive(Error, Debug)]
pub enum SessionError {
    #[error("could not deserialize model: {0}")]
    ModelDeserializationError(#[from] ProtobufError),

    #[error("an error occurred reading the model file: {0}")]
    ModelReadingError(#[from] std::io::Error),

    #[error(
        "invalid input name '{0}'; inspect the file with e.g. Netron to find the correct name"
    )]
    InvalidInput(String),

    #[error(
        "invalid output name '{0}'; inspect the file with e.g. Netron to find the correct name"
    )]
    InvalidOutput(String),

    #[error("optimization failed: {0}")]
    OptimizationFailed(#[from] OptimizationError),

    #[error("more than one ONNX opset was specified: {0} and {1}")]
    DuplicateONNXOpset(i64, i64),

    #[error("the model references an unknown opset: '{0}'")]
    UnknownOpset(String),

    #[error("the model did not reference a specific version of the ONNX opset")]
    UnknownONNXOpsetVersion,
}

impl Session {
    // Read an ONNX model from a path and create a session.
    pub async fn from_path(path: &str) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;

        Session::from_model(model).await
    }

    pub async fn from_bytes(bytes: &[u8]) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;

        Session::from_model(model).await
    }

    // Create a Session given an ONNX model.
    pub async fn from_model(model: onnx::ModelProto) -> Result<Session, SessionError> {
        let (device, queue) = resource::request_device_queue().await;

        // Find the version of the ONNX operator set this model is using (this is useful because some operators' specifications change over time).
        // Note, if any other op set than the ONNX operator set is referenced, we cannot run the model.
        // See https://github.com/onnx/onnx/blob/master/docs/Versioning.md#operator-sets
        let mut onnx_opset_version = None;
        for opset_import in model.get_opset_import() {
            match opset_import.get_domain() {
                "" => {
                    // This is a reference to the ONNX specification op set
                    if let Some(onnx_version) = onnx_opset_version {
                        if opset_import.get_version() != onnx_version {
                            return Err(SessionError::DuplicateONNXOpset(
                                onnx_version,
                                opset_import.get_version(),
                            ));
                        }
                    } else {
                        onnx_opset_version = Some(opset_import.get_version());
                    }
                }
                some_other_opset => {
                    return Err(SessionError::UnknownOpset(some_other_opset.to_string()));
                }
            }
        }

        // Optimize and compile the model graph to a set of buffers and 'builders' which can basically run GPU shader code referencing these buffers
        let onnx_opset_version = onnx_opset_version.ok_or(SessionError::UnknownONNXOpsetVersion)?;
        let graph = model.get_graph();
        let optimized_model = optimisation::load(graph, &device, onnx_opset_version)?;

        // The data is loaded after the first submit
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        queue.submit(Some(encoder.finish()));

        Ok(Session {
            device,
            queue,
            optimized_model,
            outputs: graph.get_output().to_vec(),
        })
    }

    // Run use the element loaded into the session to produce the inference.
    // It copy input data to the buffers.
    // Run the command encoder.
    // Copy the output into an exit buffer that can be deleted.
    pub async fn run(
        &self,
        inputs: HashMap<String, &[f32]>,
    ) -> Result<HashMap<String, Vec<f32>>, SessionError> {
        let device = &self.device;
        let queue = &self.queue;
        let buffers = &self.optimized_model.buffers;
        let outputs = &self.outputs;

        // Copy input data to the GPU buffer where our first node will read it
        for (input, data) in inputs {
            queue.write_buffer(
                buffers
                    .get(&input)
                    .ok_or(SessionError::InvalidInput(input))?,
                0,
                bytemuck::cast_slice(&resize(data.to_vec())),
            )
        }

        // Run the command encoder generated during the load
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Sequentially execute each node (compiled to a shader 'builder')
        for builder in &self.optimized_model.builders {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&builder.pipeline);
            for (index, bind_group) in builder.bind_groups.iter().enumerate() {
                cpass.set_bind_group(index as u32, bind_group, &[]);
            }

            let (x, y, z) = builder.threads;
            cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
        }

        queue.submit(Some(encoder.finish()));
        let mut results = HashMap::new();

        for output in outputs {
            let output_name = output.get_name();

            // Copy the output data into the staging buffer.
            let buffer = buffers
                .get(output_name)
                .ok_or_else(|| SessionError::InvalidOutput(output_name.to_string()))?;

            let buffer_slice = buffer.slice(..);
            // TODO: Define behavior for multi output.
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

            device.poll(wgpu::Maintain::Wait);

            buffer_future.await.expect("failed to run compute on gpu!");
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32

            // The actual buffer may be bigger than what we should return, because buffers have a minimum size in wgpu
            // Fetch the size we should expect so we can chop the buffer to the correct size
            let output_buffer_size = output.get_shape().element_count() as usize;

            results.insert(
                output_name.to_string(),
                bytemuck::cast_slice(&data)[..output_buffer_size].to_vec(),
            );
            drop(data);
            buffer.unmap();
        }

        Ok(results)
    }
}
