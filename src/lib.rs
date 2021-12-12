pub mod compute;
use std::error;
pub mod compiler;
pub mod onnx;
pub mod optimisation;
pub mod resource;
pub mod sequencer;
pub mod utils;

use optimisation::EncoderBuilder;
use protobuf::{self, Message};
use std::collections::HashMap;
use std::time::Instant;
use utils::{get_dimension, len};
use wgpu::BufferUsages;

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
// |Session         |
// +----------------+
//         v
// +----------------+
// |load (once)     |
// +----------------+
//         v
// +----------------+
// |run             |
// +----------------+
//         v
// +----------------+
// |dispatch        |
// +----------------+
//         v
// +----------------+
// |Output          |
// +----------------+
///
///
pub struct Session {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub inner_infos: HashMap<String, wgpu::Buffer>,
    pub builders: Vec<EncoderBuilder>,
    pub output: String,
    pub output_dims: Vec<i64>,
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

        let (inner_infos, builders) = optimisation::load(model.get_graph(), &device).unwrap();

        let graph = model.get_graph();
        let output = graph.get_output()[0].get_name().to_string();
        let output_info = graph.get_output();
        let output_dims = get_dimension(output_info, &output).unwrap();

        Ok(Session {
            device,
            queue,
            inner_infos,
            builders,
            output,
            output_dims,
        })
    }
}

// Run use the element loaded into the session to produce the inference.
// It copy input data to the buffers.
// Run the command encoder.
// Copy the output into an exit buffer that can be deleted.
pub async fn run(session: &Session, input_data: HashMap<String, &[f32]>) -> Result<Vec<f32>> {
    let time_pre_run = Instant::now();
    let device = &session.device;
    let queue = &session.queue;
    let builders = &session.builders;
    let inner_infos = &session.inner_infos;

    // Copy input data
    for (input, data) in input_data {
        let buffer = inner_infos.get(&input).unwrap();
        {
            let buffer_slice = buffer.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);
            device.poll(wgpu::Maintain::Wait);
            buffer_future.await.unwrap();
            let mut buffer_write = buffer_slice.get_mapped_range_mut();
            buffer_write.copy_from_slice(bytemuck::cast_slice(&resize(data.to_vec())));
            drop(buffer_write);
            buffer.unmap();
        }
    }

    println!("time: pre_run: {:#?}", time_pre_run.elapsed());
    let time_run = Instant::now();

    // Run the command encoder generated during the load
    for builder in builders {
        compute::wrapper(device, queue, builder).unwrap();
    }

    // Copy the output data into the exit buffer.
    let staging_buffer = resource::buffer(
        device,
        len(&session.output_dims) as _,
        &(String::from("staging_") + &session.output),
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let buffer = inner_infos.get(&session.output).unwrap();
    encoder.copy_buffer_to_buffer(
        buffer,
        0,
        &staging_buffer,
        0,
        (len(&session.output_dims) * 4) as _,
    );
    queue.submit(Some(encoder.finish()));

    println!("time: run: {:#?}", time_run.elapsed());
    let time_post_run = Instant::now();

    let buffer_slice = staging_buffer.slice(..);
    // TODO: Define behavior for multi output.
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    // OUTPUT

    buffer_future.await.expect("failed to run compute on gpu!");
    // Gets contents of buffer
    let data = buffer_slice.get_mapped_range();
    // Since contents are got in bytes, this converts these bytes back to f32
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    println!("time: post_run: {:#?}", time_post_run.elapsed());
    Ok(result)
}
