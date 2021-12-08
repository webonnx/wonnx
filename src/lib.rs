pub mod compute;
use std::error;
pub mod compiler;
pub mod onnx;
pub mod optimisation;
pub mod resource;
pub mod sequencer;
pub mod utils;

use protobuf::{self, Message, RepeatedField};
use std::collections::HashMap;
use std::time::Instant;
use utils::{get_dimension, len};
use wgpu::BufferUsages;
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
// |load (omce)     |
// +----------------+
//         v
// +----------------+
// |run             |
// +----------------+
//         v
// +----------------+
// |wrap            |
// +----------------+
//         v
// +----------------+
// |compile         |
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
    pub model: onnx::ModelProto,
    pub inner_infos: HashMap<String, wgpu::Buffer>,
    pub pipelines: Vec<wgpu::ComputePipeline>,
    pub groups: Vec<Vec<wgpu::BindGroup>>,
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
    pub async fn from_model(mut model: onnx::ModelProto) -> Result<Session> {
        let promise = resource::request_device_queue();

        let (device, queue) = promise.await;

        let (nodes, inner_infos, pipelines, groups) =
            optimisation::load(model.get_graph(), &device).unwrap();

        let graph = model.mut_graph();
        graph.set_node(RepeatedField::from(nodes));

        Ok(Session {
            device,
            queue,
            model,
            inner_infos,
            pipelines,
            groups,
        })
    }
}

pub async fn run(session: &Session, input_data: HashMap<String, &[f32]>) -> Result<Vec<f32>> {
    let time_pre_run = Instant::now();
    let device = &session.device;
    let queue = &session.queue;
    let pipelines = &session.pipelines;
    let groups = &session.groups;
    let inner_infos = &session.inner_infos;
    for (input, data) in input_data {
        let n = data.len();
        let input_buffer =
            resource::create_buffer_init(device, data, &input, BufferUsages::COPY_SRC);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let buffer = inner_infos.get(&input).unwrap();
        encoder.copy_buffer_to_buffer(&input_buffer, 0, buffer, 0, (n * 4) as _);
        queue.submit(Some(encoder.finish()));
    }

    println!("time: pre_run: {:#?}", time_pre_run.elapsed());
    let graph = session.model.get_graph();
    let nodes = graph.get_node();
    let time_run = Instant::now();
    for ((node, pipeline), bind_groups) in nodes.iter().zip(pipelines).zip(groups) {
        compute::wrapper(device, queue, node, pipeline, bind_groups).unwrap();
    }

    let outputs = graph.get_output();
    let output_info = &graph.get_output();
    let output_dims = get_dimension(output_info, outputs[0].get_name()).unwrap();
    let buffer_exit = resource::buffer(
        device,
        len(&output_dims) as _,
        outputs[0].get_name(),
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let buffer = inner_infos.get(outputs[0].get_name()).unwrap();
    encoder.copy_buffer_to_buffer(buffer, 0, &buffer_exit, 0, (len(&output_dims) * 4) as _);
    queue.submit(Some(encoder.finish()));

    println!("time: run: {:#?}", time_run.elapsed());
    // ompute::compute(device, queue, graph, inner_infos, tera).unwrap();
    let time_post_run = Instant::now();
    let buffer_slice = buffer_exit.slice(..);
    // TODO: Define behavior for multi output.
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);
    // // OUTPUT

    buffer_future.await.expect("failed to run compute on gpu!");
    // Gets contents of buffer
    let data = buffer_slice.get_mapped_range();
    // Since contents are got in bytes, this converts these bytes back to f32
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    println!("time: post_run: {:#?}", time_post_run.elapsed());
    Ok(result)
}
