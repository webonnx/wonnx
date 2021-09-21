use std::time::Instant;
use wgpu::util::DeviceExt;
use wonnx::{compute::InnerType, *};
fn main() {
    let time_0 = Instant::now();
    let (device, queue) = pollster::block_on(resource::request_device_queue());
    let time = Instant::now();
    println!("time - time_0: {:#?}", time - time_0);
    let a: Vec<f32> = (0..50_176).map(|x| x as f32).collect();
    let b = vec![4.0; 50_176];

    let buffer_a = resource::read_only_buffer(&device, &a);
    let buffer_b = resource::read_only_buffer(&device, &b);
    let buffer_c = resource::create_buffer(&device, 50_176);
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 50_176,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let binding_group_entry = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer_a.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: buffer_b.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: buffer_c.as_entire_binding(),
        },
    ];
    let time_shader = Instant::now();
    println!("time_shader - time: {:#?}", time_shader - time);
    crate::compute::wrapper(
        &device,
        &queue,
        &binding_group_entry,
        &[InnerType::Array, InnerType::Array, InnerType::Array],
        &crate::op::matmul(224),
        224,
        224,
        1,
    )
    .unwrap();
    println!("1: {:#?}", 1);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    println!("aa");
    encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, 50_176);
    println!("aa: {:#?}", 1);
    queue.submit(Some(encoder.finish()));
    // cpass.insert_debug_marker("compute collatz iterations");
    let time_resource = Instant::now();
    println!(
        "time_resource - time_shader: {:#?}",
        time_resource - time_shader
    );
    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);
    if let Ok(()) = pollster::block_on(buffer_future) {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to f32
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        println!(
            "Instant::now() - time: {:#?}",
            Instant::now() - time_resource
        );
        println!("result: {:#?}", &result[0]);
        drop(data);
    } else {
        panic!("failed to run compute on gpu!")
    }
}
