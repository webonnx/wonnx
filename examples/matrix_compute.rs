use std::time::Instant;
use wgpu::util::DeviceExt;
use wonnx::*;

fn main() {
    let time_0 = Instant::now();
    let (device, queue) = pollster::block_on(ressource::request_device_queue());
    let time = Instant::now();
    println!("time - time_0: {:#?}", time - time_0);

    let n = 1024;
    let n4 = n / 4;
    let n2 = n * n;
    let a: Vec<f32> = (0..n2).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..n2).map(|x| x as f32).collect();
    let c: Vec<f32> = (0..n2).map(|x| x as f32).collect();

    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&c),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    });

    let time_shader = Instant::now();
    println!("time_shader - time: {:#?}", time_shader - time);
    crate::compute::wrapper(
        &device,
        &queue,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_c.as_entire_binding(),
            },
        ],
        &[
            compute::InnerType::ArrayMatrix,
            compute::InnerType::ArrayMatrix,
            compute::InnerType::ArrayMatrix,
        ],
        &crate::op::vector_matmul(n4),
        n4 as u32,
        n4 as u32,
        1,
    )
    .unwrap();

    // cpass.insert_debug_marker("compute collatz iterations");
    let time_ressource = Instant::now();
    println!(
        "time_ressource - time_shader: {:#?}",
        time_ressource - time_shader
    );
    // Note that we're not calling `.await` here.
    let buffer_slice = buffer_c.slice(..);
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
            Instant::now() - time_ressource
        );
        println!("result: {:#?}", &result[0]);
        drop(data);
    } else {
        panic!("failed to run compute on gpu!")
    }
}
