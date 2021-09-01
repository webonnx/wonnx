// importing common module.
use bytemuck;
use wonnx::*;

#[test]
fn test_add() {
    let (device, queue) = pollster::block_on(ressource::request_device_queue());
    let data = [1.0, 2.0, 3.0, 4.0];
    let buffer = ressource::create_buffer_init(&device, &data);

    let binding_group_entry = [wgpu::BindGroupEntry {
        binding: 0,
        resource: buffer.as_entire_binding(),
    }];

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let operators = ["cos", "sin", "cosh", "sinh"];
    let computors: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)> = operators
        .iter()
        .map(|op| compute::wrapper(&device, &binding_group_entry, &[0], &op::map(op)))
        .collect();
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        for (cp, bg) in computors.iter() {
            cpass.set_pipeline(&cp);
            cpass.set_bind_group(0, &bg, &[]);
            // cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch(4, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
    }
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);
    if let Ok(()) = pollster::block_on(buffer_future) {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to f32
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        println!("result: {:#?}", result);
        drop(data);
    } else {
        panic!("failed to run compute on gpu!")
    }
}
