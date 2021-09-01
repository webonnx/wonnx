use wonnx::*;

fn main() {
    const n: i32 = i32::pow(2, 20);
    let (device, queue) = pollster::block_on(ressource::request_device_queue());
    let data = [1.0; n as _];
    let buffer = ressource::create_buffer_init(&device, &data);

    let binding_group_entry = [wgpu::BindGroupEntry {
        binding: 0,
        resource: buffer.as_entire_binding(),
    }];
    let mut i = 1;
    let mut v = Vec::new();
    while i < n {
        v.push(crate::compute::wrapper(
            &device,
            &binding_group_entry,
            &[0],
            &crate::op::scan(&[1.0; 2], &[i], true),
        ));
        i *= 2;
    }
    let mut i = n;
    {
        for (cp, bg) in v.iter() {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                i /= 2;
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&cp);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch(i as _, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
            }
            queue.submit(Some(encoder.finish()));
        }
        // cpass.insert_debug_marker("compute collatz iterations");
    }

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

        println!("result: {:#?}", &result[0]);
        drop(data);
    } else {
        panic!("failed to run compute on gpu!")
    }
}
