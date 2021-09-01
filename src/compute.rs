#[cfg(not(test))]
use log::debug;
use std::borrow::Cow;

#[cfg(test)]
use std::println as debug;

pub fn wrapper(
    device: &wgpu::Device,
    buffers: &[wgpu::BindGroupEntry],
    bindings: &[i32],
    main: &str,
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    // Generating the shader
    let mut shader = crate::boilerplate::INIT.to_string();

    for i in bindings {
        shader.push_str(
            format!(
                r#"
[[group(0), binding({i})]]
var<storage, read_write> b_{i}: Array;

"#,
                i = i
            )
            .as_str(),
        )
    }

    shader.push_str(&format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
{main}    
}}
"#,
        main = main
    ));

    debug!("shader: {}", shader);

    // Generating the compute pipeline and binding group.
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
    });

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(bindings[0] as _);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: buffers,
    });

    (compute_pipeline, bind_group)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_map() {
        let (device, queue) = pollster::block_on(crate::ressource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let buffer = crate::ressource::create_buffer_init(&device, &data);

        let binding_group_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        };
        let (cp, bg) = crate::compute::wrapper(
            &device,
            &[binding_group_entry],
            &[0],
            &crate::op::map(&"cos"),
        );
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            cpass.set_pipeline(&cp);
            cpass.set_bind_group(0, &bg, &[]);
            // cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch(4, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        queue.submit(Some(encoder.finish()));
        let buffer_slice = buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        device.poll(wgpu::Maintain::Wait);
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            let expected = vec![f32::cos(1.0), f32::cos(2.), f32::cos(3.), f32::cos(4.)];

            for (res, exp) in result.iter().zip(expected) {
                assert!((res - exp) < f32::EPSILON)
            }
            drop(data);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    #[test]
    fn test_conv() {
        let (device, queue) = pollster::block_on(crate::ressource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let buffer = crate::ressource::create_buffer_init(&device, &data);
        let output = crate::ressource::create_buffer(&device, 4);
        let binding_group_entry = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ];

        let (cp, bg) = crate::compute::wrapper(
            &device,
            &binding_group_entry,
            &[0, 1],
            &crate::op::conv(&[1.0, 2.0], &[1], false),
        );
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            cpass.set_pipeline(&cp);
            cpass.set_bind_group(0, &bg, &[]);
            // cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch(4, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        queue.submit(Some(encoder.finish()));
        let buffer_slice = output.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        device.poll(wgpu::Maintain::Wait);
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            let expected = vec![5.0, 11.0];

            for (res, exp) in result.iter().zip(expected) {
                assert!((res - exp) < f32::EPSILON)
            }
            drop(data);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
