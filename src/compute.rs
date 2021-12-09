use crate::optimisation::EncoderBuilder;

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    builder: &EncoderBuilder,
) -> Result<(), wgpu::Error> {
    // Generating the shader

    let (x, y, z) = builder.threads;
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&builder.pipeline);
        for (index, bind_group) in builder.bind_groups.iter().enumerate() {
            cpass.set_bind_group(index as u32, bind_group, &[]);
        }
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
