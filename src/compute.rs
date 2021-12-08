use crate::utils::get_attribute;

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    node: &crate::onnx::NodeProto,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &Vec<wgpu::BindGroup>,
) -> Result<(), wgpu::Error> {
    // Generating the shader

    let threads = get_attribute::<Vec<i64>>("threads", None, node);
    let x = threads[0];
    let y = threads[1];
    let z = threads[2];
    // debug!("x: {}", x);
    // TODO: Make defining threads more clean.
    // Generating the compute pipeline and binding group.
    // Instantiates the pipeline.

    // Instantiates the bind group, once again specifying the binding of buffers.

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&(node.get_name().to_string() + "_encoder")),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&(node.get_name().to_string() + "_pass")),
        });
        cpass.set_pipeline(&pipeline);
        for (index, bind_group) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(index as u32, bind_group, &[]);
        }
        cpass.dispatch(x as u32, y as u32, z as u32); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
