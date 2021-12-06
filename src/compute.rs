use std::borrow::Cow;
use std::collections::HashMap;

use crate::utils::get_attribute;

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, wgpu::Buffer>,
) -> Result<(), wgpu::Error> {
    let mut binding_counter: u32 = 0;
    // Generating the shader

    let inputs = node.get_input();
    let outputs = node.get_output();

    let inputs = if ["Reshape", "Clip", "Squeeze"].contains(&node.get_op_type()) {
        inputs.get(0..1).unwrap()
    } else {
        inputs
    };

    // Generating the shader
    let mut entries = vec![];

    for tensor in inputs {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap_or_else(|| panic!("Tensor {} is not present in the inner infos", tensor))
                .as_entire_binding(),
        });
        binding_counter += 1;
    }

    for tensor in outputs {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap_or_else(|| panic!("Tensor {} is not present in the inner infos", tensor))
                .as_entire_binding(),
        });
        binding_counter += 1;
    }

    let shader = get_attribute::<String>("WGSL", None, node);
    let threads = get_attribute::<Vec<i64>>("threads", None, node);
    let x = threads[0];
    let y = threads[1];
    let z = threads[2];
    // debug!("x: {}", x);
    // TODO: Make defining threads more clean.
    // Generating the compute pipeline and binding group.
    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&(node.get_name().to_string() + "_pipeline")),
        layout: None,
        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&(node.get_name().to_string() + "_shader")),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        }),
        entry_point: "main",
    });
    let mut bind_groups = vec![];
    if binding_counter / 4 > 0 {
        for index in 0..(binding_counter / 4) as usize {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &compute_pipeline.get_bind_group_layout(index as u32),
                entries: &entries[index * 4..(index + 1) * 4],
            }));
        }
    } else {
        bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_pipeline.get_bind_group_layout(0u32),
            entries: &entries,
        }));
    }
    // Instantiates the bind group, once again specifying the binding of buffers.

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&(node.get_name().to_string() + "_encoder")),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&(node.get_name().to_string() + "_pass")),
        });
        cpass.set_pipeline(&compute_pipeline);
        if binding_counter / 4 > 0 {
            for (index, bind_group) in bind_groups
                .iter()
                .enumerate()
                .take((binding_counter / 4) as usize)
            {
                cpass.set_bind_group(index as u32, bind_group, &[]);
            }
        } else {
            cpass.set_bind_group(0u32, &bind_groups[0], &[]);
        }
        cpass.dispatch(x as u32, y as u32, z as u32); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
