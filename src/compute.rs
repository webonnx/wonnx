use log::{debug, info};
use serde_derive::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use tera::{Context, Tera};

#[derive(Serialize)]
struct Bindings {
    counter: u32,
    tensor: String,
}

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
    tera: &Tera,
) -> Result<(), wgpu::Error> {
    let mut binding_counter: u32 = 0;
    // Generating the shader

    let mut context = Context::new();
    let inputs = node.get_input();
    let outputs = node.get_output();

    let inputs = if ["Reshape", "Clip", "Squeeze"].contains(&node.get_op_type()) {
        inputs.get(0..1).unwrap()
    } else {
        inputs
    };
    // Generating the shader
    let mut entries = vec![];
    let mut bindings = vec![];

    for tensor in inputs {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap_or_else(|| panic!("Tensor {} is not present in the inner infos", tensor))
                .buffer
                .as_entire_binding(),
        });
        bindings.push(Bindings {
            counter: binding_counter,
            tensor: tensor.to_string(),
        });
        binding_counter += 1;
    }

    for tensor in outputs {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap()
                .buffer
                .as_entire_binding(),
        });
        bindings.push(Bindings {
            counter: binding_counter,
            tensor: tensor.to_string(),
        });
        binding_counter += 1;
        debug!(
            "output {} has size: {:?} at counter {}",
            tensor,
            inner_infos.get(tensor.as_str()).unwrap().dims,
            binding_counter
        );
    }
    context.insert("bindings", &bindings);

    // TODO: Add attribute value binding
    let (shader_template, x, y, z) = crate::compiler::format_node(node, inner_infos, &mut context);

    let shader = tera
        .render(&shader_template, &context)
        .expect("failed to render shader");

    debug!("shader: {}", shader);
    // debug!("x: {}", x);
    // TODO: Make defining threads more clean.
    println!("node.get_name(): {:#?}", node.get_name());
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
            layout: &compute_pipeline.get_bind_group_layout(0 as u32),
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
            for index in 0..(binding_counter / 4) as usize {
                cpass.set_bind_group(index as u32, &bind_groups[index], &[]);
            }
        } else {
            cpass.set_bind_group(0 as u32, &bind_groups[0], &[]);
        }
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
