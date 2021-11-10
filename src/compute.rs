use crate::{onnx, utils::node};
use log::debug;
use serde_derive::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use tera::{Context, Tera};

const MAX_OPTIMIZATION_LEN: usize = 2;
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

    // Generating the compute pipeline and binding group.

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        }),
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}

pub fn compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &crate::onnx::GraphProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
    tera: &Tera,
) -> Result<(), wgpu::Error> {
    let original_nodes = graph.get_node();
    let n = original_nodes.iter().count();

    let mut node_index = 0;

    while node_index < n {
        let nodes = &original_nodes[node_index..(usize::min(node_index + MAX_OPTIMIZATION_LEN, n))];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<&str>>();
        let runnable_node = match names.as_slice() {
            ["Conv", "Relu", ..] => {
                node_index += 2;
                node(
                    nodes[0].get_input().iter().map(|x| x.as_str()).collect(),
                    nodes[1].get_output().iter().map(|x| x.as_str()).collect(),
                    "ConvRelu",
                    "ConvRelu",
                    nodes[0].get_attribute().to_vec(),
                )
            }
            [..] => {
                node_index += 1;
                nodes[0].clone()
            }
        };

        crate::compute::wrapper(device, queue, &runnable_node, inner_infos, tera).unwrap();
    }

    Ok(())
}
