use crate::{
    compiler::format_node,
    resource,
    sequencer::sequence,
    utils::{ceil, dimensions_infos, get_dimension, initializers, len},
    Result,
};

use std::{borrow::Cow, collections::HashMap};

use log::debug;
use tera::Tera;
use wgpu::BufferUsages;

const MAX_OPTIMIZATION_LEN: usize = 7;

pub struct EncoderBuilder {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub threads: (u32, u32, u32),
}

pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(HashMap<String, wgpu::Buffer>, Vec<EncoderBuilder>)> {
    let tera = match Tera::new("templates/**/*.wgsl") {
        Ok(t) => t,
        Err(e) => {
            panic!("Parsing error(s): {}", e);
        }
    };

    let mut initializers = initializers(graph);
    let dims_info = dimensions_infos(graph);

    let base_nodes = graph.get_node();

    let mut inner_infos = HashMap::new();

    let input_info = graph.get_input();
    for input in input_info {
        let input_dims = get_dimension(input_info, input.get_name()).unwrap();
        inner_infos.insert(
            input.get_name().to_string(),
            resource::buffer(
                device,
                len(&input_dims) as _,
                input.get_name(),
                BufferUsages::STORAGE | BufferUsages::MAP_WRITE,
            ),
        );
    }

    let n = base_nodes.len();

    let mut node_index = 0;

    let mut builders = vec![];

    let output_info = &graph.get_output().to_vec();

    while node_index < n {
        let nodes = &base_nodes[node_index..(usize::min(node_index + MAX_OPTIMIZATION_LEN, n))];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<&str>>();
        let (current_node, optimisation_length) =
            sequence(&names, nodes, device, &mut initializers, &mut inner_infos);
        let (shader, x, y, z) = format_node(&current_node, &dims_info, &tera);
        debug!("shader: {}", shader);

        // Initalialising Output
        let output = &nodes[optimisation_length - 1].get_output()[0];
        if let Some(output_dims) = dims_info.get(output) {
            if output_info
                .iter()
                .any(|el| el.get_name() == output.as_str())
            {
                inner_infos.insert(
                    output.clone(),
                    resource::buffer(
                        device,
                        len(output_dims) as _,
                        output.as_str(),
                        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    ),
                );
            } else {
                inner_infos.insert(
                    output.clone(),
                    resource::buffer(
                        device,
                        len(output_dims) as _,
                        output.as_str(),
                        BufferUsages::STORAGE,
                    ),
                );
            }
        } else {
            panic!("output dims was not provided. You can use python's onnx-simplifier to generate implied dimensions.")
        }

        let mut binding_counter: u32 = 0;

        let inputs = current_node.get_input();
        let inputs = if ["Reshape", "Clip", "Squeeze"].contains(&current_node.get_op_type()) {
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
                    .unwrap_or_else(|| {
                        panic!("Tensor {} is not present in the inner infos", tensor)
                    })
                    .as_entire_binding(),
            });
            binding_counter += 1;
        }

        for tensor in current_node.get_output() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: inner_infos
                    .get(tensor.as_str())
                    .unwrap_or_else(|| {
                        panic!("Tensor {} is not present in the inner infos", tensor)
                    })
                    .as_entire_binding(),
            });
            binding_counter += 1;
        }

        let mut bind_groups = vec![];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
            }),
            entry_point: "main",
        });
        for index in 0..ceil(binding_counter.into(), 4) as usize {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(index as u32),
                entries: &entries[index * 4..usize::min(binding_counter as _, (index + 1) * 4)],
            }));
        }
        builders.push(EncoderBuilder {
            pipeline,
            bind_groups,
            threads: (x, y, z),
        });

        node_index += optimisation_length;
    }

    Ok((inner_infos, builders))
}
