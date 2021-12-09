use crate::{
    resource,
    sequencer::sequence,
    utils::attribute,
    utils::{ceil, get_dimension, len},
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

    let mut initializers = HashMap::new();
    for initializer in graph.get_initializer() {
        let input = initializer.get_name().to_string();
        let data = initializer.get_float_data();
        let raw_data = if !data.is_empty() {
            bytemuck::cast_slice(data)
        } else {
            initializer.get_raw_data()
        };

        initializers.insert(input, raw_data);
    }
    let mut value_info = HashMap::new();

    for info in graph.get_input() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph.get_output() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph.get_value_info() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    let base_nodes = graph.get_node();

    let mut inner_infos = HashMap::new();

    let inputs = graph.get_input();
    let input_info = &graph.get_input();
    for input in inputs {
        let input_dims = get_dimension(input_info, input.get_name()).unwrap();
        inner_infos.insert(
            input.get_name().to_string(),
            resource::buffer(
                device,
                len(&input_dims) as _,
                input.get_name(),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
        );
    }

    let n = base_nodes.len();

    let mut node_index = 0;

    let mut optimised_nodes = vec![];
    let mut builders = vec![];

    let output_info = &graph.get_output().to_vec();
    let output_names: Vec<&str> = output_info.iter().map(|output| output.get_name()).collect();

    while node_index < n {
        let nodes = &base_nodes[node_index..(usize::min(node_index + MAX_OPTIMIZATION_LEN, n))];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<&str>>();
        let (mut current_node, optimisation_length) =
            sequence(&names, nodes, device, &mut initializers, &mut inner_infos);

        // Exit node if necessary.
        // Initalialising Output
        let output = &nodes[optimisation_length - 1].get_output()[0];
        if let Some(output_dims) = value_info.get(output) {
            if output_names.contains(&output.as_str()) {
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

        let (shader, x, y, z) = crate::compiler::format_node(&current_node, &value_info, &tera);
        debug!("shader: {}", shader);
        let attributes = current_node.mut_attribute();
        attributes.push(attribute("WGSL", shader.clone()));
        attributes.push(attribute("threads", vec![x as i64, y as i64, z as i64]));

        let mut binding_counter: u32 = 0;
        // Generating the shader

        let inputs = current_node.get_input();
        let outputs = current_node.get_output();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
            }),
            entry_point: "main",
        });

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

        for tensor in outputs {
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

        // debug!("x: {}", x);
        // TODO: Make defining threads more clean.
        // Generating the compute pipeline and binding group.
        // Instantiates the pipeline.

        let mut bind_groups = vec![];
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
        // Instantiates the bind group, once again specifying the binding of buffers.
        optimised_nodes.push(current_node);

        node_index += optimisation_length;
    }

    Ok((inner_infos, builders))
}
