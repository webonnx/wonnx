use crate::{
    compiler::compile,
    resource,
    sequencer::sequence,
    utils::{ceil, dimensions_infos, initializers, len},
    Result,
};

use std::{borrow::Cow, collections::HashMap};

use log::debug;
use tera::Tera;
use wgpu::BufferUsages;

pub struct EncoderBuilder {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub threads: (u32, u32, u32),
}

lazy_static! {
    pub static ref TEMPLATES: Tera = {
        let mut tera = Tera::default();
        tera.add_raw_template(
            "endomorphism/activation.wgsl",
            include_str!("../templates/endomorphism/activation.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/arithmetic.wgsl",
            include_str!("../templates/endomorphism/arithmetic.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/batchnormalization.wgsl",
            include_str!("../templates/endomorphism/batchnormalization.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/copy.wgsl",
            include_str!("../templates/endomorphism/copy.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/map.wgsl",
            include_str!("../templates/endomorphism/map.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/concat.wgsl",
            include_str!("../templates/matrix/concat.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm_1.wgsl",
            include_str!("../templates/matrix/gemm_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm.wgsl",
            include_str!("../templates/matrix/gemm.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/transpose.wgsl",
            include_str!("../templates/matrix/transpose.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/aggregate.wgsl",
            include_str!("../templates/pool/aggregate.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_1.wgsl",
            include_str!("../templates/pool/conv_kernel_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_3.wgsl",
            include_str!("../templates/pool/conv_kernel_3.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv.wgsl",
            include_str!("../templates/pool/conv.wgsl"),
        )
        .unwrap();
        tera.add_raw_template("structs.wgsl", include_str!("../templates/structs.wgsl"))
            .unwrap();
        tera
    };
}

pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(HashMap<String, wgpu::Buffer>, Vec<EncoderBuilder>)> {
    let initializers = initializers(graph);
    let dims_info = dimensions_infos(graph);

    let mut inner_infos = HashMap::new();

    for (input_name, input_dims) in dims_info.iter() {
        inner_infos.insert(
            input_name.clone(),
            resource::buffer(
                device,
                len(input_dims) as _,
                input_name,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
        );
    }

    let base_nodes = graph.get_node();
    let n = base_nodes.len();

    let mut node_index = 0;

    let mut builders = vec![];

    let output_info = &graph.get_output().to_vec();

    while node_index < n {
        let nodes = &base_nodes[node_index..];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<_>>();

        let (current_node, optimisation_length) =
            sequence(&names, nodes, device, &initializers, &mut inner_infos);
        let (shader, x, y, z) = compile(&current_node, &dims_info, &TEMPLATES);
        debug!("shader: {}", shader);

        // Initalialising Output
        let output = &current_node.get_output()[0];
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
                        BufferUsages::STORAGE | BufferUsages::MAP_READ,
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
        let mut entries = vec![];

        for tensor in current_node.get_input() {
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
