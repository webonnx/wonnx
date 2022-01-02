use crate::{
    compiler::{compile, CompileError, CompiledNode},
    resource,
    sequencer::{sequence, SequenceError},
    utils::{buffer_len, ceil, dimensions_infos, initializers},
    Result,
};

use std::{borrow::Cow, collections::HashMap};

use log::info;
use tera::Tera;
use thiserror::Error;
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
            "endomorphism/softmax.wgsl",
            include_str!("../templates/endomorphism/softmax.wgsl"),
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
            "matrix/resize.wgsl",
            include_str!("../templates/matrix/resize.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/split.wgsl",
            include_str!("../templates/matrix/split.wgsl"),
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
        tera.add_raw_template(
            "snippets/activation_vec.wgsl",
            include_str!("../templates/snippets/activation_vec.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "snippets/activation_scalar.wgsl",
            include_str!("../templates/snippets/activation_scalar.wgsl"),
        )
        .unwrap();
        tera
    };
}

const MAX_BINDINGS_PER_GROUP: usize = 4;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("compilation failed")]
    CompilationFailed(#[from] CompileError),

    #[error("output dims was not provided. You can use python's onnx-simplifier to generate implied dimensions.")]
    OutputDimsMissing,

    #[error("tensor metadata missing: '{0}'")]
    TensorMetadataMissing(String),

    #[error("sequencing failed")]
    SequencingFailed(#[from] SequenceError),
}

pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(HashMap<String, wgpu::Buffer>, Vec<EncoderBuilder>), OptimizationError> {
    let initializers = initializers(graph);
    let dims_info = dimensions_infos(graph);

    let mut buffers = HashMap::new();

    for (input_name, input_dims) in dims_info.iter() {
        buffers.insert(
            input_name.clone(),
            resource::buffer(
                device,
                buffer_len(input_dims) as _,
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

        // Generate the shader source code for this node
        let (current_node, optimisation_length) =
            sequence(&names, nodes, device, &initializers, &mut buffers)?;
        let CompiledNode { shader, threads } = compile(&current_node, &dims_info, &TEMPLATES)?;
        info!("shader: {}", shader);

        // Create buffers for all outputs of this node
        for output in current_node.get_output().iter() {
            if let Some(output_dims) = dims_info.get(output) {
                if output_info
                    .iter()
                    .any(|el| el.get_name() == output.as_str())
                {
                    buffers.insert(
                        output.clone(),
                        resource::buffer(
                            device,
                            buffer_len(output_dims) as _,
                            output.as_str(),
                            BufferUsages::STORAGE | BufferUsages::MAP_READ,
                        ),
                    );
                } else {
                    buffers.insert(
                        output.clone(),
                        resource::buffer(
                            device,
                            buffer_len(output_dims) as _,
                            output.as_str(),
                            BufferUsages::STORAGE,
                        ),
                    );
                }
            } else {
                return Err(OptimizationError::OutputDimsMissing);
            }
        }

        // Find all the buffers we want to use as inputs and outputs for this node. These will be 'bound' to variables in the shader
        let mut binding_counter: usize = 0;
        let mut entries = vec![];

        for tensor in current_node.get_input() {
            // Bindings are numbered 0...3 (MAX_BINDINGS_PER_GROUP-1) in binding groups (starting at group 0)
            let binding_index = (binding_counter % MAX_BINDINGS_PER_GROUP) as u32;
            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: buffers
                    .get(tensor.as_str())
                    .ok_or_else(|| OptimizationError::TensorMetadataMissing(tensor.to_string()))?
                    .as_entire_binding(),
            });
            binding_counter += 1;
        }

        for tensor in current_node.get_output() {
            // Bindings are numbered 0...3 (MAX_BINDINGS_PER_GROUP-1) in binding groups (starting at group 0)
            let binding_index = (binding_counter % MAX_BINDINGS_PER_GROUP) as u32;
            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: buffers
                    .get(tensor.as_str())
                    .ok_or_else(|| OptimizationError::TensorMetadataMissing(tensor.to_string()))?
                    .as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Compile the shader source code for this node
        let mut bind_groups = vec![];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(current_node.get_name()),
            layout: None,
            module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some(current_node.get_name()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
            }),
            entry_point: "main",
        });

        // Perform the binding for each group
        let number_of_groups = ceil(binding_counter as i64, MAX_BINDINGS_PER_GROUP as i64) as usize;
        for group_index in 0..number_of_groups {
            let group_range = group_index * MAX_BINDINGS_PER_GROUP
                ..usize::min(
                    binding_counter as _,
                    (group_index + 1) * MAX_BINDINGS_PER_GROUP,
                );
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(current_node.get_name()),
                layout: &pipeline.get_bind_group_layout(group_index as u32),
                entries: &entries[group_range],
            }));
        }

        builders.push(EncoderBuilder {
            pipeline,
            bind_groups,
            threads,
        });

        node_index += optimisation_length;
    }

    Ok((buffers, builders))
}
