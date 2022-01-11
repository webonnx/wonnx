use crate::{
    compiler::{compile, CompileError, CompiledNode},
    resource,
    sequencer::{sequence, SequenceError},
    utils::{ceil, dimensions_infos, initializers},
    Result,
};

use std::{borrow::Cow, collections::HashMap};

use log::info;
use thiserror::Error;
use wgpu::BufferUsages;

pub struct EncoderBuilder {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub threads: (u32, u32, u32),
}

const MAX_BINDINGS_PER_GROUP: usize = 4;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("compilation failed: {0}")]
    CompilationFailed(#[from] CompileError),

    #[error("output shape was not provided. You can use python's onnx-simplifier to generate implied shapes.")]
    OutputShapeMissing,

    #[error("tensor metadata missing: '{0}'")]
    TensorMetadataMissing(String),

    #[error("sequencing failed: {0}")]
    SequencingFailed(#[from] SequenceError),
}

pub struct OptimizedModel {
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub builders: Vec<EncoderBuilder>,
}

pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
    opset_version: i64,
) -> Result<OptimizedModel, OptimizationError> {
    let initializers = initializers(graph);
    let mut shapes_info = dimensions_infos(graph);
    let mut buffers = HashMap::new();

    for (input_name, input_shape) in shapes_info.iter() {
        buffers.insert(
            input_name.clone(),
            resource::buffer(
                device,
                input_shape.buffer_len() as _,
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
        let sequence = sequence(
            &names,
            nodes,
            device,
            &initializers,
            &mut buffers,
            &mut shapes_info,
        )?;

        let current_node = sequence.node;
        let CompiledNode { shader, threads } = compile(&current_node, &shapes_info, opset_version)?;
        info!("shader: {}", shader);

        // Create buffers for all outputs of this node
        for output in current_node.get_output().iter() {
            if let Some(output_shapes) = shapes_info.get(output) {
                if output_info
                    .iter()
                    .any(|el| el.get_name() == output.as_str())
                {
                    buffers.insert(
                        output.clone(),
                        resource::buffer(
                            device,
                            output_shapes.buffer_len() as _,
                            output.as_str(),
                            BufferUsages::STORAGE | BufferUsages::MAP_READ,
                        ),
                    );
                } else {
                    buffers.insert(
                        output.clone(),
                        resource::buffer(
                            device,
                            output_shapes.buffer_len() as _,
                            output.as_str(),
                            BufferUsages::STORAGE,
                        ),
                    );
                }
            } else {
                return Err(OptimizationError::OutputShapeMissing);
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
        let number_of_groups = ceil(binding_counter as u64, MAX_BINDINGS_PER_GROUP as u64) as usize;
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
        node_index += sequence.nodes_consumed;
    }

    Ok(OptimizedModel { buffers, builders })
}
