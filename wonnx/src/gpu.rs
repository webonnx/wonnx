//! Manages execution of shader code and buffer allocation on the GPU
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    convert::TryInto,
    sync::{mpsc, Arc},
};

use bytemuck::NoUninit;
use num::FromPrimitive;
use thiserror::Error;
use wgpu::{Buffer, BufferAsyncError, BufferUsages, CommandEncoder};

use crate::{
    compiler::{compile, CompileError, CompiledNode},
    ir::{Node, NodeDefinition, NodeIdentifier, OperatorDefinition},
    onnx::TensorProto,
    resource::{self, resize},
    utils::{
        ceil, DataTypeError, InputTensor, OutputTensor, ScalarType, Shape,
        MINIMUM_BUFFER_SIZE_BYTES,
    },
};

/// The maximum number of bindings in a binding group (defined by wgpu)
const MAX_BINDINGS_PER_GROUP: usize = 4;

pub struct GpuModel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    onnx_opset_version: i64,
    steps: Vec<GpuStep>,
    inference_outputs: HashMap<String, InferenceOutput>,
}

/// An operation that is performed on the GPU as part of inference
enum GpuStep {
    /// A statically, pre-filled buffer containing tensor data
    Initializer(Arc<Buffer>),

    /// A buffer containing tensor data that is obtained from inference input
    Input(String, Arc<Buffer>),

    /// A GPU program (shader) that reads from buffers created by other steps and writes to output buffers
    Operator {
        pipeline: wgpu::ComputePipeline,
        bind_groups: Vec<wgpu::BindGroup>,
        threads: (u32, u32, u32),
        output_tensors: Vec<GpuTensor>,
    },

    /// Operation that takes the output from a previous operation and assigns it to a second logical output
    Forward(GpuTensor),

    /// No-operation
    None,
}

/// A tensor that resides in GPU memory
#[derive(Clone)]
struct GpuTensor {
    buffer: Arc<Buffer>,
    shape: Shape,
}

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("compiling node '{node}' failed: {error}")]
    CompileError { node: String, error: CompileError },

    #[error("inference input not found: '{0}'")]
    InferenceInputMissing(String),

    #[error("node output not found: index {0}")]
    OutputMissing(usize),

    #[error("scalar type error: {0}")]
    ScalarType(#[from] DataTypeError),

    #[error("value out of bounds")]
    OutOfBoundsError,

    #[error("async buffer error: {0}")]
    BufferAsyncError(#[from] BufferAsyncError),
}

enum InferenceOutput {
    InferenceInput(String),
    Tensor(GpuTensor),
}

impl GpuModel {
    /// Create a version of the specified model for which inference can be performed using the powers of the GPU
    pub fn from(
        root: Arc<Node>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        onnx_opset_version: i64,
    ) -> Result<GpuModel, GpuError> {
        let mut gpu_model = GpuModel {
            device,
            queue,
            onnx_opset_version,
            steps: vec![],
            inference_outputs: HashMap::new(),
        };

        // Walk the IR DAG and encode into GPU execution steps
        let mut readable_nodes: HashSet<NodeIdentifier> = HashSet::new();
        let mut node_outputs = HashMap::<NodeIdentifier, Vec<GpuTensor>>::new();
        let mut nodes_seen = HashSet::new();
        gpu_model.sequence(
            root.clone(),
            &mut readable_nodes,
            &mut node_outputs,
            &mut nodes_seen,
        )?;

        // Find out which outputs we should return as inference outputs
        if let NodeDefinition::Outputs { names } = &root.definition {
            for (usize, output_name) in names.iter().enumerate() {
                let input = &root.inputs[usize];
                gpu_model.inference_outputs.insert(
                    output_name.to_string(),
                    match &input.source_node.definition {
                        NodeDefinition::Operator(_) | NodeDefinition::Tensor(_) => {
                            let source_identifier = input.source_node.identifier();
                            let outputs = &node_outputs[&source_identifier];
                            let tensor = outputs[input.output_index].clone();
                            InferenceOutput::Tensor(tensor)
                        }
                        NodeDefinition::Input(proto) => {
                            InferenceOutput::InferenceInput(proto.get_name().to_string())
                        }
                        NodeDefinition::Outputs { .. } => {
                            unimplemented!("output after output node")
                        }
                        NodeDefinition::Missing => {
                            unimplemented!("optional input after output node")
                        }
                    },
                );
            }
        } else {
            unimplemented!("reading from non-outputs IR node")
        }

        // Upload the data (for initializers etc.) by submitting an empty command queue
        log::info!("submit initializer buffers");
        let encoder = gpu_model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        gpu_model.queue.submit(Some(encoder.finish()));

        Ok(gpu_model)
    }

    /// Write commands to the GPU to create the necessary resources to be able to perform inference (e.g. allocates buffers
    /// for intermediate results, compiles shader code, determines which outputs to return, etc.).
    fn sequence<'model>(
        &mut self,
        node: Arc<Node<'model>>,
        nodes_readable: &mut HashSet<NodeIdentifier<'model>>,
        node_outputs: &mut HashMap<NodeIdentifier<'model>, Vec<GpuTensor>>,
        nodes_seen: &mut HashSet<NodeIdentifier<'model>>,
    ) -> Result<(), GpuError> {
        let node_identifier = node.identifier();
        let mut outputs_readable = nodes_readable.contains(&node_identifier);

        // Sequence inputs
        let mut input_tensors: Vec<GpuTensor> = vec![];
        for node_input in &node.inputs {
            let identifier = node_input.source_node.identifier();
            // If this node is an output node, mark input nodes as 'readable', meaning that their output buffers need to be created as readable buffers
            if let NodeDefinition::Outputs { .. } = &node.definition {
                nodes_readable.insert(identifier.clone());
                outputs_readable = true;
            }

            if outputs_readable {
                if let NodeDefinition::Operator(op_def) = &node.definition {
                    // For these ops we just forward the buffer (so we should also forward readability)
                    if op_forwards_input(op_def.proto.get_op_type()) {
                        nodes_readable.insert(identifier.clone());
                    }
                }
            }

            if !nodes_seen.contains(&identifier) {
                nodes_seen.insert(identifier.clone());
                // Sequence the source node
                self.sequence(
                    node_input.source_node.clone(),
                    nodes_readable,
                    node_outputs,
                    nodes_seen,
                )?;
            }

            // Select the tensor we want for our input
            let source_identifier = node_input.source_node.identifier();
            let input_tensor = {
                let source_outputs = &node_outputs[&source_identifier];
                source_outputs
                    .get(node_input.output_index)
                    .ok_or(GpuError::OutputMissing(node_input.output_index))?
                    .clone()
            };
            input_tensors.push(input_tensor);
        }

        // Sequence self
        if let std::collections::hash_map::Entry::Vacant(e) = node_outputs.entry(node_identifier) {
            log::info!(
                "sequence {:?} (outputs readable={:?})",
                node.definition,
                outputs_readable
            );

            let mut output_tensors = vec![];
            let gpu_op: GpuStep = match &node.definition {
                NodeDefinition::Operator(op_def) => {
                    let gpu_op = op_def.gpu_op(
                        &self.device,
                        outputs_readable,
                        self.onnx_opset_version,
                        &input_tensors,
                    )?;

                    match &gpu_op {
                        GpuStep::Operator {
                            output_tensors: op_output_tensors,
                            ..
                        } => {
                            output_tensors.extend(op_output_tensors.iter().cloned());
                        }
                        GpuStep::Forward(output_tensor) => {
                            output_tensors.push(output_tensor.clone());
                        }
                        _ => unreachable!("gpu_op for operator produced something unexpected"),
                    }

                    gpu_op
                }
                NodeDefinition::Tensor(tensor_def) => {
                    let tensor_buffer =
                        Arc::new(tensor_def.buffer(&self.device, outputs_readable)?);
                    output_tensors.push(GpuTensor {
                        shape: Shape::from(
                            ScalarType::from_i32(tensor_def.get_data_type())?,
                            tensor_def.get_dims(),
                        ),
                        buffer: tensor_buffer.clone(),
                    });
                    GpuStep::Initializer(tensor_buffer)
                }
                NodeDefinition::Input(input_def) => {
                    if outputs_readable {
                        log::warn!(
                            "it looks like you will be reading back inference input '{}' as output",
                            input_def.get_name()
                        );
                    }

                    let input_shape = input_def.get_shape()?;
                    log::info!(
                        "creating input buffer for {} shape {} size {}",
                        input_def.get_name(),
                        input_shape,
                        input_shape.buffer_bytes()
                    );
                    let input_buffer = Arc::new(resource::buffer(
                        &self.device,
                        input_shape.buffer_bytes(),
                        input_def.get_name(),
                        BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    ));

                    output_tensors.push(GpuTensor {
                        shape: input_shape,
                        buffer: input_buffer.clone(),
                    });

                    GpuStep::Input(input_def.get_name().to_string(), input_buffer)
                }
                NodeDefinition::Missing | NodeDefinition::Outputs { .. } => {
                    // Nothing to sequence
                    GpuStep::None
                }
            };

            e.insert(output_tensors);
            self.steps.push(gpu_op);
            Ok(())
        } else {
            // This node is already sequenced
            log::debug!("not sequencing (already seen) {:?}", node.definition);
            Ok(())
        }
    }

    /// Perform inference using this model and the specified inference inputs.
    pub async fn infer<'a>(
        &self,
        inference_inputs: &HashMap<String, InputTensor<'a>>,
    ) -> Result<HashMap<String, OutputTensor>, GpuError> {
        log::info!("encode inference steps");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for step in &self.steps {
            step.encode(&self.queue, &mut encoder, inference_inputs)?;
        }
        log::info!("submit inference steps");
        self.queue.submit(Some(encoder.finish()));
        log::info!("inference completed");
        self.read_outputs(inference_inputs).await
    }

    /// Reads the relevant buffers for the requested inference outputs
    async fn read_outputs<'a>(
        &self,
        inference_inputs: &HashMap<String, InputTensor<'a>>,
    ) -> Result<HashMap<String, OutputTensor>, GpuError> {
        let mut output_data: HashMap<String, OutputTensor> = HashMap::new();

        for (output_name, output_source) in &self.inference_outputs {
            output_data.insert(
                output_name.to_string(),
                match output_source {
                    InferenceOutput::InferenceInput(input_name) => {
                        (&inference_inputs[input_name]).into()
                    }
                    InferenceOutput::Tensor(tensor) => {
                        tensor.read_to_vec(&self.device, &self.queue).await?
                    }
                },
            );
        }

        Ok(output_data)
    }
}

trait TensorProtoExtra {
    fn buffer(&self, device: &wgpu::Device, readable: bool) -> Result<Buffer, GpuError>;
}

impl TensorProtoExtra for TensorProto {
    /// Create a GPU buffer containing the data of this initializer
    fn buffer(&self, device: &wgpu::Device, readable: bool) -> Result<Buffer, GpuError> {
        let scalar_type = ScalarType::from_i32(self.get_data_type())?;
        let input_shape = Shape::from(scalar_type, self.get_dims());
        log::info!(
            "creating buffer for tensor {} shape {}",
            self.get_name(),
            input_shape
        );

        match scalar_type {
            ScalarType::F32 => {
                let data = self.get_float_data();
                buffer_with_bytes(
                    device,
                    readable,
                    self.get_name(),
                    if !data.is_empty() {
                        bytemuck::cast_slice(data)
                    } else {
                        self.get_raw_data()
                    },
                )
            }
            ScalarType::I64 => {
                // WGSL doesn't support 64 bit integers, so we load 64 bit tensors as 32 bit ints
                log::warn!("initializers with int64 data type are not supported, converting into int32 initializer");
                let ints: Vec<i32> = self
                    .get_int64_data()
                    .iter()
                    .map(|x| (*x).try_into())
                    .collect::<Result<Vec<i32>, _>>()
                    .map_err(|_e| GpuError::OutOfBoundsError)?;
                let raw_data = bytemuck::cast_slice(&ints);
                buffer_with_bytes(device, readable, self.get_name(), raw_data)
            }
            ScalarType::I32 => {
                let data = self.get_int32_data();
                buffer_with_bytes(
                    device,
                    readable,
                    self.get_name(),
                    if !data.is_empty() {
                        bytemuck::cast_slice(data)
                    } else {
                        self.get_raw_data()
                    },
                )
            }
        }
    }
}

fn buffer_with_bytes(
    device: &wgpu::Device,
    readable: bool,
    name: &str,
    raw_data: &[u8],
) -> Result<Buffer, GpuError> {
    let buffer_usage = match readable {
        true => BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        false => BufferUsages::STORAGE,
    };

    // Do not create buffers that are too small
    Ok(if raw_data.len() < MINIMUM_BUFFER_SIZE_BYTES as _ {
        let mut larger_raw_data = raw_data.to_vec();
        larger_raw_data.resize(MINIMUM_BUFFER_SIZE_BYTES as _, 0);
        resource::create_buffer_init(device, &larger_raw_data, name, buffer_usage)
    } else {
        resource::create_buffer_init(device, raw_data, name, buffer_usage)
    })
}

/// Returns whether the op of the specified type will forward inputs unchanged. If this is the case, the inputs of such
/// an op should be marked as 'outputs readable' if the output of the op itself is to be readable.
fn op_forwards_input(op_type: &str) -> bool {
    matches!(
        op_type,
        "Reshape" | "Identity" | "Flatten" | "Squeeze" | "Unsqueeze" | "Dropout"
    )
}

impl<'model> OperatorDefinition<'model> {
    fn gpu_op(
        &self,
        device: &wgpu::Device,
        outputs_readable: bool,
        opset_version: i64,
        input_tensors: &[GpuTensor],
    ) -> Result<GpuStep, GpuError> {
        let proto = &self.proto;

        // Some nodes have specific GPU implementations, match these here
        if op_forwards_input(proto.get_op_type()) {
            // Some ops do nothing but forward their input
            let value_shape = &self.output_shapes[0];
            let output_tensor = GpuTensor {
                buffer: input_tensors[0].buffer.clone(),
                shape: value_shape.clone(),
            };
            return Ok(GpuStep::Forward(output_tensor));
        }

        let label = Some(proto.get_name());

        // Create output buffers for this op node
        let output_tensors: Vec<GpuTensor> = proto
            .get_output()
            .iter()
            .enumerate()
            .map(|(output_index, output_name)| {
                let value_shape = &self.output_shapes[output_index];
                log::info!(
                    "Creating op output buffer for output #{} ({}) of {} shaped {}",
                    output_index,
                    output_name,
                    proto.get_name(),
                    value_shape
                );

                let buffer_usage = if outputs_readable {
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC
                } else {
                    BufferUsages::STORAGE
                };

                let buffer = Arc::new(resource::buffer(
                    device,
                    value_shape.buffer_bytes(),
                    output_name.as_str(),
                    buffer_usage,
                ));
                GpuTensor {
                    buffer,
                    shape: value_shape.clone(),
                }
            })
            .collect();

        let input_shapes: Vec<&Shape> = input_tensors.iter().map(|input| &input.shape).collect();
        let output_shapes: Vec<&Shape> = self.output_shapes.iter().collect();

        // Compile shader for node
        let CompiledNode { shader, threads } =
            compile(proto, &input_shapes, &output_shapes, opset_version).map_err(|ce| {
                GpuError::CompileError {
                    node: proto.get_name().to_string(),
                    error: ce,
                }
            })?;
        log::debug!("shader: {}", shader);

        // Bind input and output buffers to the shader
        let mut binding_counter: usize = 0;
        let mut entries = vec![];

        // Bind input buffers
        for input in input_tensors {
            // Bindings are numbered 0...3 (MAX_BINDINGS_PER_GROUP-1) in binding groups (starting at group 0)
            let binding_index = (binding_counter % MAX_BINDINGS_PER_GROUP) as u32;

            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: input.buffer.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Bind output buffers
        for output_tensor in &output_tensors {
            // Bindings are numbered 0...3 (MAX_BINDINGS_PER_GROUP-1) in binding groups (starting at group 0)
            let binding_index = (binding_counter % MAX_BINDINGS_PER_GROUP) as u32;

            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: output_tensor.buffer.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Set up a pipeline (basically the shader source code with some metadata that determines how it will be executed)
        let mut bind_groups = vec![];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout: None,
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
            }),
            entry_point: "main",
        });

        // Create 'bind groups' (groups of bound buffers)
        let number_of_groups = ceil(binding_counter as u64, MAX_BINDINGS_PER_GROUP as u64) as usize;
        for group_index in 0..number_of_groups {
            let group_range = group_index * MAX_BINDINGS_PER_GROUP
                ..usize::min(
                    binding_counter as _,
                    (group_index + 1) * MAX_BINDINGS_PER_GROUP,
                );
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &pipeline.get_bind_group_layout(group_index as u32),
                entries: &entries[group_range],
            }));
        }

        Ok(GpuStep::Operator {
            output_tensors,
            pipeline,
            bind_groups,
            threads,
        })
    }
}

impl GpuStep {
    /// Writes the necessary commands for the GPU to execute this step into the command queue. Among other things this means
    /// writing the inference input data to the appropriate (already created) buffers.
    fn encode<'a>(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut CommandEncoder,
        inputs: &HashMap<String, InputTensor<'a>>,
    ) -> Result<(), GpuError> {
        match self {
            GpuStep::None | GpuStep::Forward(_) | GpuStep::Initializer(_) => {
                // Buffer already filled, no need to encode anything at this point.
                Ok(())
            }
            GpuStep::Input(input_name, input_buffer) => {
                // Encode a command to write the input data to the corresponding input buffer (which was created empty
                // by `GpuModel::from`
                let input_data = inputs
                    .get(input_name)
                    .ok_or_else(|| GpuError::InferenceInputMissing(input_name.to_string()))?;
                log::info!("- write input data for {}", input_name);

                match input_data {
                    InputTensor::F32(float_input) => {
                        queue.write_buffer(
                            input_buffer,
                            0,
                            bytemuck::cast_slice(&resize(float_input.to_vec())),
                        );
                    }
                    InputTensor::I32(int_input) => {
                        queue.write_buffer(
                            input_buffer,
                            0,
                            bytemuck::cast_slice(&resize(int_input.to_vec())),
                        );
                    }
                    InputTensor::I64(int_input) => {
                        log::warn!("reading int64 input as int32 (int64 is not supported for calculation but can be used as input as long as values fit in int32)");
                        let int32_input = int_input
                            .iter()
                            .map(|i| i32::from_i64(*i).ok_or(GpuError::OutOfBoundsError))
                            .collect::<Result<_, _>>()?;
                        queue.write_buffer(
                            input_buffer,
                            0,
                            bytemuck::cast_slice(&resize(int32_input)),
                        );
                    }
                }

                Ok(())
            }
            GpuStep::Operator {
                pipeline,
                bind_groups,
                threads,
                ..
            } => {
                // Encode a command for invocation of a shader.
                let mut compute_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                compute_pass.set_pipeline(pipeline);
                for (index, bind_group) in bind_groups.iter().enumerate() {
                    compute_pass.set_bind_group(index as u32, bind_group, &[]);
                }
                let (x, y, z) = *threads;
                compute_pass.dispatch_workgroups(x, y, z);
                Ok(())
            }
        }
    }
}

impl GpuTensor {
    /// Read the tensor from GPU memory to main memory (as Vec<f32>)
    async fn read_to_vec(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<OutputTensor, GpuError> {
        let buffer_slice = self.buffer.slice(..);
        let shape = self.shape.clone();

        let (tx, rx) = mpsc::sync_channel(1);

        wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer_slice, move |buffer| {
            // Called on download completed
            tx.send(match buffer {
                Ok(bytes) => Ok(Self::read_bytes_to_vec(&bytes, shape)),
                Err(error) => Err(GpuError::BufferAsyncError(error)),
            })
            .unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        // The callback will have been called by now due to poll(Wait)
        rx.recv().unwrap()
    }

    fn read_bytes_to_vec<A>(output_data: &[A], shape: Shape) -> OutputTensor
    where
        A: NoUninit,
    {
        // The actual buffer may be bigger than what we should return, because buffers have a minimum size in wgpu
        // Fetch the size we should expect so we can chop the buffer to the correct size
        let output_buffer_size = shape.element_count() as usize;
        match shape.data_type {
            ScalarType::F32 => {
                OutputTensor::F32(bytemuck::cast_slice(output_data)[..output_buffer_size].to_vec())
            }
            ScalarType::I32 => {
                OutputTensor::I32(bytemuck::cast_slice(output_data)[..output_buffer_size].to_vec())
            }
            ScalarType::I64 => {
                log::warn!("reading int64 output as int32 because internally int64 scalars are not supported");
                let result_ints: Vec<i32> =
                    bytemuck::cast_slice(output_data)[..output_buffer_size].to_vec();
                OutputTensor::I64(result_ints.iter().map(|i| *i as i64).collect())
            }
        }
    }
}
