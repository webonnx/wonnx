//! Manages execution of shader code and buffer allocation on the GPU
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{HashMap, HashSet},
    convert::TryInto,
    ops::Sub,
    sync::Arc,
};

use bytemuck::NoUninit;
use num::FromPrimitive;
use thiserror::Error;
use wgpu::{Buffer, BufferAsyncError, BufferUsages, CommandEncoder, Device};

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

/// The buffer manager tracks the use of buffers and manages recycling (sharing) of buffers for intermediate values that
/// are passed from node to node. It can be used to assign a `LeasableBuffer` to each node output in the graph. This
/// works as follows: in `GpuModel::pre_sequence`, the model is traversed starting from the output node. As soon as a node
/// needs to use the output from another node (that executes earlier in the graph), the `pre_sequence` function calls `lease`
/// on the buffer manager. This indicates the end of the 'lifetime' of the buffer for that intermediate buffer (if another
/// node also uses the same node output, a second call to `lease` will be made, but this will have no effect as the lifetime
/// cannot be shortened). When the node that actually produces the intermediate value is encountered, `release` is called
/// on the buffer manager, indicating that the buffer is now free to be re-used for nodes that execute before this node.
/// When assigning buffers, the buffer manager takes care to assign a buffer of a matching size whenever possible.
struct BufferManager<'a> {
    assignments: HashMap<Output<'a>, Arc<RefCell<LeaseableBuffer>>>,
    free: Vec<Arc<RefCell<LeaseableBuffer>>>,

    /// Counter for assigning buffer IDs (only available in debug)
    #[cfg(debug_assertions)]
    buffer_id_counter: usize,

    /// The total size of individual buffer requests (only tracked in debug builds)
    #[cfg(debug_assertions)]
    total_unshared_bytes: usize,
}

type Output<'model> = (NodeIdentifier<'model>, usize);

/// A `LeaseableBuffer` is a buffer that may be re-used multiple times during graph execution. Each time a buffer of a
/// different size may be needed; the shared buffer takes the largest size requested.
#[derive(Debug)]
struct LeaseableBuffer {
    #[cfg(debug_assertions)]
    id: usize,
    largest_size: usize,
    buffer: Option<Arc<Buffer>>,
}

impl LeaseableBuffer {
    fn allocated_on(&mut self, device: &Device) -> Arc<Buffer> {
        match self.buffer {
            Some(ref b) => {
                // TODO check buffer belongs to the same device, otherwise panic
                b.clone()
            }
            None => {
                log::debug!("allocating shared intermediate buffer {self:?}");

                #[cfg(debug_assertions)]
                let name = format!("Shared_{}", self.id);

                #[cfg(not(debug_assertions))]
                let name = String::from("shared_intermediate");

                let b = Arc::new(resource::buffer(
                    device,
                    self.largest_size,
                    &name,
                    BufferUsages::STORAGE,
                ));
                self.buffer = Some(b.clone());
                b
            }
        }
    }
}

impl<'a> BufferManager<'a> {
    fn new() -> BufferManager<'a> {
        BufferManager {
            #[cfg(debug_assertions)]
            buffer_id_counter: 0,
            assignments: HashMap::new(),
            free: vec![],

            #[cfg(debug_assertions)]
            total_unshared_bytes: 0,
        }
    }

    /// Tells the buffer manager that the node producing a value for a certain shared buffer has been encountered. Hence
    /// nodes executing before it may re-use the buffer as long as it is unused again when this node executes.
    fn release(
        &mut self,
        node_identifier: NodeIdentifier<'a>,
        output_index: usize,
        output_size_bytes: usize,
    ) {
        if let Some(buffer) = self
            .assignments
            .get(&(node_identifier.clone(), output_index))
        {
            let mut buffer_raw = buffer.borrow_mut();
            log::debug!("shared intermediate buffer: free {buffer_raw:?} size={output_size_bytes}");

            // The buffer may not already be released
            assert!(
                !self.free.iter().any(|x| x.as_ptr() == buffer.as_ptr()),
                "shared intermediate buffer lease already released"
            );

            // Only track statistics while debugging
            #[cfg(debug_assertions)]
            {
                self.total_unshared_bytes += output_size_bytes;
            }

            buffer_raw.largest_size = buffer_raw.largest_size.max(output_size_bytes);
            drop(buffer_raw);
            self.free.push(buffer.clone());
        } else {
            log::debug!("shared intermediate buffer: unused: {node_identifier:?}@{output_index}");
        }
    }

    /// Allocates a new leaseable shared buffer
    fn new_buffer(&mut self, size: usize) -> Arc<RefCell<LeaseableBuffer>> {
        #[cfg(debug_assertions)]
        let id = {
            let x = self.buffer_id_counter;
            self.buffer_id_counter += 1;
            x
        };
        Arc::new(RefCell::new(LeaseableBuffer {
            #[cfg(debug_assertions)]
            id,
            largest_size: size,
            buffer: None,
        }))
    }

    /// Returns a buffer that can be (re-)used. If there are multiple buffers on the free list, the function will look for
    /// the buffer that has a `largest_size` that is the closest match to the requested size. If there is just one buffer
    /// on the free list, it will be returned. If the free list is empty, the function will create a new leaseable buffer.
    fn new_or_free_buffer(&mut self, output_bytes: usize) -> Arc<RefCell<LeaseableBuffer>> {
        // TODO remove when usize::abs_diff is stabilized / MSRV is raised
        fn abs_difference<T: Sub<Output = T> + Ord>(x: T, y: T) -> T {
            if x < y {
                y - x
            } else {
                x - y
            }
        }

        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, b) in self.free.iter().enumerate() {
            let rb = b.as_ref().borrow();
            match closest_size_diff {
                None => {
                    closest_index = Some(idx);
                    closest_size_diff = Some(abs_difference(rb.largest_size, output_bytes))
                }
                Some(d) if d > abs_difference(rb.largest_size, output_bytes) => {
                    closest_index = Some(idx);
                    closest_size_diff = Some(abs_difference(rb.largest_size, output_bytes))
                }
                _ => {}
            }
        }

        match closest_index {
            Some(idx) => self.free.remove(idx),
            None => self.new_buffer(output_bytes),
        }
    }

    /// Tells the buffer manager that a certain value is used by a node and that nodes executing before it may not re-use
    /// the same buffer until a node is encountered that actually producdes a value in the buffer. As outputs can be used
    /// by multiple nodes, `lease` can be called multiple times (but only one call to `release` is necessary).
    fn lease(
        &mut self,
        node_identifier: NodeIdentifier<'a>,
        output_index: usize,
        output_bytes: usize,
    ) {
        let output = (node_identifier, output_index);
        #[allow(clippy::map_entry)]
        if !self.assignments.contains_key(&output) {
            let ac = self.new_or_free_buffer(output_bytes);
            log::debug!("intermediate buffer lease {output:?} => {:?}", ac);
            self.assignments.insert(output, ac);
        }
    }

    /// Returns the total size of shared buffers currently assigned
    #[cfg(debug_assertions)]
    fn total_shared_size(&self) -> usize {
        use std::iter::FromIterator;
        let unique_buffers: HashSet<(usize, usize)> = HashSet::from_iter(
            self.assignments
                .values()
                .map(|x| (x.as_ref().borrow().id, x.as_ref().borrow().largest_size)),
        );
        unique_buffers.iter().map(|x| x.1).sum()
    }

    /// Returns the total size of shared buffers currently assigned
    #[cfg(debug_assertions)]
    fn shared_buffer_count(&self) -> usize {
        use std::iter::FromIterator;
        let unique_buffers: HashSet<usize> =
            HashSet::from_iter(self.assignments.values().map(|x| x.as_ref().borrow().id));
        unique_buffers.len()
    }
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

        let mut buffer_manager = BufferManager::new();

        let mut nodes = vec![];
        let mut nodes_seen = HashSet::new();
        GpuModel::topological_sort(root.clone(), &mut nodes_seen, &mut nodes);
        drop(nodes_seen);
        GpuModel::pre_sequence(&nodes, &mut readable_nodes, &mut buffer_manager)?;

        #[cfg(debug_assertions)]
        {
            let total_shared_size = buffer_manager.total_shared_size();
            log::info!(
                "shared intermediate value buffers: {} buffers, total unshared size is {}b, total shared size is {total_shared_size}b, improvement {}x",
                buffer_manager.shared_buffer_count(),
                buffer_manager.total_unshared_bytes,
               (buffer_manager.total_unshared_bytes as f64) / (total_shared_size as f64)
            );
        }

        let mut nodes_seen = HashSet::new();
        gpu_model.sequence(
            root.clone(),
            &readable_nodes,
            &mut node_outputs,
            &mut nodes_seen,
            &mut buffer_manager,
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
        log::debug!("submit initializer buffers");
        let encoder = gpu_model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        gpu_model.queue.submit(Some(encoder.finish()));

        Ok(gpu_model)
    }

    /// Traverse the graph and sort nodes in the order of execution (topological sort)
    fn topological_sort<'model>(
        node: Arc<Node<'model>>,
        nodes_seen: &mut HashSet<NodeIdentifier<'model>>,
        sorted_nodes: &mut Vec<Arc<Node<'model>>>,
    ) {
        let identifier = node.identifier();
        if !nodes_seen.contains(&identifier) {
            nodes_seen.insert(identifier);
            for node_input in &node.inputs {
                GpuModel::topological_sort(
                    node_input.source_node.clone(),
                    nodes_seen,
                    sorted_nodes,
                );
            }
            sorted_nodes.push(node);
        }
    }

    /// Run a first pass over the IR graph to determine the outputs of which nodes are supposed to be readable as outputs
    /// of the graph after inference. This needs to be done in a separate pass because otherwise we may run into an issue
    /// where nodes are not marked as 'outputs readable' when their outputs are used by some node while also being used as
    /// output (and the sequenceer might simply follow one 'path' before the other). This pass is also used to assign buffers
    /// for intermediate values (which may be re-used). For this reason, a 'breadth first'-search is performed (whereas
    /// `GpuModel::sequence` will perform a depth-first search).
    fn pre_sequence<'model>(
        nodes: &[Arc<Node<'model>>],
        nodes_readable: &mut HashSet<NodeIdentifier<'model>>,
        buffer_manager: &mut BufferManager<'model>,
    ) -> Result<(), GpuError> {
        for node in nodes.iter().rev() {
            let node_identifier = node.identifier();
            let mut outputs_readable = nodes_readable.contains(&node_identifier);

            for node_input in &node.inputs {
                // Tell the buffer manager that we are consuming an intermediate result produced by some earlier op
                if let NodeDefinition::Operator(ref source_node_def) =
                    node_input.source_node.definition
                {
                    // If the input node forwards a buffer, we need to take a lease on *its* input
                    let mut ultimate_input = node_input.clone();
                    while let NodeDefinition::Operator(ultimate_input_op_def) =
                        ultimate_input.source_node.definition()
                    {
                        if op_forwards_input(ultimate_input_op_def.proto.get_op_type()) {
                            assert_eq!(ultimate_input.source_node.inputs.len(), 1);
                            ultimate_input = ultimate_input.source_node.inputs[0].clone();
                        } else {
                            break;
                        }
                    }

                    let output_shape = &source_node_def.output_shapes[node_input.output_index];
                    buffer_manager.lease(
                        ultimate_input.source_node.identifier(),
                        ultimate_input.output_index,
                        output_shape.buffer_bytes(),
                    );
                }

                let source_node_identifier = node_input.source_node.identifier();
                // If this node is an output node, mark input nodes as 'readable', meaning that their output buffers need to be created as readable buffers
                if let NodeDefinition::Outputs { .. } = &node.definition {
                    nodes_readable.insert(source_node_identifier.clone());
                    outputs_readable = true;
                }

                if outputs_readable {
                    if let NodeDefinition::Operator(op_def) = &node.definition {
                        // For these ops we just forward the buffer (so we should also forward readability)
                        if op_forwards_input(op_def.proto.get_op_type()) {
                            nodes_readable.insert(source_node_identifier.clone());
                        }
                    }
                }
            }

            // Tell the buffer manager we are producing an intermediate value; nodes that run 'before' us may reuse this buffer
            if let NodeDefinition::Operator(op_def) = &node.definition {
                if !op_forwards_input(op_def.proto.get_op_type()) {
                    for (output_index, output_shape) in op_def.output_shapes.iter().enumerate() {
                        buffer_manager.release(
                            node_identifier.clone(),
                            output_index,
                            output_shape.buffer_bytes(),
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Write out the GPU commands and create the necessary resources to be able to perform inference (e.g. allocates buffers
    /// for intermediate results, compiles shader code, determines which outputs to return, etc.).
    fn sequence<'model>(
        &mut self,
        node: Arc<Node<'model>>,
        nodes_readable: &HashSet<NodeIdentifier<'model>>,
        node_outputs: &mut HashMap<NodeIdentifier<'model>, Vec<GpuTensor>>,
        nodes_seen: &mut HashSet<NodeIdentifier<'model>>,
        buffer_manager: &mut BufferManager<'model>,
    ) -> Result<(), GpuError> {
        let node_identifier = node.identifier();
        let outputs_readable = nodes_readable.contains(&node_identifier);

        // Sequence inputs of this node first (recursively)
        let mut input_tensors: Vec<GpuTensor> = vec![];
        for node_input in &node.inputs {
            let identifier = node_input.source_node.identifier();

            // If we haven't seen this input node yet, sequence it now
            if !nodes_seen.contains(&identifier) {
                nodes_seen.insert(identifier.clone());

                // Sequence the source node
                self.sequence(
                    node_input.source_node.clone(),
                    nodes_readable,
                    node_outputs,
                    nodes_seen,
                    buffer_manager,
                )?;
            }

            // Select the tensor we want for our input from the outputs created during sequencing
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

        // Sequence self (if by now we haven't yet)
        if let std::collections::hash_map::Entry::Vacant(e) = node_outputs.entry(node_identifier) {
            log::debug!(
                "sequence {:?} (outputs readable={:?})",
                node.definition,
                outputs_readable
            );

            // Create output tensors for all outputs of this node
            let mut output_tensors = vec![];
            let gpu_op: GpuStep = match &node.definition {
                // If this node is an operator, the outputs can either be the output of the operation itself, or it can be
                // outputs forwarded from the (only) input node of this operation (if the operation itself only modifies
                // metadata, e.g. shapes, or is a no-op).
                NodeDefinition::Operator(op_def) => {
                    // Can we use shared buffers for outputs of this node?
                    let shared_buffers: Vec<Option<Arc<RefCell<LeaseableBuffer>>>> =
                        (0..op_def.output_shapes.len())
                            .map(|output_index| {
                                let identifier = node.identifier();
                                buffer_manager
                                    .assignments
                                    .get(&(identifier, output_index))
                                    .cloned()
                            })
                            .collect();

                    let gpu_op = op_def.gpu_op(
                        &self.device,
                        outputs_readable,
                        self.onnx_opset_version,
                        &input_tensors,
                        &shared_buffers,
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
                // For tensor (initializer) nodes, we just create a buffer and fill it with the initializer data
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
                // For inputs we create an empty buffer that can be used at inference time to supply input data
                NodeDefinition::Input(input_def) => {
                    if outputs_readable {
                        log::warn!(
                            "it looks like you will be reading back inference input '{}' as output",
                            input_def.get_name()
                        );
                    }

                    let input_shape = input_def.get_shape()?;
                    log::debug!(
                        "creating input buffer for {} shape {} size {}",
                        input_def.get_name(),
                        input_shape,
                        input_shape.buffer_bytes()
                    );
                    let input_buffer = Arc::new(resource::buffer(
                        &self.device,
                        input_shape.buffer_bytes(),
                        input_def.get_name(),
                        // Usage is not COPY_SRC/MAP_READ even when outputs_readable is true; we'll deal with the special
                        // case of reading back inputs as outputs separately.
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
        log::debug!("submit inference steps");
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
        log::debug!(
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
            ScalarType::U8 => {
                // WGSL doesn't support 8 bit unsigned integers, so we load them as 32 bit ints
                log::warn!("initializers with uint8 data type are not supported, converting into int32 initializer");
                let ints: Vec<i32> = self
                    .get_raw_data()
                    .iter()
                    .map(|x| (*x).try_into())
                    .collect::<Result<Vec<i32>, _>>()
                    .map_err(|_e| GpuError::OutOfBoundsError)?;
                let raw_data = bytemuck::cast_slice(&ints);
                buffer_with_bytes(device, readable, self.get_name(), raw_data)
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
        shared_buffers: &[Option<Arc<RefCell<LeaseableBuffer>>>],
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

                let buffer = match shared_buffers.get(output_index) {
                    Some(Some(shared_buffer)) if !outputs_readable => {
                        let mut shared_buffer = shared_buffer.borrow_mut();
                        shared_buffer.allocated_on(device)
                    }
                    _ => {
                        log::debug!(
                            "creating non-shared buffer for output #{} ({}) of {} shaped {}",
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

                        Arc::new(resource::buffer(
                            device,
                            value_shape.buffer_bytes(),
                            output_name.as_str(),
                            buffer_usage,
                        ))
                    }
                };

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
                    node: if proto.has_name() {
                        proto.get_name().to_string()
                    } else {
                        proto.get_op_type().to_string()
                    },
                    error: ce,
                }
            })?;
        log::trace!("shader: {}", shader);

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
                log::debug!("write input data for {}", input_name);

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
                        log::warn!("reading int64 input '{input_name}' as int32 (int64 is not supported for calculation but can be used as input as long as values fit in int32)");
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
                    InputTensor::U8(int_input) => {
                        log::warn!("reading uint8 input as int32 (uint8 is not supported for calculation but can be used as input)");
                        let int32_input = int_input
                            .iter()
                            .map(|i| i32::from_u8(*i).ok_or(GpuError::OutOfBoundsError))
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
        let shape = self.shape.clone();

        #[cfg(target_arch = "wasm32")]
        {
            let buffer_slice = self.buffer.slice(..);
            let (sender, receiver) =
                futures::channel::oneshot::channel::<Result<OutputTensor, GpuError>>();

            wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer_slice, move |buffer| {
                // Called on download completed
                log::debug!(
                    "downloadbuffer read_buffer callback res={:?}",
                    buffer.is_ok()
                );
                sender
                    .send(match buffer {
                        Ok(bytes) => Ok(Self::read_bytes_to_vec(&bytes, shape)),
                        Err(error) => Err(GpuError::BufferAsyncError(error)),
                    })
                    .unwrap();
            });

            receiver.await.unwrap()
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let buffer_slice = self.buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);

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
            ScalarType::U8 => {
                OutputTensor::U8(bytemuck::cast_slice(output_data)[..output_buffer_size].to_vec())
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
