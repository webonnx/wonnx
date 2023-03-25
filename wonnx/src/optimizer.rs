//! Optimizer that walks the DAG and transforms or coalesces ops for quicker execution
use crate::{
    gpu::GpuModel,
    ir::{Input, Node, NodeDefinition, NodeIdentifier, OperatorDefinition},
    onnx::{NodeProto, TensorProto},
    resource::{padding, request_device_queue},
    utils::{
        attribute, AttributeNotFoundError, DataTypeError, NodeAttributes, OutputTensor, ScalarType,
        Shape,
    },
    GpuError,
};
use async_recursion::async_recursion;
use protobuf::RepeatedField;
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("node has no inputs")]
    NoInputs,

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("invalid data type {data_type:?} for input {input} of op {op}")]
    InvalidInputDataType {
        data_type: ScalarType,
        input: String,
        op: String,
    },

    #[error("error with data type: {0}")]
    InvalidDataType(#[from] DataTypeError),

    #[error("node is invalid: {0}")]
    InvalidNode(String),

    #[error("required attribute not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),

    #[error("error during constant folding: {0}")]
    ConstantFoldingError(#[from] GpuError),
}

pub struct Optimizer<'model> {
    padded_tensors: HashMap<String, Arc<Node<'model>>>,
    optimized: HashMap<NodeIdentifier<'model>, Arc<Node<'model>>>,
    onnx_opset_version: i64,
}

impl<'model> Optimizer<'model> {
    pub fn new(onnx_opset_version: i64) -> Self {
        Self {
            padded_tensors: HashMap::new(),
            optimized: HashMap::new(),
            onnx_opset_version,
        }
    }

    // Calculates the output of a constant node, then returns a node that contains the result as initializer
    async fn fold_constant_node(
        &self,
        node: Arc<Node<'model>>,
    ) -> Result<Option<Arc<Node<'model>>>, OptimizerError> {
        assert!(node.is_constant());

        match node.definition() {
            NodeDefinition::Operator(op_def) => {
                // TODO: constant nodes with multiple outputs
                if op_def.proto.output.len() != 1 {
                    log::warn!(
                        "node {:?} is constant, but has multiple outputs, which we can't fold yet",
                        node.definition()
                    );
                    return Ok(None);
                }

                match op_def.proto.get_op_type() {
                    "Constant" => Ok(Some(Arc::new(Node {
                        definition: NodeDefinition::Tensor(Box::new(Cow::Owned(
                            Self::constant_node_to_tensor(node)?,
                        ))),
                        inputs: vec![],
                    }))),
                    _ => self.infer_constant_node_to_tensor(node.clone()).await,
                }
            }
            NodeDefinition::Tensor(_) => Ok(None), // already constantized
            NodeDefinition::Input(_) | NodeDefinition::Missing => unreachable!(),
            NodeDefinition::Outputs { .. } => Ok(None), // all the outputs themselves are already constant, so nothing to do
        }
    }

    // Takes a node with operator type 'Constant' and returns its output as a tensor
    fn shape_node_to_tensor(node: Arc<Node<'model>>) -> Result<TensorProto, OptimizerError> {
        let NodeDefinition::Operator(op_def) = node.definition() else {
            panic!("node must be a Shape node");
        };
        assert_eq!(op_def.proto.get_op_type(), "Shape");

        if node.inputs.len() != 1 {
            return Err(OptimizerError::InvalidNode(format!(
                "Shape node should only have one input, has {}",
                node.inputs.len()
            )));
        }

        // Determine the shape of the input
        let input = &node.inputs[0];
        let in_node = &input.source_node.definition;
        let in_shape = match in_node {
            NodeDefinition::Input(input) => input.get_shape()?,
            NodeDefinition::Operator(input_op_def) => {
                input_op_def.output_shapes[input.output_index].clone()
            }
            NodeDefinition::Tensor(input_tensor) => Shape::from(
                ScalarType::from_i32(input_tensor.get_data_type())
                    .map_err(OptimizerError::InvalidDataType)?,
                input_tensor.get_dims(),
            ),
            NodeDefinition::Outputs { .. } => {
                return Err(OptimizerError::Unsupported(
                    "output node cannot be used as an input to Shape node".to_string(),
                ))
            }
            NodeDefinition::Missing => {
                return Err(OptimizerError::InvalidNode(
                    "Shape node has missing input".to_string(),
                ))
            }
        };
        let rank = in_shape.rank() as i64;
        let mut start: i64 = op_def.proto.get_attribute_value("start", Some(0)).unwrap();
        let mut end: i64 = op_def.proto.get_attribute_value("end", Some(rank)).unwrap();
        if start < 0 {
            start += rank;
        }
        if end < 0 {
            end += rank;
        }
        start = start.clamp(0, rank);
        end = end.clamp(0, rank);

        if start < 0 || start > rank {
            return Err(OptimizerError::InvalidNode(format!(
                "start index of Shape node cannot be below zero, found {start}"
            )));
        }

        if end < 0 || end > rank || end < start {
            return Err(OptimizerError::InvalidNode(format!(
                "end index of Shape node cannot be below zero or higher than {rank} or below start {start}, found {end}"
            )));
        }

        let values: Vec<i64> = in_shape.dims[(start as usize)..=((end - 1) as usize)]
            .iter()
            .map(|x| *x as i64)
            .collect();
        let dims = vec![values.len() as i64];
        Ok(TensorProto::from(OutputTensor::I64(values), dims))
    }

    // Takes a node with operator type 'Constant' and returns its output as a tensor
    fn constant_node_to_tensor(node: Arc<Node<'model>>) -> Result<TensorProto, OptimizerError> {
        let NodeDefinition::Operator(op_def) = node.definition() else {
            panic!("node must be a Constant node");
        };
        assert_eq!(op_def.proto.get_op_type(), "Constant");
        let proto = &op_def.proto;
        let output_name = proto.output.get(0).unwrap().to_owned();

        let mut tp: TensorProto =
            if let Ok(values) = proto.get_attribute_value::<Vec<f32>>("value_floats", None) {
                let dims = vec![values.len() as i64];
                TensorProto::from(OutputTensor::F32(values), dims)
            } else if let Ok(values) = proto.get_attribute_value::<Vec<i64>>("value_ints", None) {
                let dims = vec![values.len() as i64];
                TensorProto::from(OutputTensor::I64(values), dims)
            } else if let Ok(value) = proto.get_attribute_value::<f32>("value_float", None) {
                TensorProto::from(OutputTensor::F32(vec![value]), vec![1])
            } else if let Ok(value) = proto.get_attribute_value::<i64>("value_int", None) {
                TensorProto::from(OutputTensor::I64(vec![value]), vec![1])
            } else if let Ok(tp) = proto.get_attribute_value::<TensorProto>("value", None) {
                tp
            } else {
                return Err(OptimizerError::Unsupported(
                    "Constant node with unknown value type".to_string(),
                ));
            };

        tp.set_name(output_name);
        Ok(tp)
    }

    // Infers the output for a constant node (must be a constant and operator node, or the function panics)
    async fn infer_constant_node_to_tensor(
        &self,
        node: Arc<Node<'model>>,
    ) -> Result<Option<Arc<Node<'model>>>, OptimizerError> {
        assert!(node.is_constant());

        // Create an output node so we can perform inference for this node
        if let NodeDefinition::Operator(op_def) = node.definition() {
            let output_name = op_def.proto.output.get(0).unwrap().to_owned();

            let out_node = Arc::new(Node {
                definition: NodeDefinition::Outputs {
                    names: vec!["output".to_string()],
                },
                inputs: vec![Input {
                    source_node: node.clone(),
                    output_index: 0,
                }],
            });

            // Perform inference
            let (device, queue) = request_device_queue().await;
            let gm = GpuModel::from(out_node, device, queue, self.onnx_opset_version)
                .map_err(OptimizerError::ConstantFoldingError)?;
            let mut outputs = gm.infer(&HashMap::new()).await?;

            // Take the output tensor and make it into an initializer node
            let (_, output_tensor) = outputs.drain().take(1).next().unwrap();
            log::info!("folded {output_name} to {output_tensor:?}");
            let mut output_tensor_proto = TensorProto::from(
                output_tensor,
                op_def.output_shapes[0]
                    .dims
                    .iter()
                    .map(|x| *x as i64)
                    .collect(),
            );
            output_tensor_proto.set_name(output_name);

            let tensor_node = Node {
                definition: NodeDefinition::Tensor(Box::new(Cow::Owned(output_tensor_proto))),
                inputs: vec![],
            };

            Ok(Some(Arc::new(tensor_node)))
        } else {
            panic!("node to fold must be operator")
        }
    }

    /// Optimize a branch of a graph (memoized)
    #[async_recursion]
    pub async fn optimize(
        &mut self,
        node: Arc<Node<'model>>,
    ) -> Result<Arc<Node<'model>>, OptimizerError> {
        let identifier = node.identifier();
        match self.optimized.get(&identifier) {
            Some(opt_node) => Ok(opt_node.clone()),
            None => {
                let opt_node = self.optimize_actual(node).await?;
                self.optimized.insert(identifier, opt_node.clone());
                Ok(opt_node)
            }
        }
    }

    /// Optimize a branch of a graph. Takes a node an attempts to form a chain of nodes with single (dynamic) inputs by
    /// traversing towards the inputs.
    #[async_recursion]
    async fn optimize_actual(
        &mut self,
        node: Arc<Node<'model>>,
    ) -> Result<Arc<Node<'model>>, OptimizerError> {
        // Try to form a chain of nodes that have one dynamic input
        let prior;
        let mut chain = VecDeque::new();
        chain.push_back(node.clone());

        loop {
            let head = chain.front().unwrap();
            let dynamic_inputs = head
                .inputs
                .iter()
                .filter(|input| input.source_node.is_dynamic() && input.output_index == 0)
                .collect::<Vec<&Input>>();

            if dynamic_inputs.len() != 1 {
                prior = chain.pop_front().unwrap();
                break;
            }
            chain.push_front(dynamic_inputs[0].source_node.clone());
        }

        log::debug!(
            "optimize: node={:?} def={:?} chain={}, next={:?}",
            node.identifier(),
            node.definition,
            chain
                .iter()
                .map(|x| format!("[{:?}]", x.definition))
                .collect::<Vec<String>>()
                .join(" -> "),
            prior.identifier()
        );

        // Try to simplify this chain of nodes
        if chain.len() > 1 {
            let mut final_chain: Vec<Arc<Node>> = vec![];
            while !chain.is_empty() {
                log::debug!("optimize chain {}", chain.len());
                while self.optimize_chain(&mut chain)? {
                    log::debug!("optimize chain succeeded {}", chain.len());
                }

                if !chain.is_empty() {
                    // Now pop off the first item and make it final
                    let first = chain.pop_front().unwrap();
                    final_chain.push(first);
                }

                log::debug!(
                    "optimized chain: {}",
                    final_chain
                        .iter()
                        .map(|x| format!("[{:?}]", x.definition))
                        .collect::<Vec<String>>()
                        .join(" -> ")
                );
            }
            drop(chain);

            // optimize next node
            let optimized_next = self.optimize(prior).await?;

            if final_chain.is_empty() {
                return Ok(optimized_next);
            }

            // Fix up the connections between these nodes
            for node_index in 0..=(final_chain.len() - 1) {
                let consumer = final_chain[node_index].clone();
                let producer = if node_index == 0 {
                    optimized_next.clone()
                } else {
                    final_chain[node_index - 1].clone()
                };
                final_chain[node_index] = self
                    .locally_optimized_node_with(
                        consumer.clone(),
                        consumer
                            .inputs
                            .iter()
                            .map(|old_input| {
                                // Each node is guaranteed to have only one 'dynamic' input. This is the one we will replace
                                let is_dynamic_source = old_input.source_node.is_dynamic()
                                    && old_input.output_index == 0;
                                if is_dynamic_source {
                                    Input {
                                        source_node: producer.clone(),
                                        output_index: 0,
                                    }
                                } else {
                                    old_input.clone()
                                }
                            })
                            .collect(),
                    )
                    .await?;
            }

            Ok(final_chain.last().unwrap().clone())
        } else {
            // Just optimize this nodes' inputs recursively
            let mut new_inputs = Vec::with_capacity(node.inputs.len());
            for input in node.inputs.iter() {
                new_inputs.push(Input {
                    source_node: self.optimize(input.source_node.clone()).await?,
                    output_index: input.output_index,
                });
            }
            self.locally_optimized_node_with(node.clone(), new_inputs)
                .await
        }
    }

    /// Create a new node from an existing definition, applying optimizations local to a single node
    async fn locally_optimized_node_with(
        &mut self,
        node: Arc<Node<'model>>,
        mut new_inputs: Vec<Input<'model>>,
    ) -> Result<Arc<Node<'model>>, OptimizerError> {
        log::debug!(
            "locally_optimized_node_with {:?} {:?}",
            node.identifier(),
            node.definition()
        );

        // Fold Shape node (not considered constant but we can still fold it)
        if let NodeDefinition::Operator(op_def) = &node.definition {
            if op_def.proto.get_op_type() == "Shape" {
                return Ok(Arc::new(Node {
                    definition: NodeDefinition::Tensor(Box::new(Cow::Owned(
                        Self::shape_node_to_tensor(node)?,
                    ))),
                    inputs: vec![],
                }));
            }
        }

        // Fold constant nodes
        if node.is_constant() {
            log::debug!(
                "node is constant: {:?} {:?}",
                node.identifier(),
                node.definition()
            );
            if let Some(const_node) = self.fold_constant_node(node.clone()).await? {
                return Ok(const_node);
            }
        }

        match &node.definition {
            NodeDefinition::Operator(op_def) => {
                match op_def.proto.get_op_type() {
                    "Conv" | "ConvRelu" | "ConvLeakyRelu" => {
                        // This optimization inserts some padding to convolution between kernels with kernel 3x3, because of
                        // the stride of matrix3x3 is 16 in wgsl. It makes the computation matrixable and increases the performance.
                        if new_inputs.len() > 2
                            && op_def
                                .proto
                                .get_attribute_value::<Vec<i64>>("kernel_shape", None)?
                                == [3, 3]
                            && (op_def
                                .proto
                                .get_attribute_value("pads", Some(vec![0, 0, 0, 0]))?
                                == [1, 1, 1, 1]
                                || op_def.proto.get_attribute_value(
                                    "auto_pad",
                                    Some("SAME_UPPER".to_string()),
                                )? == "SAME_UPPER")
                            && op_def
                                .proto
                                .get_attribute_value("strides", Some(vec![1, 1]))?
                                == [1, 1]
                            && op_def.output_shapes[0].dim(1) % 4 == 0
                        {
                            if let NodeDefinition::Tensor(tensor) =
                                &new_inputs[1].source_node.definition
                            {
                                new_inputs[1] = Input {
                                    output_index: 0,
                                    source_node: match self.padded_tensors.get(tensor.get_name()) {
                                        Some(padded_tensor_node) => padded_tensor_node.clone(),
                                        None => {
                                            let data = tensor.get_float_data();
                                            let raw_data = if !data.is_empty() {
                                                bytemuck::cast_slice(data)
                                            } else {
                                                tensor.get_raw_data()
                                            };

                                            let padded_raw_data = padding(raw_data, 12, 4);

                                            log::info!(
                                                "applying padding optimization to tensor {}: strides data is {} bytes before, {} bytes after",
                                                tensor.get_name(),
                                                raw_data.len(),
                                                padded_raw_data.len()
                                            );

                                            // Create a new tensor with the padded data
                                            let mut new_tensor = tensor.clone().into_owned();
                                            new_tensor.set_float_data(vec![]);
                                            new_tensor.set_raw_data(padded_raw_data);
                                            let new_node = Arc::new(Node {
                                                definition: NodeDefinition::Tensor(Box::new(
                                                    Cow::Owned(new_tensor),
                                                )),
                                                inputs: vec![],
                                            });
                                            self.padded_tensors.insert(
                                                tensor.get_name().to_string(),
                                                new_node.clone(),
                                            );
                                            new_node
                                        }
                                    },
                                }
                            }
                        }

                        let new_node = Node {
                            inputs: new_inputs,
                            definition: NodeDefinition::Operator(op_def.clone()),
                        };

                        Ok(Arc::new(new_node))
                    }

                    // The Clip, Split, Resize, Reshape and Reduce* operators each take optional inputs that influence the operation.
                    // These are typically statically initialized tensors containing shapes. For more efficient execution we
                    // move these static values to attributes.
                    op @ ("Clip" | "Pad" | "Split" | "Resize" | "Reshape" | "ReduceMean"
                    | "ReduceSum" | "ReduceMin" | "ReduceMax" | "ReduceSumSquare"
                    | "ReduceLogSumExp" | "ReduceLogSum" | "ReduceL2" | "ReduceL1"
                    | "ReduceProd") => {
                        if new_inputs.is_empty() {
                            return Err(OptimizerError::NoInputs);
                        }

                        // Names of the inputs (see ONNX operator spec)
                        let attr_names = match op {
                            "Split" => SPLIT_INPUT_NAMES,
                            "Resize" => RESIZE_INPUT_NAMES,
                            "Reshape" => RESHAPE_INPUT_NAMES,
                            "Clip" => CLIP_INPUT_NAMES,
                            "Pad" => PAD_INPUT_NAMES,
                            "ReduceSum" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceL1" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceL2" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceLogSum" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceLogSumExp" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceMax" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceMean" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceMin" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceProd" => REDUCE_OPS_INPUT_NAMES,
                            "ReduceSumSquare" => REDUCE_OPS_INPUT_NAMES,
                            _ => unreachable!(),
                        };

                        // Make a new copy of the attributes list (we're going to add attributes)
                        let mut new_proto = op_def.proto.clone().into_owned();
                        let mut attributes = op_def.proto.get_attribute().to_vec();

                        // Loop over the inputs (skipping the first one - that's going to be the data input)
                        for input_index in 1..(new_inputs.len().min(attr_names.len())) {
                            let source_node = &new_inputs[input_index].source_node;
                            match &source_node.definition {
                                // If the input is an initializer (Tensor) we can obtain the data from the definition and move it to an attribute
                                NodeDefinition::Tensor(tensor_proto) => {
                                    let attr_name = attr_names[input_index];
                                    let data_type =
                                        ScalarType::from_i32(tensor_proto.get_data_type())?;

                                    match (op, attr_name) {
                                        // Inputs that need to be converted to an i64 attribute
                                        ("Split", "split")
                                        | ("Resize", "roi")
                                        | ("Resize", "sizes")
                                        | ("Reshape", "shape")
                                        | (
                                            "ReduceMean" | "ReduceSum" | "ReduceMin" | "ReduceMax"
                                            | "ReduceSumSquare" | "ReduceLogSumExp"
                                            | "ReduceLogSum" | "ReduceL2" | "ReduceL1"
                                            | "ReduceProd",
                                            "axes",
                                        )
                                        | ("Pad", "pads") => match data_type {
                                            ScalarType::I64 => {
                                                log::info!(
                                                        "transferring input {} for op {} to i64 attribute (initializer data type: {:?})",
                                                        attr_name,
                                                        op,
                                                        data_type
                                                    );
                                                let value = tensor_proto.get_int64_data().to_vec();
                                                attributes.push(attribute(
                                                    attr_names[input_index],
                                                    value,
                                                ));
                                            }
                                            _ => {
                                                return Err(OptimizerError::InvalidInputDataType {
                                                    data_type,
                                                    input: attr_name.to_string(),
                                                    op: op.to_string(),
                                                })
                                            }
                                        },
                                        // Inputs that need to be converted to an f32 attribute
                                        ("Resize", "scales") => match data_type {
                                            ScalarType::F32 => {
                                                log::info!(
                                                        "transferring input {} for op {} to f32 attribute (initializer data type: {:?})",
                                                        attr_name,
                                                        op,
                                                        data_type
                                                    );
                                                let value: Vec<f32> =
                                                    tensor_proto.get_float_data().to_vec();
                                                attributes.push(attribute(
                                                    attr_names[input_index],
                                                    value,
                                                ));
                                            }
                                            _ => {
                                                return Err(OptimizerError::InvalidInputDataType {
                                                    data_type,
                                                    input: attr_name.to_string(),
                                                    op: op.to_string(),
                                                })
                                            }
                                        },
                                        _ => {
                                            // Some other unspecified input that we do not support yet
                                            return Err(OptimizerError::Unsupported(format!(
                                                "data_type {} for input {} to op {}",
                                                tensor_proto.get_data_type(),
                                                attr_name,
                                                op
                                            )));
                                        }
                                    }
                                }
                                NodeDefinition::Missing => {
                                    // Just remove it
                                }
                                _ => {
                                    // One of the inputs (except the first) is something other than a tensor (e.g. 'dynamic')
                                    return Err(OptimizerError::Unsupported(format!(
                                        "{} operation with dynamic input for {}",
                                        op, attr_names[input_index]
                                    )));
                                }
                            }
                        }

                        // Create new node with extra attributes
                        new_proto.set_attribute(RepeatedField::from(attributes));

                        let new_node = Node {
                            inputs: vec![new_inputs[0].clone()],
                            definition: NodeDefinition::Operator(Box::new(OperatorDefinition {
                                proto: Cow::Owned(new_proto),
                                output_shapes: op_def.output_shapes.clone(),
                            })),
                        };

                        Ok(Arc::new(new_node))
                    }

                    _ => Ok(Arc::new(Node {
                        inputs: new_inputs,
                        definition: NodeDefinition::Operator(op_def.clone()),
                    })),
                }
            }
            NodeDefinition::Tensor(..) | NodeDefinition::Input(..) => {
                assert!(
                    new_inputs.is_empty(),
                    "non-operator node cannot have inputs"
                );
                // No need to do anything with the provided new inputs
                Ok(node.clone())
            }
            &NodeDefinition::Outputs { .. } => Ok(Arc::new(Node {
                inputs: new_inputs,
                definition: node.definition().clone(),
            })),
            NodeDefinition::Missing => Ok(node.clone()),
        }
    }

    /// Attempt to fuse several operators in a chain of operators with no other dynamic inputs. The function receives a list
    /// of nodes that are guaranteed to be operators that each have one input (exactly). It is free to remove or add nodes
    /// to this list. The caller will fix up the input/output relationships between the nodes.
    fn optimize_chain(
        &mut self,
        chain: &mut VecDeque<Arc<Node<'model>>>,
    ) -> Result<bool, OptimizerError> {
        // Start by throwing out all Identity nodes
        chain.retain(|n| match &n.definition {
            NodeDefinition::Operator(op_def) => op_def.proto.get_op_type() != "Identity",
            _ => true,
        });

        let names: Vec<&str> = chain
            .iter()
            .map(|x| match &x.definition {
                NodeDefinition::Operator(op_def) => op_def.proto.get_op_type(),
                _ => "",
            })
            .collect();

        log::debug!("optimize_chain {:?}", names);

        match &names[..] {
            // Double Neg: just cull
            ["Neg", "Neg", ..] => {
                chain.pop_front();
                chain.pop_front();
                Ok(true)
            }

            // Conv+Relu or Conv+LeakyRelu: combine into ConvRelu/ConvLeakyRelu
            ["Conv", "Relu", ..] | ["Conv", "LeakyRelu", ..] => {
                let conv = chain[0].clone();
                let relu = chain[1].clone();

                if let (NodeDefinition::Operator(conv_def), NodeDefinition::Operator(relu_def)) =
                    (&conv.definition, &relu.definition)
                {
                    // Use the Conv node as template for the new fused Conv[Leaky]Relu node
                    let mut convrelu_def = *conv_def.clone();
                    let mut convrelu_proto = conv_def.proto.clone().into_owned();
                    let new_op_type = match relu_def.proto.get_op_type() {
                        "LeakyRelu" => "ConvLeakyRelu",
                        "Relu" => "ConvRelu",
                        _ => unreachable!(),
                    };
                    convrelu_proto.set_op_type(new_op_type.to_string());

                    // Copy all Relu attributes over to the copy of the Conv node
                    let mut attributes = conv_def.proto.get_attribute().to_vec();
                    attributes.extend(relu_def.proto.get_attribute().iter().cloned());
                    convrelu_proto.set_attribute(RepeatedField::from(attributes));
                    convrelu_proto.set_name(format!(
                        "{}+{}",
                        conv.definition.get_name(),
                        relu.definition.get_name()
                    ));

                    log::debug!(
                        "can fuse chain of Conv/[Leaky]Relu to Conv[Leaky]Relu: {:?}: {:?} + {:?} = {}",
                        names,
                        conv.definition(),
                        relu.definition(),
                        convrelu_proto.get_name()
                    );

                    convrelu_def.proto = Cow::Owned(convrelu_proto);

                    let node = Arc::new(Node {
                        inputs: conv.inputs.clone(),
                        definition: NodeDefinition::Operator(Box::new(convrelu_def)),
                    });

                    chain.remove(0);
                    chain.remove(0);
                    chain.insert(0, node);
                    Ok(true)
                } else {
                    unreachable!();
                }
            }
            _ => Ok(false),
        }
    }
}

// Names associated with the inputs of the Split, Resize, Reshape and Clip operators (in positional order - see ONNX spec)
static SPLIT_INPUT_NAMES: &[&str] = &["input", "split"];
static RESIZE_INPUT_NAMES: &[&str] = &["X", "roi", "scales", "sizes"];
static RESHAPE_INPUT_NAMES: &[&str] = &["data", "shape"];
static CLIP_INPUT_NAMES: &[&str] = &["input", "min", "max"];
static REDUCE_OPS_INPUT_NAMES: &[&str] = &["input", "axes"];
static PAD_INPUT_NAMES: &[&str] = &["data", "pads", "constant_value"];

/// Generate the output for a ConstantOfShape node
pub fn constant_of_shape_output(
    node: &NodeProto,
    element_count: usize,
) -> Result<OutputTensor, OptimizerError> {
    if let Ok(constant_value_tensor) = node.get_attribute_value::<TensorProto>("value", None) {
        match ScalarType::from_i32(constant_value_tensor.get_data_type()).map_err(|_| {
            OptimizerError::Unsupported(format!(
                "unsupported data type {}",
                constant_value_tensor.get_data_type()
            ))
        })? {
            ScalarType::F32 => {
                let fd = constant_value_tensor.get_float_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(OutputTensor::F32(vec![fd[0]; element_count]))
            }
            ScalarType::I64 => {
                let fd = constant_value_tensor.get_int64_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(OutputTensor::I64(vec![fd[0]; element_count]))
            }
            ScalarType::I32 => {
                let fd = constant_value_tensor.get_int32_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(OutputTensor::I32(vec![fd[0]; element_count]))
            }
            ScalarType::U8 => {
                let fd = constant_value_tensor.get_raw_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(OutputTensor::U8(vec![fd[0]; element_count]))
            }
        }
    } else {
        // The default value is a zero f32
        Ok(OutputTensor::F32(vec![0.0; element_count]))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        ir::{self, Node, NodeDefinition},
        onnx::AttributeProto,
        utils::{attribute, graph, initializer, model, node, tensor},
    };

    use super::Optimizer;

    fn friendly_name(node: Arc<Node>) -> String {
        match node.definition() {
            NodeDefinition::Outputs { .. } => String::from("<outputs>"),
            NodeDefinition::Missing => String::from("<missing>"),
            NodeDefinition::Operator(op_def) => {
                format!("{}_{}", op_def.proto.get_op_type(), op_def.proto.get_name())
            }
            d => format!("{}", d.get_name()),
        }
    }

    fn traverse(node: Arc<Node>, pairs: &mut Vec<(String, String)>) {
        let my_name = friendly_name(node.clone());
        for input in &node.inputs {
            let source_node_name = friendly_name(input.source_node.clone());
            pairs.push((source_node_name, my_name.to_string()))
        }

        for input in &node.inputs {
            traverse(input.source_node.clone(), pairs);
        }
    }

    // Test: X -> [Identity] A -> [Identity] -> Y => X -> Y
    #[test]
    pub fn test_optimize_identity_identity() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1])],
                vec![tensor("A", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Identity", vec![]),
                    node(vec!["A"], vec!["Y"], "b", "Identity", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(new_pairs, vec![("X".to_string(), "<outputs>".to_string())]);
        })
    }

    // Test: X -> [Neg] A -> [Neg] -> Y => X -> Y
    #[test]
    pub fn test_optimize_neg_neg() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1])],
                vec![tensor("A", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["Y"], "b", "Neg", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(new_pairs, vec![("X".to_string(), "<outputs>".to_string())]);
        });
    }

    // Test: X -> [Neg] A -> [Neg] B -> [Neg] -> Y => X -> Identity -> Y
    #[test]
    pub fn test_optimize_3neg() {
        pollster::block_on(async {
            let _ = env_logger::builder().is_test(true).try_init();

            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1])],
                vec![tensor("A", &[1]), tensor("B", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "b", "Neg", vec![]),
                    node(vec!["B"], vec!["Y"], "c", "Neg", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![
                    ("Neg_c".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_c".to_string())
                ]
            );
        });
    }

    // Test: X -> [Neg] A -> [Neg] B -> [Neg] C -> [Neg] -> Y => X -> Identity -> Y
    #[test]
    pub fn test_optimize_4neg() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1])],
                vec![tensor("A", &[1]), tensor("B", &[1]), tensor("C", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "b", "Neg", vec![]),
                    node(vec!["B"], vec!["C"], "c", "Neg", vec![]),
                    node(vec!["C"], vec!["Y"], "d", "Neg", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(new_pairs, vec![("X".to_string(), "<outputs>".to_string()),]);
        });
    }

    // Test: X -> [Neg] A -> [Neg] B -> [Neg] C -> [Neg] D -> [Neg] -> Y => X -> Neg -> Y
    #[test]
    pub fn test_optimize_5neg() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1])],
                vec![
                    tensor("A", &[1]),
                    tensor("B", &[1]),
                    tensor("C", &[1]),
                    tensor("D", &[1]),
                ],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "b", "Neg", vec![]),
                    node(vec!["B"], vec!["C"], "c", "Neg", vec![]),
                    node(vec!["C"], vec!["D"], "d", "Neg", vec![]),
                    node(vec!["D"], vec!["Y"], "e", "Neg", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![
                    ("Neg_e".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_e".to_string())
                ]
            );
        });
    }

    // Test: X -> [Neg] A -> [Neg] -> A, Y => X -> A, Y
    #[test]
    pub fn test_optimize_neg_neg_branch() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1]), tensor("A", &[1])],
                vec![tensor("A", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["Y"], "b", "Neg", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![
                    ("X".to_string(), "<outputs>".to_string()),
                    ("Neg_a".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_a".to_string())
                ]
            );
        });
    }

    // Test: X -> [Neg] A -> [Identity] Z -> [Identity] -> Y with Y and Z output => X -> Y, Z
    #[test]
    pub fn test_optimize_identity_identity_two_outputs() {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", &[1])],
                vec![tensor("Y", &[1]), tensor("Z", &[1])],
                vec![tensor("A", &[1])],
                vec![],
                vec![
                    node(vec!["X"], vec!["A"], "a", "Neg", vec![]),
                    node(vec!["A"], vec!["Z"], "b", "Identity", vec![]),
                    node(vec!["A"], vec!["Y"], "c", "Identity", vec![]),
                ],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![
                    ("Neg_a".to_string(), "<outputs>".to_string()),
                    ("Neg_a".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_a".to_string()),
                    ("X".to_string(), "Neg_a".to_string()),
                ]
            );
        });
    }

    // Test: A, B -> [Add] -> C where A, B are initializers
    #[test]
    pub fn test_constant_folding() {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            let m = model(graph(
                vec![],
                vec![tensor("C", &[1])],
                vec![],
                vec![
                    initializer("A", vec![21.0], vec![1]),
                    initializer("B", vec![7.0], vec![1]),
                ],
                vec![node(vec!["A", "B"], vec!["C"], "c", "Add", vec![])],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(new_pairs, vec![("C".to_string(), "<outputs>".to_string())]);
        });
    }

    // Test: [Constant] -> Y => [initializer] -> Y
    #[test]
    pub fn test_constant_node_to_tensor() {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            let m = model(graph(
                vec![],
                vec![tensor("Y", &[1])],
                vec![],
                vec![],
                vec![node(
                    vec![],
                    vec!["Y"],
                    "y",
                    "Constant",
                    vec![attribute("value_float", 42.0)],
                )],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root.clone(), &mut new_pairs);
            assert_eq!(new_pairs, vec![("Y".to_string(), "<outputs>".to_string())]);

            let y_node = new_root.inputs[0].source_node.clone();
            assert!(matches!(y_node.definition(), NodeDefinition::Tensor(_)));
        });
    }

    // Test: Input X -> [Shape] -> Y => [initializer] -> Y with initializer containing the correct shape of input X
    #[test]
    pub fn test_shape_operator() {
        test_shape_operator_with(
            &[1, 2, 3],
            vec![attribute("start", -3), attribute("end", -2)],
            &[1],
        );
        test_shape_operator_with(&[1, 2, 3], vec![], &[1, 2, 3]);
        test_shape_operator_with(&[3, 4, 5], vec![attribute("start", 0)], &[3, 4, 5]);
        test_shape_operator_with(&[3, 4, 5], vec![attribute("start", 1)], &[4, 5]);
        test_shape_operator_with(&[3, 4, 5], vec![attribute("start", -1)], &[5]);
        test_shape_operator_with(&[3, 4, 5], vec![attribute("end", 10)], &[3, 4, 5]);
        test_shape_operator_with(&[3, 4, 5], vec![attribute("end", 1)], &[3]);
        test_shape_operator_with(
            &[3, 4, 5],
            vec![attribute("start", 10), attribute("end", 10)],
            &[],
        );
    }

    pub fn test_shape_operator_with(
        input_shape: &[i64],
        attrs: Vec<AttributeProto>,
        expected: &[i64],
    ) {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", input_shape)],
                vec![tensor("Y", &[expected.len() as i64])],
                vec![],
                vec![],
                vec![node(vec!["X"], vec!["Y"], "y", "Shape", attrs)],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root.clone(), &mut new_pairs);
            assert_eq!(new_pairs, vec![("".to_string(), "<outputs>".to_string())]);

            let y_node = new_root.inputs[0].source_node.clone();
            let NodeDefinition::Tensor(t) = y_node.definition() else {
                panic!("should be folded to an initializer");
            };
            assert_eq!(t.get_int64_data(), expected);
        });
    }
}
