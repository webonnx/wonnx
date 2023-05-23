//! Optimizer that walks the DAG and transforms or coalesces ops for quicker execution
use crate::{
    gpu::GpuModel,
    ir::{
        AttributeNotFoundError, Input, Node, NodeDefinition, NodeIdentifier, OperatorDefinition,
        Tensor,
    },
    onnx::TensorProto,
    resource::{padding, request_device_queue},
    utils::{DataTypeError, ScalarType, TensorData},
    GpuError,
};
use async_recursion::async_recursion;
use std::{
    borrow::{Borrow, Cow},
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
    padded_tensors: HashMap<NodeIdentifier<'model>, Arc<Node<'model>>>,
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
                if op_def.output_shapes().len() != 1 {
                    log::warn!(
                        "node {:?} is constant, but has multiple outputs, which we can't fold yet",
                        node.definition()
                    );
                    return Ok(None);
                }

                match op_def.get_op_type() {
                    "Constant" => Ok(Some(Arc::new(Node {
                        definition: NodeDefinition::Tensor(Self::constant_node_to_tensor(node)?),
                        inputs: vec![],
                    }))),
                    _ => self.infer_constant_node_to_tensor(node.clone()).await,
                }
            }
            NodeDefinition::Tensor(_) => Ok(None), // already constantized
            NodeDefinition::Input { .. } | NodeDefinition::Missing => unreachable!(),
            NodeDefinition::Outputs { .. } => Ok(None), // all the outputs themselves are already constant, so nothing to do
        }
    }

    // Takes a node with operator type 'Shape' and returns its output as a tensor
    fn shape_node_to_tensor(node: Arc<Node<'model>>) -> Result<Tensor<'static>, OptimizerError> {
        let NodeDefinition::Operator(op_def) = node.definition() else {
            panic!("node must be a Shape node");
        };
        assert_eq!(op_def.get_op_type(), "Shape");

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
            NodeDefinition::Input { shape, .. } => shape.clone(),
            NodeDefinition::Operator(input_op_def) => {
                input_op_def.output_shapes()[input.output_index].clone()
            }
            NodeDefinition::Tensor(input_tensor) => input_tensor.shape(),
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
        let mut start: i64 = op_def.get_attribute_value("start", Some(0)).unwrap();
        let mut end: i64 = op_def.get_attribute_value("end", Some(rank)).unwrap();
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
        let dims = vec![values.len()];
        Ok(Tensor {
            data: TensorData::I64(values.into()),
            dims,
            display_name: format!("<folded>{}", node.definition().get_display_name()),
        })
    }

    // Takes a node with operator type 'Constant' and returns its output as a tensor
    fn constant_node_to_tensor(node: Arc<Node<'model>>) -> Result<Tensor<'model>, OptimizerError> {
        let NodeDefinition::Operator(op_def) = node.definition() else {
            panic!("node must be a Constant node");
        };
        assert_eq!(op_def.get_op_type(), "Constant");
        let display_name = op_def.get_display_name().into();

        let tp: Tensor =
            if let Ok(values) = op_def.get_attribute_value::<Vec<f32>>("value_floats", None) {
                let dims = vec![values.len()];
                Tensor {
                    data: TensorData::F32(values.into()),
                    dims,
                    display_name,
                }
            } else if let Ok(values) = op_def.get_attribute_value::<Vec<i64>>("value_ints", None) {
                let dims = vec![values.len()];
                Tensor {
                    data: TensorData::I64(values.into()),
                    dims,
                    display_name,
                }
            } else if let Ok(value) = op_def.get_attribute_value::<f32>("value_float", None) {
                Tensor {
                    data: TensorData::F32(vec![value].into()),
                    dims: vec![1],
                    display_name,
                }
            } else if let Ok(value) = op_def.get_attribute_value::<i64>("value_int", None) {
                Tensor {
                    data: TensorData::I64(vec![value].into()),
                    dims: vec![1],
                    display_name,
                }
            } else if let Ok(_tp) = op_def.get_attribute_value::<TensorProto>("value", None) {
                todo!();
                // to_tensor(Cow::Owned(tp))?
            } else {
                return Err(OptimizerError::Unsupported(
                    "Constant node with unknown value type".to_string(),
                ));
            };
        Ok(tp)
    }

    // Takes a node with operator type 'Size' and returns its output as a tensor
    fn size_node_to_tensor(node: Arc<Node<'model>>) -> Result<Tensor<'static>, OptimizerError> {
        let NodeDefinition::Operator(op_def) = node.definition() else {
            panic!("node must be a Size node");
        };
        assert_eq!(op_def.get_op_type(), "Size");

        if node.inputs.len() != 1 {
            return Err(OptimizerError::InvalidNode(format!(
                "Size node should only have one input, has {}",
                node.inputs.len()
            )));
        }

        // Determine the shape of the input
        let input = &node.inputs[0];
        let in_node = &input.source_node.definition;
        let in_element_count: i64 = match in_node {
            NodeDefinition::Input { shape, .. } => shape.element_count() as i64,
            NodeDefinition::Operator(input_op_def) => {
                input_op_def.output_shapes()[input.output_index].element_count() as i64
            }
            NodeDefinition::Tensor(input_tensor) => input_tensor.shape().element_count() as i64,
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

        Ok(Tensor {
            data: TensorData::I64(vec![in_element_count].into()),
            dims: vec![1],
            display_name: format!("<folded>{}", node.definition().get_display_name()),
        })
    }

    // Infers the output for a constant node (must be a constant and operator node, or the function panics)
    async fn infer_constant_node_to_tensor(
        &self,
        node: Arc<Node<'model>>,
    ) -> Result<Option<Arc<Node<'model>>>, OptimizerError> {
        assert!(node.is_constant());

        // Create an output node so we can perform inference for this node
        if let NodeDefinition::Operator(op_def) = node.definition() {
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
            log::info!(
                "folded output of {} to {output_tensor:?}",
                op_def.get_display_name()
            );
            let shape = op_def.output_shapes()[0].clone();

            let tensor_node = Node {
                definition: NodeDefinition::Tensor(Tensor {
                    data: output_tensor.into_static(),
                    dims: shape.dims,
                    display_name: format!("<folded>{}", op_def.get_display_name().to_owned()),
                }),
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

        // Fold Shape/Size nodes (not considered constant but we can still fold it)
        if let NodeDefinition::Operator(op_def) = &node.definition {
            match op_def.get_op_type() {
                "Shape" => {
                    return Ok(Arc::new(Node {
                        definition: NodeDefinition::Tensor(Self::shape_node_to_tensor(node)?),
                        inputs: vec![],
                    }))
                }
                "Size" => {
                    return Ok(Arc::new(Node {
                        definition: NodeDefinition::Tensor(Self::size_node_to_tensor(node)?),
                        inputs: vec![],
                    }))
                }
                _ => {}
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
                match op_def.get_op_type() {
                    "Conv" | "ConvRelu" | "ConvLeakyRelu" => {
                        // This optimization inserts some padding to convolution between kernels with kernel 3x3, because of
                        // the stride of matrix3x3 is 16 in wgsl. It makes the computation matrixable and increases the performance.
                        if new_inputs.len() > 2
                            && op_def.get_attribute_value::<Vec<i64>>("kernel_shape", None)?
                                == [3, 3]
                            && (op_def.get_attribute_value("pads", Some(vec![0, 0, 0, 0]))?
                                == [1, 1, 1, 1]
                                || op_def.get_attribute_value(
                                    "auto_pad",
                                    Some("SAME_UPPER".to_string()),
                                )? == "SAME_UPPER")
                            && op_def.get_attribute_value("strides", Some(vec![1, 1]))? == [1, 1]
                            && op_def.get_attribute_value("group", Some(1))? == 1
                            && op_def.output_shapes()[0].dim(1) % 4 == 0
                        {
                            let source_node = {
                                if let NodeDefinition::Tensor(tensor) =
                                    &new_inputs[1].source_node.definition
                                {
                                    let source_identifier = new_inputs[1].source_node.identifier();
                                    if let Some(n) = self.padded_tensors.get(&source_identifier) {
                                        log::info!(
                                            "have cached padding optimized tensor for {:?}",
                                            source_identifier
                                        );
                                        n.clone()
                                    } else {
                                        match tensor.data() {
                                            TensorData::F32(floats) => {
                                                let raw_data: &[u8] =
                                                    bytemuck::cast_slice(floats.borrow());
                                                let padded_raw_data = padding(raw_data, 12, 4);
                                                log::info!(
                                                    "applying padding optimization to tensor {} shape {}: strides data is {} bytes before, {} bytes after",
                                                    tensor.display_name(),
                                                    tensor.shape(),
                                                    raw_data.len(),
                                                    padded_raw_data.len()
                                                );

                                                let padded_data =
                                                    bytemuck::pod_collect_to_vec(&padded_raw_data);

                                                // Create a new tensor with the padded data
                                                let new_node = Arc::new(Node {
                                                    definition: NodeDefinition::Tensor(Tensor {
                                                        data: TensorData::F32(Cow::Owned(
                                                            padded_data,
                                                        )),
                                                        dims: tensor.dims().to_vec(),
                                                        display_name: format!(
                                                            "<padded>{}",
                                                            tensor.display_name()
                                                        ),
                                                    }),
                                                    inputs: vec![],
                                                });
                                                self.padded_tensors.insert(
                                                    source_identifier.clone(),
                                                    new_node.clone(),
                                                );
                                                new_node
                                            }
                                            _ => {
                                                log::warn!("not applying padding optimization as source tensor does not have float data");
                                                return Ok(node.clone());
                                            }
                                        }
                                    }
                                } else {
                                    return Ok(Arc::new(Node {
                                        inputs: new_inputs,
                                        definition: node.definition().clone(),
                                    }));
                                }
                            };

                            new_inputs[1] = Input {
                                source_node: source_node.clone(),
                                output_index: 0,
                            };

                            let new_node = Node {
                                inputs: new_inputs,
                                definition: NodeDefinition::Operator(op_def.clone()),
                            };
                            log::info!(
                                "actually returning new node {} with new input 1 padded {}",
                                node.definition().get_display_name(),
                                source_node.definition().get_display_name()
                            );
                            Ok(Arc::new(new_node))
                        } else {
                            log::info!(
                                "actually returning old node {}",
                                node.definition().get_display_name()
                            );
                            Ok(Arc::new(Node {
                                inputs: new_inputs,
                                definition: node.definition().clone(),
                            }))
                        }
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
                        let mut new_proto = op_def.clone();

                        // Loop over the inputs (skipping the first one - that's going to be the data input)
                        for input_index in 1..(new_inputs.len().min(attr_names.len())) {
                            let source_node = &new_inputs[input_index].source_node;
                            match &source_node.definition {
                                // If the input is an initializer (Tensor) we can obtain the data from the definition and move it to an attribute
                                NodeDefinition::Tensor(tensor) => {
                                    let attr_name = attr_names[input_index];
                                    let data_type = tensor.shape().data_type;

                                    match (op, attr_name) {
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
                                        | ("Pad", "pads")
                                        | ("Resize", "scales")
                                        | ("Clip", "min" | "max") => match tensor.data() {
                                            TensorData::F32(value) => {
                                                log::info!(
                                                    "transferring input {} for op {} to f32 attribute (initializer data type: {:?}): {:?}",
                                                    attr_name,
                                                    op,
                                                    data_type,
                                                    value,
                                                );
                                                new_proto.set_attribute(
                                                    attr_names[input_index],
                                                    value.to_vec(),
                                                );
                                            }
                                            TensorData::I64(value) => {
                                                log::info!(
                                                    "transferring input {} for op {} to i64 attribute (initializer data type: {:?}): {:?}",
                                                    attr_name,
                                                    op,
                                                    data_type,
                                                    value,
                                                );
                                                new_proto.set_attribute(
                                                    attr_names[input_index],
                                                    value.to_vec(),
                                                );
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
                                                tensor.shape().data_type,
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

                        let new_node = Node {
                            inputs: vec![new_inputs[0].clone()],
                            definition: NodeDefinition::Operator(new_proto),
                        };

                        Ok(Arc::new(new_node))
                    }

                    _ => Ok(Arc::new(Node {
                        inputs: new_inputs,
                        definition: NodeDefinition::Operator(op_def.clone()),
                    })),
                }
            }
            NodeDefinition::Tensor(..) | NodeDefinition::Input { .. } => {
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
            NodeDefinition::Operator(op_def) => op_def.get_op_type() != "Identity",
            _ => true,
        });

        let names: Vec<&str> = chain
            .iter()
            .map(|x| match &x.definition {
                NodeDefinition::Operator(op_def) => op_def.get_op_type(),
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
                    let mut convrelu_def = conv_def.clone();
                    let new_op_type = match relu_def.get_op_type() {
                        "LeakyRelu" => "ConvLeakyRelu",
                        "Relu" => "ConvRelu",
                        _ => unreachable!(),
                    };
                    convrelu_def.set_op_type(new_op_type);

                    // Copy all Relu attributes over to the copy of the Conv node
                    convrelu_def.append_attributes_from(relu_def);

                    log::debug!(
                        "can fuse chain of Conv/[Leaky]Relu to Conv[Leaky]Relu: {:?}: {:?} + {:?} = {}",
                        names,
                        conv.definition(),
                        relu.definition(),
                        convrelu_def.get_display_name()
                    );

                    let node = Arc::new(Node {
                        inputs: conv.inputs.clone(),
                        definition: NodeDefinition::Operator(convrelu_def),
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
    node: &OperatorDefinition,
    element_count: usize,
) -> Result<TensorData<'static>, OptimizerError> {
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
                Ok(TensorData::F32(vec![fd[0]; element_count].into()))
            }
            ScalarType::I64 => {
                let fd = constant_value_tensor.get_int64_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(TensorData::I64(vec![fd[0]; element_count].into()))
            }
            ScalarType::I32 => {
                let fd = constant_value_tensor.get_int32_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(TensorData::I32(vec![fd[0]; element_count].into()))
            }
            ScalarType::U8 => {
                let fd = constant_value_tensor.get_raw_data();
                if fd.is_empty() {
                    return Err(OptimizerError::InvalidNode(
                        "value tensor for ConstantOfShape is empty".to_string(),
                    ));
                }
                Ok(TensorData::U8(vec![fd[0]; element_count].into()))
            }
        }
    } else {
        // The default value is a zero f32
        Ok(TensorData::F32(vec![0.0; element_count].into()))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        ir::{self, Node, NodeDefinition},
        onnx::AttributeProto,
        utils::{attribute, graph, initializer, model, node, tensor, TensorData},
    };

    use super::Optimizer;

    fn friendly_name(node: Arc<Node>) -> String {
        match node.definition() {
            NodeDefinition::Outputs { .. } => String::from("<outputs>"),
            NodeDefinition::Missing => String::from("<missing>"),
            NodeDefinition::Operator(op_def) => {
                format!("{}_{}", op_def.get_op_type(), op_def.get_display_name())
            }
            d => format!("{}", d.get_display_name()),
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
                    node(vec!["X"], vec!["A"], "Identity", vec![]),
                    node(vec!["A"], vec!["Y"], "Identity", vec![]),
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["Y"], "Neg", vec![]),
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "Neg", vec![]),
                    node(vec!["B"], vec!["Y"], "Neg", vec![]),
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
                    ("Neg_Y".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_Y".to_string())
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "Neg", vec![]),
                    node(vec!["B"], vec!["C"], "Neg", vec![]),
                    node(vec!["C"], vec!["Y"], "Neg", vec![]),
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["B"], "Neg", vec![]),
                    node(vec!["B"], vec!["C"], "Neg", vec![]),
                    node(vec!["C"], vec!["D"], "Neg", vec![]),
                    node(vec!["D"], vec!["Y"], "Neg", vec![]),
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
                    ("Neg_Y".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_Y".to_string())
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["Y"], "Neg", vec![]),
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
                    ("Neg_A".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_A".to_string())
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
                    node(vec!["X"], vec!["A"], "Neg", vec![]),
                    node(vec!["A"], vec!["Z"], "Identity", vec![]),
                    node(vec!["A"], vec!["Y"], "Identity", vec![]),
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
                    ("Neg_A".to_string(), "<outputs>".to_string()),
                    ("Neg_A".to_string(), "<outputs>".to_string()),
                    ("X".to_string(), "Neg_A".to_string()),
                    ("X".to_string(), "Neg_A".to_string()),
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
                vec![node(vec!["A", "B"], vec!["C"], "Add", vec![])],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root, &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![("<folded>C".to_string(), "<outputs>".to_string())]
            );
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
                vec![node(vec!["X"], vec!["Y"], "Shape", attrs)],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root.clone(), &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![("<folded>Y".to_string(), "<outputs>".to_string())]
            );

            let y_node = new_root.inputs[0].source_node.clone();
            let NodeDefinition::Tensor(t) = y_node.definition() else {
                panic!("should be folded to an initializer");
            };
            assert_eq!(t.data(), &TensorData::I64(expected.into()));
        });
    }

    // Test: Input X -> [Size] -> Y => [initializer] -> Y with initializer containing the correct shape of input X
    #[test]
    pub fn test_size_operator() {
        test_size_operator_with(&[1, 2, 3], &[6]);
        test_size_operator_with(&[1], &[1]);
        test_size_operator_with(&[], &[1]);
    }

    pub fn test_size_operator_with(input_shape: &[i64], expected: &[i64]) {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            let m = model(graph(
                vec![tensor("X", input_shape)],
                vec![tensor("Y", &[expected.len() as i64])],
                vec![],
                vec![],
                vec![node(vec!["X"], vec!["Y"], "Size", vec![])],
            ));

            let root = ir::Node::from_model(&m, None).unwrap();
            let mut opt = Optimizer::new(13);
            let new_root = opt.optimize(root).await.unwrap();
            let mut new_pairs = vec![];
            traverse(new_root.clone(), &mut new_pairs);
            assert_eq!(
                new_pairs,
                vec![("<folded>Y".to_string(), "<outputs>".to_string())]
            );

            let y_node = new_root.inputs[0].source_node.clone();
            let NodeDefinition::Tensor(t) = y_node.definition() else {
                panic!("should be folded to an initializer");
            };
            assert_eq!(t.data(), &TensorData::I64(expected.into()));
        });
    }
}
