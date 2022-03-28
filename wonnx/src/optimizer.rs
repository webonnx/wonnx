use protobuf::RepeatedField;
use std::{borrow::Cow, collections::HashMap, sync::Arc};
use thiserror::Error;

use crate::{
    ir::{Input, Node, NodeDefinition, NodeIdentifier, OperatorDefinition},
    resource::padding,
    utils::{attribute, get_attribute, AttributeNotFoundError, DataTypeError, ScalarType},
};

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

    #[error("required attribute not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),
}

#[derive(Clone)]
struct Sequence<'model> {
    node: Arc<Node<'model>>,
    skip: usize,
}

pub struct Optimizer<'model> {
    padded_tensors: HashMap<String, Arc<Node<'model>>>,
    optimized: HashMap<NodeIdentifier<'model>, Sequence<'model>>,
}

impl<'model> Optimizer<'model> {
    pub fn new() -> Self {
        Self {
            padded_tensors: HashMap::new(),
            optimized: HashMap::new(),
        }
    }

    /// Optimize a graph
    pub fn optimize(
        &mut self,
        node: Arc<Node<'model>>,
    ) -> Result<Arc<Node<'model>>, OptimizerError> {
        let mut path = vec![];
        let seq = self.optimize_branch_cached(node, &mut path)?;
        if seq.skip != 0 {
            panic!("sequencing gone wrong");
        }
        Ok(seq.node)
    }

    fn optimize_branch_cached(
        &mut self,
        node: Arc<Node<'model>>,
        chain: &mut Vec<(String, Arc<Node<'model>>)>,
    ) -> Result<Sequence<'model>, OptimizerError> {
        if let Some(optimized) = self.optimized.get(&node.identifier()) {
            Ok(optimized.clone())
        } else {
            let identifier = node.identifier();
            let optimized = self.optimize_branch(node, chain)?;
            self.optimized.insert(identifier, optimized.clone());
            Ok(optimized)
        }
    }

    /// Optimize a branch of a graph. This function is recursively called on inputs. The 'chain' parameter is a list of
    /// operations that follow the node on which optimize_branch is called, and that have exactly one 'dynamic' input
    /// (either inference input or output of the previous operator). For instance, if the model is A->B->C, then the
    /// following calls will happen:
    ///
    /// - C.optimize_branch(chain = [])
    ///   - B.optimize_branch(chain = ["C"])
    ///     - A.optimize_branch(chain = ["C", "B"])
    ///
    /// If the model were A -> B+X -> C instead, then the chain would be 'broken' as 'B' has more than one input. The
    /// 'chain' information can be used to fuse multiple steps. When optimize_branch detects that a chain can be fused,
    /// it will generate the fused operation and return a Sequence struct with 'skip' set to a positive number.
    /// This number equals the number of operations that have been fused, and  which  should be omitted from the DAG.
    /// In the above example, suppose B->C can be fused. This can be detected in the last call:
    ///
    /// - C.optimize_branch(chain = []) receives BC(skip=1) from B.optimize_branch, will omit C and return BC(skip=0)
    ///   - B.optimize_branch(chain = ["C"]) receives BC(skip=2) from A.optimize_branch, will omit B and return BC(skip=1)
    ///     - A.optimize_branch(chain = ["C", "B"]) detects B->C fusable, returns BC node with skip=2
    fn optimize_branch(
        &mut self,
        node: Arc<Node<'model>>,
        chain: &mut Vec<(String, Arc<Node<'model>>)>,
    ) -> Result<Sequence<'model>, OptimizerError> {
        log::info!(
            "Optimize {:?} chain length={:?}",
            node.definition,
            chain.len()
        );

        if let NodeDefinition::Operator(op_def) = &node.definition {
            // Specific operators can be spliced out of the DAG
            match op_def.proto.get_op_type() {
                // Identity: we just remove these nodes and connect the input of the destination node to our input's output
                // A -> Identity -> B => A -> B
                "Identity" => {
                    if node.inputs.len() != 1 {
                        return Err(OptimizerError::NoInputs);
                    }
                    return self.optimize_branch_cached(node.inputs[0].source_node.clone(), chain);
                }
                // The Dropout operation does nothing when its training_mode is set to 0. We do not support training_mode=1
                "Dropout" => {
                    if node.inputs.is_empty() {
                        return Err(OptimizerError::NoInputs);
                    }
                    let training_mode =
                        get_attribute("training_mode", Some(0), &op_def.proto)? == 1;

                    if training_mode {
                        return Err(OptimizerError::Unsupported(String::from(
                            "Dropout with training_mode=1",
                        )));
                    }

                    return self.optimize_branch_cached(node.inputs[0].source_node.clone(), chain);
                }
                _ => {}
            }

            // Optimize chains of operators. A chain can only be formed for operations that each have at most one 'dynamic'
            // input (e.g. inference input or the input from another operation)
            let dynamic_input_count = node
                .inputs
                .iter()
                .filter(|input| {
                    matches!(
                        input.source_node.definition,
                        NodeDefinition::Operator(..) | NodeDefinition::Input(..)
                    )
                })
                .count();

            if dynamic_input_count == 1 {
                chain.push((op_def.proto.get_op_type().to_string(), node.clone()));

                if let Some(seq) = self.optimize_chain(chain)? {
                    log::info!(
                        "chain optimization: fuse {} operators to {:?}",
                        seq.skip,
                        seq.node.definition
                    );
                    return Ok(seq);
                }

                // Continue optimizing inputs, making the chain longer for one input
                let mut new_inputs = Vec::with_capacity(node.inputs.len());

                for input in &node.inputs {
                    let source_sequence =
                        self.optimize_branch_cached(input.source_node.clone(), chain)?;

                    if source_sequence.skip > 0 {
                        log::info!("operator is optimized away: {:?}", node.definition);
                        return Ok(Sequence {
                            node: source_sequence.node,
                            skip: source_sequence.skip - 1,
                        });
                    }

                    new_inputs.push(Input {
                        source_node: source_sequence.node,
                        output_index: input.output_index,
                    });
                }

                return Ok(Sequence {
                    node: self.optimized_with(&node, new_inputs)?,
                    skip: 0,
                });
            }
        }

        // This node has multiple inputs; optimize each input as a new chain
        let new_inputs = node
            .inputs
            .iter()
            .map(|input| {
                let mut input_chain = vec![];
                let source_sequence =
                    self.optimize_branch_cached(input.source_node.clone(), &mut input_chain)?;

                assert!(
                    source_sequence.skip == 0,
                    "cannot skip items in a chain that has multiple inputs for a single op"
                );

                Ok(Input {
                    source_node: source_sequence.node,
                    output_index: input.output_index,
                })
            })
            .collect::<Result<Vec<Input>, OptimizerError>>()?;

        Ok(Sequence {
            node: self.optimized_with(&node, new_inputs)?,
            skip: 0,
        })
    }

    /// Create a new node from an existing definition, applying optimizations local to a single node
    fn optimized_with(
        &mut self,
        node: &Arc<Node<'model>>,
        mut new_inputs: Vec<Input<'model>>,
    ) -> Result<Arc<Node<'model>>, OptimizerError> {
        match &node.definition {
            NodeDefinition::Operator(op_def) => {
                match op_def.proto.get_op_type() {
                    "Conv" | "ConvRelu" | "ConvLeakyRelu" => {
                        // This optimization inserts some padding to convolution between kernels with kernel 3x3, because of
                        // the stride of matrix3x3 is 16 in wgsl. It makes the computation matrixable and increases the performance.
                        if new_inputs.len() > 2
                            && get_attribute::<Vec<i64>>("kernel_shape", None, &op_def.proto)?
                                == [3, 3]
                            && (get_attribute("pads", Some(vec![0, 0, 0, 0]), &op_def.proto)?
                                == [1, 1, 1, 1]
                                || get_attribute(
                                    "auto_pad",
                                    Some("SAME_UPPER".to_string()),
                                    &op_def.proto,
                                )? == "SAME_UPPER")
                            && get_attribute("strides", Some(vec![1, 1]), &op_def.proto)? == [1, 1]
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

                    // The Clip, Split, Resize and Reshape operator each take optional inputs that influence the operation.
                    // These are typically statically initialized tensors containing shapes. For more efficient execution we
                    // move these static values to attributes.
                    op @ ("Clip" | "Split" | "Resize" | "Reshape" | "ReduceSum") => {
                        if new_inputs.is_empty() {
                            return Err(OptimizerError::NoInputs);
                        }

                        // Names of the inputs (see ONNX operator spec)
                        let attr_names = match op {
                            "Split" => SPLIT_INPUT_NAMES,
                            "Resize" => RESIZE_INPUT_NAMES,
                            "Reshape" => RESHAPE_INPUT_NAMES,
                            "Clip" => CLIP_INPUT_NAMES,
                            "ReduceSum" => REDUCESUM_INPUT_NAMES,
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
                                        | ("ReduceSum", "axes") => match data_type {
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

    /// Attempt to fuse several operators in a chain of operators with no other dynamic inputs.
    fn optimize_chain(
        &mut self,
        chain: &[(String, Arc<Node<'model>>)],
    ) -> Result<Option<Sequence<'model>>, OptimizerError> {
        let path_slices: Vec<&str> = chain.iter().rev().map(|x| x.0.as_str()).collect();
        match &path_slices[..] {
            // Conv+Relu or Conv+LeakyRelu: combine into ConvRelu/ConvLeakyRelu
            ["Conv", "Relu", ..] | ["Conv", "LeakyRelu", ..] => {
                let conv = chain[chain.len() - 1].1.clone();
                let relu = chain[chain.len() - 2].1.clone();

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
                        path_slices,
                        conv.definition(),
                        relu.definition(),
                        convrelu_proto.get_name()
                    );

                    convrelu_def.proto = Cow::Owned(convrelu_proto);

                    let new_inputs = conv
                        .inputs
                        .iter()
                        .map(|input| -> Result<Input, OptimizerError> {
                            Ok(Input {
                                source_node: self.optimize(input.source_node.clone())?,
                                output_index: input.output_index,
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let node = Arc::new(Node {
                        inputs: conv.inputs.clone(),
                        definition: NodeDefinition::Operator(Box::new(convrelu_def)),
                    });

                    Ok(Some(Sequence {
                        node: self.optimized_with(&node, new_inputs)?,
                        skip: 1,
                    }))
                } else {
                    unreachable!();
                }
            }
            _ => Ok(None),
        }
    }
}

impl<'model> Default for Optimizer<'model> {
    fn default() -> Self {
        Self::new()
    }
}

// Names associated with the inputs of the Split, Resize, Reshape and Clip operators (in positional order - see ONNX spec)
static SPLIT_INPUT_NAMES: &[&str] = &["input", "split"];
static RESIZE_INPUT_NAMES: &[&str] = &["X", "roi", "scales", "sizes"];
static RESHAPE_INPUT_NAMES: &[&str] = &["data", "shape"];
static CLIP_INPUT_NAMES: &[&str] = &["input", "min", "max"];
static REDUCESUM_INPUT_NAMES: &[&str] = &["input", "axes"];
