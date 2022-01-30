use protobuf::{ProtobufEnum, RepeatedField};
use std::{borrow::Cow, sync::Arc};
use thiserror::Error;

use crate::{
    ir::{Input, Node, NodeDefinition, OperatorDefinition},
    onnx::TensorProto_DataType,
    utils::{attribute, get_attribute, AttributeNotFoundError},
};

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("node has no inputs")]
    NoInputs,

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("invalid data type {data_type:?} for input {input} of op {op}")]
    InvalidDataType {
        data_type: TensorProto_DataType,
        input: String,
        op: String,
    },

    #[error("required attribute not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),
}

struct Sequence<'model> {
    node: Arc<Node<'model>>,
    skip: usize,
}

impl<'model> Node<'model> {
    /// Optimize a graph
    pub fn optimize(self: Arc<Self>) -> Result<Arc<Self>, OptimizerError> {
        let mut path = vec![];
        let seq = self.optimize_branch(&mut path)?;
        if seq.skip != 0 {
            panic!("sequencing gone wrong");
        }
        Ok(seq.node)
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
        self: Arc<Self>,
        chain: &mut Vec<(String, Arc<Self>)>,
    ) -> Result<Sequence<'model>, OptimizerError> {
        log::info!(
            "Optimize {:?} chain length={:?}",
            self.definition,
            chain.len()
        );

        if let NodeDefinition::Operator(_, op_def) = &self.definition {
            // Specific operators can be spliced out of the DAG
            match op_def.proto.get_op_type() {
                // Identity: we just remove these nodes and connect the input of the destination node to our input's output
                // A -> Identity -> B => A -> B
                "Identity" => {
                    if self.inputs.len() != 1 {
                        return Err(OptimizerError::NoInputs);
                    }
                    return self.inputs[0].source_node.clone().optimize_branch(chain);
                }
                // The Dropout operation does nothing when its training_mode is set to 0. We do not support training_mode=1
                "Dropout" => {
                    if self.inputs.is_empty() {
                        return Err(OptimizerError::NoInputs);
                    }
                    let training_mode =
                        get_attribute("training_mode", Some(0), &op_def.proto)? == 1;

                    if training_mode {
                        return Err(OptimizerError::Unsupported(String::from(
                            "Dropout with training_mode=1",
                        )));
                    }

                    return self.inputs[0].source_node.clone().optimize_branch(chain);
                }
                _ => {}
            }

            // Optimize chains of operators. A chain can only be formed for operations that each have at most one 'dynamic'
            // input (e.g. inference input or the input from another operation)
            let dynamic_input_count = self
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
                chain.push((op_def.proto.get_op_type().to_string(), self.clone()));

                if let Some(seq) = optimize_chain(chain)? {
                    log::info!(
                        "chain optimization: fuse {} operators to {:?}",
                        seq.skip,
                        seq.node.definition
                    );
                    return Ok(seq);
                }

                // Continue optimizing inputs, making the chain longer for one input
                let mut new_inputs = Vec::with_capacity(self.inputs.len());

                for input in &self.inputs {
                    let source_sequence = input.source_node.clone().optimize_branch(chain)?;

                    if source_sequence.skip > 0 {
                        log::info!("operator is optimized away: {:?}", self.definition);
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
                    node: Self::new_optimized(self.definition.clone(), new_inputs)?,
                    skip: 0,
                });
            }
        }

        // This node has multiple inputs; optimize each input as a new chain
        let new_inputs = self
            .inputs
            .iter()
            .map(|input| {
                let mut input_chain = vec![];
                let source_sequence = input
                    .source_node
                    .clone()
                    .optimize_branch(&mut input_chain)?;

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
            node: Self::new_optimized(self.definition.clone(), new_inputs)?,
            skip: 0,
        })
    }
}

/// Attempt to fuse several operators in a chain of operators with no other dynamic inputs.
fn optimize_chain<'model>(
    chain: &[(String, Arc<Node<'model>>)],
) -> Result<Option<Sequence<'model>>, OptimizerError> {
    let path_slices: Vec<&str> = chain.iter().rev().map(|x| x.0.as_str()).collect();
    match &path_slices[..] {
        // Conv+Relu or Conv+LeakyRelu: combine into ConvRelu/ConvLeakyRelu
        ["Conv", "Relu", ..] | ["Conv", "LeakyRelu", ..] => {
            log::info!("can fuse chain to Conv[Leaky]Relu: {:?}", path_slices);

            let conv = chain[chain.len() - 1].1.clone();
            let relu = chain[chain.len() - 2].1.clone();

            if let (
                NodeDefinition::Operator(conv_index, conv_def),
                NodeDefinition::Operator(_relu_index, relu_def),
            ) = (&conv.definition, &relu.definition)
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

                // Copy all Relu attributes over
                let mut attributes = conv_def.proto.get_attribute().to_vec();
                attributes.extend(relu_def.proto.get_attribute().iter().cloned());
                convrelu_proto.set_attribute(RepeatedField::from(attributes));
                convrelu_def.proto = Cow::Owned(convrelu_proto);

                let node = Node {
                    inputs: conv.inputs.clone(),
                    definition: NodeDefinition::Operator(*conv_index, Box::new(convrelu_def)),
                };

                Ok(Some(Sequence {
                    node: Arc::new(node).optimize()?,
                    skip: 1,
                }))
            } else {
                unreachable!();
            }
        }
        _ => Ok(None),
    }
}

// Names associated with the inputs of the Split, Resize, Reshape and Clip operators (in positional order - see ONNX spec)
static SPLIT_INPUT_NAMES: &[&str] = &["input", "split"];
static RESIZE_INPUT_NAMES: &[&str] = &["X", "roi", "scales", "sizes"];
static RESHAPE_INPUT_NAMES: &[&str] = &["data", "shape"];
static CLIP_INPUT_NAMES: &[&str] = &["input", "min", "max"];

impl<'model> Node<'model> {
    /// Create a new node from an existing definition, applying optimizations local to a single node
    fn new_optimized(
        definition: NodeDefinition<'model>,
        inputs: Vec<Input<'model>>,
    ) -> Result<Arc<Self>, OptimizerError> {
        if let NodeDefinition::Operator(op_index, op_def) = definition {
            match op_def.proto.get_op_type() {
                // The Clip, Split, Resize and Reshape operator each take optional inputs that influence the operation.
                // These are typically statically initialized tensors containing shapes. For more efficient execution we
                // move these static values to attributes.
                op @ ("Clip" | "Split" | "Resize" | "Reshape") => {
                    if inputs.is_empty() {
                        return Err(OptimizerError::NoInputs);
                    }

                    // Names of the inputs (see ONNX operator spec)
                    let attr_names = match op {
                        "Split" => SPLIT_INPUT_NAMES,
                        "Resize" => RESIZE_INPUT_NAMES,
                        "Reshape" => RESHAPE_INPUT_NAMES,
                        "Clip" => CLIP_INPUT_NAMES,
                        _ => unreachable!(),
                    };

                    // Make a new copy of the attributes list (we're going to add attributes)
                    let mut new_proto = op_def.proto.clone().into_owned();
                    let mut attributes = op_def.proto.get_attribute().to_vec();

                    // Loop over the inputs (skipping the first one - that's going to be the data input)
                    for input_index in 1..(inputs.len().min(attr_names.len())) {
                        let source_node = &inputs[input_index].source_node;
                        match &source_node.definition {
                            // If the input is an initializer (Tensor) we can obtain the data from the definition and move it to an attribute
                            NodeDefinition::Tensor(_, tensor_proto) => {
                                let attr_name = attr_names[input_index];
                                let data_type =
                                    TensorProto_DataType::from_i32(tensor_proto.get_data_type())
                                        .unwrap_or(TensorProto_DataType::UNDEFINED);

                                match (op, attr_name) {
                                    // Inputs that need to be converted to an i64 attribute
                                    ("Split", "split")
                                    | ("Resize", "roi")
                                    | ("Resize", "sizes")
                                    | ("Reshape", "shape") => match data_type {
                                        TensorProto_DataType::INT64
                                        | TensorProto_DataType::UNDEFINED => {
                                            log::info!(
                                                    "transferring input {} for op {} to i64 attribute (initializer data type: {:?})",
                                                    attr_name,
                                                    op,
                                                    data_type
                                                );
                                            let value = tensor_proto.get_int64_data().to_vec();
                                            attributes
                                                .push(attribute(attr_names[input_index], value));
                                        }
                                        _ => {
                                            return Err(OptimizerError::InvalidDataType {
                                                data_type,
                                                input: attr_name.to_string(),
                                                op: op.to_string(),
                                            })
                                        }
                                    },
                                    // Inputs that need to be converted to an f32 attribute
                                    ("Resize", "scales") => match data_type {
                                        TensorProto_DataType::FLOAT
                                        | TensorProto_DataType::UNDEFINED => {
                                            log::info!(
                                                    "transferring input {} for op {} to f32 attribute (initializer data type: {:?})",
                                                    attr_name,
                                                    op,
                                                    data_type
                                                );
                                            let value: Vec<f32> =
                                                tensor_proto.get_float_data().to_vec();
                                            attributes
                                                .push(attribute(attr_names[input_index], value));
                                        }
                                        _ => {
                                            return Err(OptimizerError::InvalidDataType {
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

                    let new_node: Self = Self {
                        inputs: vec![inputs[0].clone()],
                        definition: NodeDefinition::Operator(
                            op_index,
                            Box::new(OperatorDefinition {
                                proto: Cow::Owned(new_proto),
                                output_shapes: op_def.output_shapes.clone(),
                            }),
                        ),
                    };

                    Ok(Arc::new(new_node))
                }

                _ => Ok(Arc::new(Self {
                    inputs,
                    definition: NodeDefinition::Operator(op_index, op_def),
                })),
            }
        } else {
            Ok(Arc::new(Self { definition, inputs }))
        }
    }
}
