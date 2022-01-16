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

impl<'model> Node<'model> {
    /// Optimize a graph
    pub fn optimize(self: Arc<Self>) -> Result<Arc<Self>, OptimizerError> {
        let mut path = vec![];
        self.optimize_sequence(&mut path)
    }

    /// Optimize a branch of a graph
    fn optimize_sequence(
        self: Arc<Self>,
        path: &mut Vec<String>,
    ) -> Result<Arc<Self>, OptimizerError> {
        log::info!("Optimize {:?} path={:?}", self.definition, path);

        if let NodeDefinition::Operator(_, op_def) = &self.definition {
            match op_def.proto.get_op_type() {
                // Identity: we just remove these nodes and connect the input of the destination node to our input's output
                // A -> Identity -> B => A -> B
                "Identity" => {
                    if self.inputs.len() != 1 {
                        return Err(OptimizerError::NoInputs);
                    }
                    return self.inputs[0].source_node.clone().optimize_sequence(path);
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

                    return self.inputs[0].source_node.clone().optimize_sequence(path);
                }
                _ => {}
            }

            path.push(op_def.proto.get_op_type().to_string());
        }

        // Optimize inputs
        let new_inputs = self
            .inputs
            .iter()
            .map(|input| {
                Ok(Input {
                    source_node: input.source_node.clone().optimize_sequence(path)?,
                    output_index: input.output_index,
                })
            })
            .collect::<Result<Vec<Input>, OptimizerError>>()?;

        if let NodeDefinition::Operator(_, _) = &self.definition {
            path.pop();
        }

        Self::new_optimized(self.definition.clone(), new_inputs)
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
