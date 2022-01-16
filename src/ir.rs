use crate::onnx::{ModelProto, NodeProto, TensorProto, ValueInfoProto};
use crate::utils::Shape;
use std::borrow::Cow;
use std::fmt::Debug;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

#[derive(Clone)]
pub struct OperatorDefinition<'model> {
    pub(crate) proto: Cow<'model, NodeProto>,
    pub(crate) output_shapes: Vec<Shape>,
}

impl<'model> OperatorDefinition<'model> {
    pub fn from(
        node: Cow<'model, NodeProto>,
        value_shapes: &HashMap<&'model str, Shape>,
    ) -> Result<OperatorDefinition<'model>, IrError> {
        let mut output_shapes: Vec<Shape> = Vec::with_capacity(node.get_output().len());
        for output_name in node.get_output() {
            if !value_shapes.contains_key(output_name.as_str()) {
                return Err(IrError::OutputNodeNotFound(output_name.to_string()));
            }

            output_shapes.push(value_shapes[&output_name.as_str()].clone());
        }
        Ok(OperatorDefinition {
            proto: node,
            output_shapes,
        })
    }
}

#[derive(Clone)]
pub enum NodeDefinition<'model> {
    Operator(usize, Box<OperatorDefinition<'model>>),
    Tensor(usize, &'model TensorProto),
    Input(usize, &'model ValueInfoProto),
    Outputs { names: Vec<&'model str> },
    Missing, // A missing input (optional)
}

static MISSING_OPTIONAL_INPUT: NodeDefinition<'static> = NodeDefinition::Missing;

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum NodeIdentifier {
    Op(usize),
    Tensor(usize),
    Input(usize),
    Outputs,
    Missing,
}

#[derive(Clone)]
pub struct Input<'model> {
    pub source_node: Arc<Node<'model>>,
    pub output_index: usize,
}

pub struct Node<'model> {
    pub definition: NodeDefinition<'model>,
    pub inputs: Vec<Input<'model>>,
}

#[derive(Debug, Error)]
pub enum IrError {
    #[error("output node for output {0} not found")]
    OutputNodeNotFound(String),

    #[error("could not find node corresponding to input {input_name} of node {target_node_name}")]
    InputNodeNotFound {
        target_node_name: String,
        input_name: String,
    },
}

impl<'m> NodeDefinition<'m> {
    pub fn get_identifier(&self) -> NodeIdentifier {
        match self {
            NodeDefinition::Operator(idx, _) => NodeIdentifier::Op(*idx),
            NodeDefinition::Tensor(idx, _) => NodeIdentifier::Tensor(*idx),
            NodeDefinition::Input(idx, _) => NodeIdentifier::Input(*idx),
            NodeDefinition::Outputs { .. } => NodeIdentifier::Outputs,
            NodeDefinition::Missing => NodeIdentifier::Missing,
        }
    }

    pub fn get_name(&self) -> Cow<'_, str> {
        match self {
            NodeDefinition::Operator(_, op_def) => Cow::from(op_def.proto.get_name()),
            NodeDefinition::Tensor(_, t) => Cow::from(t.get_name()),
            NodeDefinition::Input(_, i) => Cow::from(i.get_name()),
            NodeDefinition::Outputs { .. } => Cow::from(" "),
            NodeDefinition::Missing => Cow::from(""),
        }
    }

    pub fn output_name(&self, output_index: usize) -> Cow<'_, str> {
        match self {
            NodeDefinition::Operator(_, op_def) => {
                Cow::Borrowed(&op_def.proto.get_output()[output_index])
            }
            NodeDefinition::Tensor(_, proto) => Cow::from(proto.get_name()),
            NodeDefinition::Input(_, proto) => Cow::from(proto.get_name()),
            NodeDefinition::Outputs { .. } => panic!("can't get output name for outputs node"),
            NodeDefinition::Missing => panic!("can't get output name for missing node"),
        }
    }
}

impl<'model> Node<'model> {
    pub fn new(variant: NodeDefinition<'model>) -> Node<'model> {
        Node {
            definition: variant,
            inputs: vec![],
        }
    }

    pub fn definition(&self) -> &NodeDefinition<'model> {
        &self.definition
    }

    /// Construct part of the intermediate representation tree for the indicated node.
    pub fn from_node<'a>(
        model: &'model ModelProto,
        node: Cow<'model, NodeProto>,
        node_index: usize,
        value_shapes: &HashMap<&'model str, Shape>,
        node_definitions_by_output: &'a HashMap<String, NodeDefinition<'model>>,
        nodes_by_name: &mut HashMap<String, Arc<Node<'model>>>,
    ) -> Result<Arc<Node<'model>>, IrError> {
        let node_name = node.get_name();
        // Did we already translate this node before?
        if nodes_by_name.contains_key(node_name) {
            let n = nodes_by_name.get(node_name).unwrap();
            return Ok(n.clone());
        }

        let inputs: Result<Vec<Input<'model>>, IrError> = node
            .get_input()
            .iter()
            .map(|input_name: &'model String| {
                let source_node_definition = node_definitions_by_output
                    .get(&input_name.to_string())
                    .unwrap_or(&MISSING_OPTIONAL_INPUT);

                Ok(match source_node_definition {
                    // The source is another op - continue translating that node
                    NodeDefinition::Operator(index, source_node_proto) => Input {
                        source_node: Node::from_node(
                            model,
                            source_node_proto.proto.clone(),
                            *index,
                            value_shapes,
                            node_definitions_by_output,
                            nodes_by_name,
                        )?,
                        output_index: source_node_proto
                            .proto
                            .get_output()
                            .iter()
                            .position(|s| s == input_name)
                            .ok_or_else(|| IrError::OutputNodeNotFound(input_name.to_string()))?,
                    },
                    _ => {
                        // The source is an initializer or model onput
                        let source_name = source_node_definition.get_name();

                        Input {
                            output_index: 0,
                            // Did we already translate this node?
                            source_node: match nodes_by_name.get(&source_name.to_string()) {
                                Some(node) => node.clone(),
                                None => {
                                    let node = Arc::new(Node::new(source_node_definition.clone()));
                                    nodes_by_name.insert(source_name.into(), node.clone());
                                    node
                                }
                            },
                        }
                    }
                })
            })
            .collect();

        let translated = Arc::new(Node {
            definition: NodeDefinition::Operator(
                node_index,
                Box::new(OperatorDefinition::from(node.clone(), value_shapes)?),
            ),
            inputs: inputs?,
        });
        nodes_by_name.insert(node_name.to_string(), translated.clone());
        Ok(translated)
    }

    /// Construct an intermediate representation graph for calculating the output with the specified name.
    pub fn from_model(model: &'model ModelProto) -> Result<Arc<Node<'model>>, IrError> {
        // Collect value shapes
        let mut value_shapes: HashMap<&'model str, Shape> = HashMap::new();
        for vi in model.get_graph().get_value_info() {
            value_shapes.insert(vi.get_name(), vi.get_shape());
        }

        for vi in model.get_graph().get_output() {
            let output_name = vi.get_name();
            if !output_name.is_empty() {
                value_shapes.insert(output_name, vi.get_shape());
            }
        }

        // Sort nodes by output nodes
        let mut node_definitions_by_output = HashMap::<String, NodeDefinition<'model>>::new();
        for (index, node) in model.get_graph().get_node().iter().enumerate() {
            let node_def = NodeDefinition::Operator(
                index,
                Box::new(OperatorDefinition::from(
                    Cow::Borrowed(node),
                    &value_shapes,
                )?),
            );
            for output in node.get_output() {
                if !output.is_empty() {
                    node_definitions_by_output.insert(output.to_string(), node_def.clone());
                }
            }
        }

        // Collect intializer info
        for (index, initializer) in model.get_graph().get_initializer().iter().enumerate() {
            log::info!("Initializer {}", initializer.get_name());
            node_definitions_by_output.insert(
                initializer.get_name().to_string(),
                NodeDefinition::Tensor(index, initializer),
            );
        }

        // Collect input name
        for (index, input) in model.get_graph().get_input().iter().enumerate() {
            if !node_definitions_by_output.contains_key(input.get_name()) {
                log::info!("Input {}", input.get_name());
                node_definitions_by_output.insert(
                    input.get_name().to_string(),
                    NodeDefinition::Input(index, input),
                );
            } else {
                log::info!(
                    "Skipping input definition {}: already defined",
                    input.get_name()
                );
            }
        }

        let mut nodes_by_name = HashMap::new();

        let output_nodes: Result<Vec<Input<'model>>, IrError> = model
            .get_graph()
            .get_output()
            .iter()
            .map(|output_def| {
                let output_name_string = output_def.get_name().to_string();
                let (output_node_index, output_node) = model
                    .get_graph()
                    .get_node()
                    .iter()
                    .enumerate()
                    .find(|(_index, x)| -> bool { x.get_output().contains(&output_name_string) })
                    .ok_or(IrError::OutputNodeNotFound(output_name_string))?;

                let source_node = Node::<'model>::from_node(
                    model,
                    Cow::Borrowed(output_node),
                    output_node_index,
                    &value_shapes,
                    &node_definitions_by_output,
                    &mut nodes_by_name,
                )?;

                let output_index = output_node
                    .get_output()
                    .iter()
                    .position(|s| s == output_def.get_name())
                    .ok_or_else(|| {
                        IrError::OutputNodeNotFound(output_def.get_name().to_string())
                    })?;

                Ok(Input {
                    source_node,
                    output_index,
                })
            })
            .collect();

        let output_names: Vec<&str> = model
            .get_graph()
            .get_output()
            .iter()
            .map(|output_def| output_def.get_name())
            .collect();

        Ok(Arc::new(Node {
            definition: NodeDefinition::Outputs {
                names: output_names,
            },
            inputs: output_nodes?,
        }))
    }

    pub fn output_shape(&self, output_index: usize) -> Shape {
        match (&self.definition, output_index) {
            (NodeDefinition::Operator(_, op_def), index) => op_def.output_shapes[index].clone(),
            (NodeDefinition::Tensor(_, tensor_proto), 0) => Shape::from(tensor_proto.get_dims()),
            (NodeDefinition::Input(_, input_proto), 0) => input_proto.get_shape(),
            (NodeDefinition::Outputs { .. }, _) => panic!("output node has no outputs!"),
            (_, _) => panic!("node has no output at index {}", output_index),
        }
    }
}

impl<'model> Debug for NodeDefinition<'model> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeDefinition::Operator(idx, def) => {
                write!(
                    f,
                    "Op #{}: {} ({})",
                    idx,
                    def.proto.get_name(),
                    def.proto.get_op_type()
                )
            }
            NodeDefinition::Tensor(idx, def) => write!(f, "Tensor #{}: {}", idx, def.get_name()),
            NodeDefinition::Input(idx, def) => write!(f, "Input #{}: {}", idx, def.get_name()),
            NodeDefinition::Outputs { .. } => write!(f, "Outputs"),
            NodeDefinition::Missing => write!(f, "Missing (optional)"),
        }
    }
}

impl<'model> PartialEq for Node<'model> {
    fn eq(&self, other: &Self) -> bool {
        self.definition
            .get_identifier()
            .eq(&other.definition.get_identifier())
    }
}
