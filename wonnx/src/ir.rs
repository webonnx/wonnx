//! DAG representation of ops allowing for transformations and optimizations before compilation
use crate::onnx::{self, GraphProto, ModelProto, NodeProto};
use crate::tensor::{to_tensor, DataTypeError, Shape, TensorData};
use std::borrow::{Borrow, Cow};
use std::fmt::Debug;
use std::hash::Hash;
use std::ptr;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

pub type AttributeValue = onnx::AttributeProto;

#[derive(Clone)]
pub struct OperatorDefinition {
    op_type: String,
    attributes: HashMap<String, AttributeValue>,
    output_shapes: Vec<Shape>,
    display_name: String,
}

#[derive(Error, Debug)]
#[error("did not find attribute '{attribute}' for node '{node_name}'")]
pub struct AttributeNotFoundError {
    attribute: String,
    node_name: String,
}

impl OperatorDefinition {
    pub fn new(
        op_type: &str,
        output_shapes: Vec<Shape>,
        display_name: String,
    ) -> OperatorDefinition {
        OperatorDefinition {
            op_type: op_type.to_string(),
            attributes: HashMap::new(),
            output_shapes,
            display_name,
        }
    }

    pub fn from(node: &NodeProto, output_shapes: Vec<Shape>) -> OperatorDefinition {
        assert_eq!(node.get_output().len(), output_shapes.len());
        let mut attributes = HashMap::new();
        for attr in node.get_attribute() {
            attributes.insert(attr.get_name().to_string(), attr.clone());
        }

        OperatorDefinition {
            op_type: node.get_op_type().to_string(),
            attributes,
            output_shapes,
            display_name: node.get_output()[0].to_string(),
        }
    }

    pub fn output_shapes(&self) -> &[Shape] {
        &self.output_shapes
    }

    pub fn get_display_name(&self) -> &str {
        &self.display_name
    }

    pub fn append_attributes_from(&mut self, rhs: &Self) {
        for (k, v) in rhs.attributes.iter() {
            self.attributes.insert(k.clone(), v.clone());
        }
    }

    pub fn set_attribute(&mut self, name: &str, inputs: impl Into<AttributeValue>) {
        let attribute: AttributeValue = inputs.into();
        self.attributes.insert(name.to_string(), attribute);
    }

    pub fn set_op_type(&mut self, op_type: &str) {
        self.op_type = op_type.to_string();
    }

    pub fn get_op_type(&self) -> &str {
        &self.op_type
    }

    pub fn get_attribute_value<'a, T: From<&'a AttributeValue>>(
        &'a self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError> {
        match (self.attributes.get(attribute), default) {
            (Some(attribute_value), _) => Ok(attribute_value.into()),
            (None, Some(default_value)) => Ok(default_value),
            (None, None) => Err(AttributeNotFoundError {
                attribute: attribute.to_string(),
                node_name: self.get_display_name().to_string(),
            }),
        }
    }
}

#[derive(Clone)]
pub struct Tensor<'a> {
    pub(crate) data: TensorData<'a>,
    pub(crate) dims: Vec<usize>,
    pub(crate) display_name: String,
}

impl<'a> Tensor<'a> {
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn shape(&self) -> Shape {
        Shape::from(self.data.scalar_type(), &self.dims)
    }

    pub fn display_name(&self) -> &str {
        self.display_name.borrow()
    }

    pub fn data(&self) -> &TensorData<'a> {
        &self.data
    }
}

#[derive(Clone)]
pub enum NodeDefinition<'model> {
    Operator(OperatorDefinition),
    Tensor(Tensor<'model>),
    Input { name: String, shape: Shape },
    Outputs { names: Vec<String> },
    Missing, // A missing input (optional)
}

#[derive(Clone, Debug)]
pub struct Input<'model> {
    pub source_node: Arc<Node<'model>>,
    pub output_index: usize,
}

#[derive(Debug)]
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

    #[error("issue with data types: {0}")]
    Type(#[from] DataTypeError),
}

impl<'m> NodeDefinition<'m> {
    pub fn get_display_name(&self) -> Cow<'_, str> {
        match self {
            // Nodes are identified by their first output's name, because node names are optional (and "only to be used
            // for diagnostic purposes" according to the ONNX IR specification) whereas output names are required and should be unique.
            NodeDefinition::Operator(op_def) => Cow::from(op_def.get_display_name()),
            NodeDefinition::Tensor(t) => Cow::from(t.display_name()),
            NodeDefinition::Input { name, .. } => Cow::from(name),
            NodeDefinition::Outputs { .. } => Cow::from(" "),
            NodeDefinition::Missing => Cow::from(""),
        }
    }
}

impl<'model> Node<'model> {
    pub fn new(variant: NodeDefinition<'model>, inputs: Vec<Input<'model>>) -> Node<'model> {
        Node {
            definition: variant,
            inputs,
        }
    }

    pub fn is_dynamic(&self) -> bool {
        matches!(
            self.definition,
            NodeDefinition::Operator(..) | NodeDefinition::Input { .. }
        )
    }

    pub fn is_constant(&self) -> bool {
        !self.is_dynamic()
            || (matches!(self.definition, NodeDefinition::Operator(..))
                && self.inputs.iter().all(|i| i.source_node.is_constant()))
    }

    pub fn definition(&self) -> &NodeDefinition<'model> {
        &self.definition
    }

    /// Construct part of the intermediate representation tree for the indicated node.
    pub fn from_node<'a>(
        node: Cow<'model, NodeProto>,
        value_shapes: &HashMap<&'model str, Shape>,
        node_definitions_by_output: &'a HashMap<String, Cow<'model, NodeProto>>,
        nodes_by_output_names: &mut HashMap<String, Arc<Node<'model>>>,
    ) -> Result<Arc<Node<'model>>, IrError> {
        for output_name in node.get_output() {
            if nodes_by_output_names.contains_key(output_name) {
                let n = nodes_by_output_names.get(output_name).unwrap();
                return Ok(n.clone());
            }
        }

        let inputs: Result<Vec<Input<'model>>, IrError> = node
            .get_input()
            .iter()
            .map(|input_name: &String| {
                let my_input_name = input_name.clone();

                // An empty input name signifies missing
                if input_name.is_empty() {
                    return Ok(Input {
                        source_node: Arc::new(Node::new(NodeDefinition::Missing, vec![])),
                        output_index: 0,
                    });
                }

                Ok(match node_definitions_by_output.get(&my_input_name) {
                    Some(source_node_proto) => {
                        // The source is another op - continue translating that node
                        Input {
                            source_node: Node::from_node(
                                source_node_proto.clone(),
                                value_shapes,
                                node_definitions_by_output,
                                nodes_by_output_names,
                            )?,
                            output_index: source_node_proto
                                .get_output()
                                .iter()
                                .position(|s| s == input_name)
                                .ok_or_else(|| {
                                    IrError::OutputNodeNotFound(input_name.to_string())
                                })?,
                        }
                    }
                    None => {
                        Input {
                            output_index: 0,
                            // Did we already translate this node?
                            source_node: match nodes_by_output_names.get(input_name) {
                                Some(node) => node.clone(),
                                None => {
                                    return Err(IrError::InputNodeNotFound {
                                        target_node_name: node.get_name().to_string(),
                                        input_name: input_name.clone(),
                                    })
                                }
                            },
                        }
                    }
                })
            })
            .collect();

        // Obtain output shapes
        let mut output_shapes: Vec<Shape> = Vec::with_capacity(node.get_output().len());
        for output_name in node.get_output() {
            if !value_shapes.contains_key(output_name.as_str()) {
                return Err(IrError::OutputNodeNotFound(output_name.to_string()));
            }

            output_shapes.push(value_shapes[&output_name.as_str()].clone());
        }

        let translated = Arc::new(Node {
            definition: NodeDefinition::Operator(OperatorDefinition::from(&node, output_shapes)),
            inputs: inputs?,
        });

        // Register the translated node by all of its output names
        for output_name in node.get_output() {
            nodes_by_output_names.insert(output_name.clone(), translated.clone());
        }

        Ok(translated)
    }

    /// Construct an intermediate representation graph for calculating the output with the specified name.
    pub fn from_model(
        model: &'model ModelProto,
        outputs: Option<&[String]>,
    ) -> Result<Arc<Node<'model>>, IrError> {
        let graph: &'model GraphProto = model.get_graph();

        // Collect value shapes
        let mut value_shapes: HashMap<&'model str, Shape> = HashMap::new();
        for vi in graph.get_value_info() {
            value_shapes.insert(vi.get_name(), vi.get_shape()?);
        }

        for vi in graph.get_output() {
            let output_name = vi.get_name();
            if !output_name.is_empty() {
                value_shapes.insert(output_name, vi.get_shape()?);
            }
        }

        // Sort nodes by output nodes
        let mut node_definitions_by_output = HashMap::<String, Cow<'model, NodeProto>>::new();
        for node in graph.get_node().iter() {
            for output in node.get_output() {
                if !output.is_empty() {
                    node_definitions_by_output.insert(output.to_string(), Cow::Borrowed(node));
                }
            }
        }

        let mut nodes_by_output_name = HashMap::new();

        // Translate initializers
        for initializer in graph.initializer.iter() {
            nodes_by_output_name.insert(
                initializer.get_name().to_string(),
                Arc::new(Node::new(
                    NodeDefinition::Tensor(to_tensor(initializer)?),
                    vec![],
                )),
            );
        }

        // Translate inputs
        for input in model.get_graph().get_input().iter() {
            if !nodes_by_output_name.contains_key(input.get_name()) {
                nodes_by_output_name.insert(
                    input.get_name().to_string(),
                    Arc::new(Node::new(
                        NodeDefinition::Input {
                            name: input.get_name().to_string(),
                            shape: input.get_shape()?,
                        },
                        vec![],
                    )),
                );
            } else {
                log::warn!(
                    "Skipping input definition {}: already defined",
                    input.get_name()
                );
            }
        }

        let output_names: Vec<String> = match outputs {
            Some(outputs) => outputs.to_vec(),
            None => model
                .get_graph()
                .get_output()
                .iter()
                .map(|x| x.get_name().to_string())
                .collect(),
        };

        let output_nodes: Result<Vec<Input<'model>>, IrError> = output_names
            .iter()
            .map(|output_name| {
                let output_node = model
                    .get_graph()
                    .get_node()
                    .iter()
                    .find(|x| -> bool { x.get_output().contains(output_name) })
                    .ok_or_else(|| IrError::OutputNodeNotFound(output_name.clone()))?;

                let source_node = Node::<'model>::from_node(
                    Cow::Borrowed(output_node),
                    &value_shapes,
                    &node_definitions_by_output,
                    &mut nodes_by_output_name,
                )?;

                let output_index = output_node
                    .get_output()
                    .iter()
                    .position(|s| s == output_name)
                    .ok_or_else(|| IrError::OutputNodeNotFound(output_name.clone()))?;

                Ok(Input {
                    source_node,
                    output_index,
                })
            })
            .collect();

        Ok(Arc::new(Node {
            definition: NodeDefinition::Outputs {
                names: output_names,
            },
            inputs: output_nodes?,
        }))
    }
}

impl<'model> Debug for NodeDefinition<'model> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeDefinition::Operator(def) => {
                write!(f, "op: {} ({})", def.get_display_name(), def.get_op_type())
            }
            NodeDefinition::Tensor(def) => write!(f, "tensor {}", def.display_name()),
            NodeDefinition::Input { name, .. } => write!(f, "input {}", name),
            NodeDefinition::Outputs { .. } => write!(f, "outputs"),
            NodeDefinition::Missing => write!(f, "missing (optional)"),
        }
    }
}

/// Wrap an Arc<Node> in a struct so we can implement pointer-based comparison for it, and use them as keys in a HashSet/HashMap
#[derive(Clone)]
pub struct NodeIdentifier<'model>(Arc<Node<'model>>);

impl<'model> Debug for NodeIdentifier<'model> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NodeIdentifier")
            .field(&Arc::as_ptr(&self.0))
            .field(&self.0.definition.get_display_name())
            .finish()
    }
}

impl<'model> Hash for NodeIdentifier<'model> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        ptr::hash(Arc::as_ptr(&self.0), state)
    }
}

impl<'model> PartialEq for NodeIdentifier<'model> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<'model> Eq for NodeIdentifier<'model> {}

impl<'model> Node<'model> {
    pub fn identifier(self: &Arc<Self>) -> NodeIdentifier<'model> {
        NodeIdentifier(self.clone())
    }
}
