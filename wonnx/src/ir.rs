//! DAG representation of ops allowing for transformations and optimizations before compilation
use crate::tensor::{DataTypeError, Shape, TensorData};
use std::borrow::{Borrow, Cow};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::ptr;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

#[derive(Clone, Debug)]
pub enum AttributeValue<'a> {
    F32(f32),
    I64(i64),
    I64s(Cow<'a, [i64]>),
    F32s(Cow<'a, [f32]>),
    String(String),
    Tensor(Tensor<'a>),
}

impl<'a> AttributeValue<'a> {
    pub fn into_static(self) -> AttributeValue<'static> {
        match self {
            AttributeValue::F32(f) => AttributeValue::F32(f),
            AttributeValue::I64(f) => AttributeValue::I64(f),
            AttributeValue::I64s(f) => AttributeValue::I64s(Cow::Owned(f.into_owned())),
            AttributeValue::F32s(f) => AttributeValue::F32s(Cow::Owned(f.into_owned())),
            AttributeValue::String(s) => AttributeValue::String(s),
            AttributeValue::Tensor(t) => AttributeValue::Tensor(t.into_static()),
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for f32 {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::F32(v) = value {
            Ok(*v)
        } else {
            Err(())
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for i64 {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::I64(v) = value {
            Ok(*v)
        } else {
            Err(())
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for Vec<i64> {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::I64s(v) = value {
            Ok(v.to_vec())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for String {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::String(v) = value {
            Ok(v.clone())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for Tensor<'static> {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::Tensor(v) = value {
            Ok(v.clone().into_static())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&AttributeValue<'_>> for Vec<f32> {
    type Error = ();

    fn try_from(value: &AttributeValue) -> Result<Self, Self::Error> {
        if let AttributeValue::F32s(v) = value {
            Ok(v.to_vec())
        } else {
            Err(())
        }
    }
}

impl From<i64> for AttributeValue<'_> {
    fn from(value: i64) -> Self {
        AttributeValue::I64(value)
    }
}

impl From<f32> for AttributeValue<'_> {
    fn from(value: f32) -> Self {
        AttributeValue::F32(value)
    }
}

impl From<Vec<f32>> for AttributeValue<'_> {
    fn from(value: Vec<f32>) -> Self {
        AttributeValue::F32s(Cow::Owned(value))
    }
}

impl From<Vec<i64>> for AttributeValue<'_> {
    fn from(value: Vec<i64>) -> Self {
        AttributeValue::I64s(Cow::Owned(value))
    }
}

#[derive(Clone)]
pub struct OperatorDefinition {
    pub(crate) op_type: String,
    pub(crate) attributes: HashMap<String, AttributeValue<'static>>,
    pub(crate) output_shapes: Vec<Shape>,
    pub(crate) display_name: String,
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

    pub fn set_attribute(&mut self, name: &str, inputs: impl Into<AttributeValue<'static>>) {
        let attribute: AttributeValue = inputs.into();
        self.attributes.insert(name.to_string(), attribute);
    }

    pub fn set_op_type(&mut self, op_type: &str) {
        self.op_type = op_type.to_string();
    }

    pub fn get_op_type(&self) -> &str {
        &self.op_type
    }

    pub fn get_attribute_value<'a, T>(
        &'a self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError>
    where
        T: TryFrom<&'a AttributeValue<'a>>,
    {
        match (self.attributes.get(attribute), default) {
            (Some(attribute_value), _) => {
                Ok(
                    T::try_from(attribute_value).map_err(|_| AttributeNotFoundError {
                        attribute: attribute.to_string(),
                        node_name: self.get_display_name().to_string(),
                    })?,
                )
            }
            (None, Some(default_value)) => Ok(default_value),
            (None, None) => Err(AttributeNotFoundError {
                attribute: attribute.to_string(),
                node_name: self.get_display_name().to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor<'a> {
    pub(crate) data: TensorData<'a>,
    pub(crate) dims: Vec<usize>,
    pub(crate) display_name: String,
}

impl<'a> Tensor<'a> {
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn into_static(self) -> Tensor<'static> {
        Tensor {
            data: self.data.into_static(),
            dims: self.dims,
            display_name: self.display_name,
        }
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
