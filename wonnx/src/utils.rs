use protobuf::ProtobufEnum;
use protobuf::RepeatedField;

use crate::onnx;
use crate::onnx::OperatorSetIdProto;
use crate::onnx::TensorProto_DataType;
use crate::onnx::ValueInfoProto;
use std::borrow::Cow;
use std::convert::From;
use std::convert::Into;
use std::fmt::Display;
use std::str::from_utf8;
use thiserror::Error;

/* Minimum size of a buffer you can create with wgpu. Creating buffers smaller than this leads to panic "Validation
* error: buffer binding size X is less than minimum 64" in Device::create_bind_group */
pub const MINIMUM_BUFFER_SIZE_BYTES: u64 = 64;

#[derive(Debug, Clone)]
pub struct Shape {
    pub dims: Vec<u64>,
    pub data_type: ScalarType,
}

impl Shape {
    pub fn from(data_type: ScalarType, dims: &[i64]) -> Shape {
        Shape {
            data_type,
            dims: dims.iter().map(|x| *x as u64).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn buffer_bytes(&self) -> usize {
        // Dividing by 4 since that is the size of an f32 in bytes
        (self.element_count() as usize) * self.data_type.stride()
    }

    pub fn dim(&self, idx: usize) -> u64 {
        self.dims[idx]
    }

    pub fn chunks(&self) -> Vec<u64> {
        let mut chunk = vec![];
        let ds = &self.dims;
        for i in 1..self.dims.len() {
            chunk.push(ds[i..].iter().product::<u64>());
        }
        chunk.push(1);
        chunk
    }
}

pub enum InputTensor<'a> {
    F32(Cow<'a, [f32]>),
    I32(Cow<'a, [i32]>),
}

impl<'a> From<&'a [f32]> for InputTensor<'a> {
    fn from(a: &'a [f32]) -> Self {
        InputTensor::F32(Cow::Borrowed(a))
    }
}

impl<'a> From<&'a [i32]> for InputTensor<'a> {
    fn from(a: &'a [i32]) -> Self {
        InputTensor::I32(Cow::Borrowed(a))
    }
}

#[derive(Error, Debug)]
pub enum DataTypeError {
    #[error("the ONNX scalar data type '{0:?}' is not supported")]
    NotSupported(TensorProto_DataType),

    #[error("the ONNX data type '{0}' is not recognized")]
    NotRecognized(i32),

    #[error("type is undefined")]
    Undefined,
}

/// Data type for a single number
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    I64,
    I32,
}

impl ScalarType {
    pub fn from_i32(onnx: i32) -> Result<ScalarType, DataTypeError> {
        let onnx_dt =
            TensorProto_DataType::from_i32(onnx).ok_or(DataTypeError::NotRecognized(onnx))?;
        Self::from(onnx_dt)
    }

    pub fn from(onnx: TensorProto_DataType) -> Result<ScalarType, DataTypeError> {
        Ok(match onnx {
            TensorProto_DataType::FLOAT => ScalarType::F32,
            TensorProto_DataType::INT64 => ScalarType::I64,
            TensorProto_DataType::INT32 => ScalarType::I32,
            _ => return Err(DataTypeError::NotSupported(onnx)),
        })
    }

    pub fn stride(&self) -> usize {
        match self {
            ScalarType::F32 => 4,
            ScalarType::I32 => 4,
            ScalarType::I64 => 8,
        }
    }

    pub fn wgsl_type_name(&self) -> &'static str {
        match self {
            ScalarType::F32 => "f32",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
        }
    }
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.wgsl_type_name())
    }
}

/// Represents a WGSL data type that can be used in a shader to perform an operation on multiple scalars at once. The
/// larger the data type, the more efficiently the GPU can perform operations. However, large data types require the size
/// of the data that is being worked on to be a multiple of the data type (e.g. a vec4 can be used to work on a vector of
/// 256 elements, but when used on a vector of 255 elements calculation would overflow if the shader doesn't take this
/// into account). The WGSL declaration looks like:
///
/// struct Block {
///     data: [[stride( dt.size_bytes() )]] dt.wgsl_type_name();
/// };
pub enum MultiType {
    Scalar(ScalarType),
    Vec(ScalarType, usize),
    Mat(ScalarType, usize, usize),
}

impl MultiType {
    /// Determine the appropriate data type given the data size
    pub fn for_size(n: usize, scalar: ScalarType) -> MultiType {
        let d = num::integer::gcd(n, 4);
        match d {
            1 => MultiType::Scalar(scalar),
            2 | 4 => MultiType::Vec(scalar, d),
            /* 3 can't occur here because it is not a divisor of 4. Even so, we wouldn't be able to use vec3, because
            its stride is 16 instead of the expected 12, which would require padding to work properly. */
            _ => unreachable!(),
        }
    }

    /// Size (in bytes) of the data type (useful for setting the 'stride' in WGSL)
    pub fn stride(&self) -> usize {
        match self {
            MultiType::Scalar(s) => s.stride(),

            // FIXME: this may not always be right!
            MultiType::Vec(st, n) => st.stride() * n,
            MultiType::Mat(st, w, h) => st.stride() * w * h,
        }
    }

    /// Name of this data type in WGSL
    pub fn wgsl_type_name(&self) -> String {
        match self {
            MultiType::Scalar(s) => s.wgsl_type_name().to_string(),
            MultiType::Vec(st, n) => format!("vec{}<{}>", n, st.wgsl_type_name()),
            MultiType::Mat(st, w, h) => format!("mat{}x{}<{}>", w, h, st.wgsl_type_name()),
        }
    }

    /// The number of elements in this data type
    pub fn elements(&self) -> usize {
        match self {
            MultiType::Scalar(_) => 1,

            // FIXME: this may not always be right
            MultiType::Vec(_, n) => *n,
            &MultiType::Mat(_, w, h) => w * h,
        }
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.dims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("x")
        )
    }
}

#[derive(Error, Debug)]
#[error("did not find attribute '{attribute}' for node '{node_name}'")]
pub struct AttributeNotFoundError {
    attribute: String,
    node_name: String,
}

pub fn get_attribute<T: std::convert::From<onnx::AttributeProto>>(
    attribute: &str,
    default: Option<T>,
    node: &onnx::NodeProto,
) -> Result<T, AttributeNotFoundError> {
    match (
        node.get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute),
        default,
    ) {
        (Some(attr), _) => Ok(attr.clone().into()),
        (None, Some(default_attr)) => Ok(default_attr),
        (None, None) => Err(AttributeNotFoundError {
            attribute: attribute.to_string(),
            node_name: node.get_name().to_string(),
        }),
    }
}

/// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
pub fn ceil(num: u64, div: u64) -> u64 {
    num / div + (num % div != 0) as u64
}

impl ValueInfoProto {
    pub fn get_shape(&self) -> Result<Shape, DataTypeError> {
        Ok(match &self.get_field_type().value {
            Some(t) => match t {
                onnx::TypeProto_oneof_value::tensor_type(tensor_proto) => Shape::from(
                    ScalarType::from_i32(tensor_proto.get_elem_type())?,
                    self.get_field_type()
                        .get_tensor_type()
                        .get_shape()
                        .get_dim()
                        .iter()
                        .map(|x| x.get_dim_value() as i64)
                        .collect::<Vec<i64>>()
                        .as_slice(),
                ),
                onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => return Err(DataTypeError::Undefined),
        })
    }
}

/// Shorthand method to define an ONNX tensor with the specified name and shape (data type is f32)
pub fn tensor(name: &str, dimensions: &[i64]) -> onnx::ValueInfoProto {
    tensor_of_type(name, dimensions, TensorProto_DataType::FLOAT)
}

/// Shorthand method to define an ONNX tensor with the specified name, shape and data type
pub fn tensor_of_type(
    name: &str,
    dimensions: &[i64],
    data_type: TensorProto_DataType,
) -> onnx::ValueInfoProto {
    let mut dim_value = vec![];
    for dimension in dimensions {
        let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
        dim_channel.set_dim_value(*dimension);
        dim_value.push(dim_channel);
    }

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(dim_value));

    let mut type_proto_tensor = onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(data_type.value());
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut tensor = onnx::ValueInfoProto::new();
    tensor.set_name(name.to_string());
    tensor.set_field_type(type_proto);

    tensor
}

// Remove dimensions
pub fn initializer(name: &str, data: Vec<f32>) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    initializer.set_name(name.to_string());
    initializer.set_data_type(1); // FLOAT
    initializer.set_float_data(data);
    initializer
}

pub fn attribute(name: &str, inputs: impl Into<onnx::AttributeProto>) -> onnx::AttributeProto {
    let mut attributes: onnx::AttributeProto = inputs.into();
    attributes.set_name(name.to_string());
    attributes
}

pub fn node(
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    name: &str,
    op_type: &str,
    attributes: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    let mut node = crate::onnx::NodeProto::new();

    node.set_op_type(op_type.to_string());
    node.set_name(name.to_string());
    node.set_input(protobuf::RepeatedField::from(
        inputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_output(protobuf::RepeatedField::from(
        outputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_attribute(protobuf::RepeatedField::from(attributes));
    node
}

pub fn graph(
    inputs: Vec<onnx::ValueInfoProto>,
    outputs: Vec<onnx::ValueInfoProto>,
    infos: Vec<onnx::ValueInfoProto>,
    initializers: Vec<onnx::TensorProto>,
    nodes: Vec<onnx::NodeProto>,
) -> onnx::GraphProto {
    let mut graph = onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(nodes));
    graph.set_input(protobuf::RepeatedField::from(inputs));
    graph.set_value_info(protobuf::RepeatedField::from(infos));
    graph.set_output(protobuf::RepeatedField::from(outputs));
    graph.set_initializer(protobuf::RepeatedField::from(initializers));
    graph
}

pub fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    let mut model = crate::onnx::ModelProto::new();
    let mut onnx_opset_import = OperatorSetIdProto::new();
    onnx_opset_import.set_domain("".to_string());
    onnx_opset_import.set_version(13);
    model.set_opset_import(RepeatedField::from_slice(&[onnx_opset_import]));
    model.set_graph(graph);
    model
}

impl From<Vec<i64>> for onnx::AttributeProto {
    fn from(value: Vec<i64>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_ints(value);
        attributes
    }
}

impl From<Vec<f32>> for onnx::AttributeProto {
    fn from(value: Vec<f32>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_floats(value);
        attributes
    }
}

impl From<f32> for onnx::AttributeProto {
    fn from(value: f32) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_f(value);
        attributes
    }
}

impl From<i64> for onnx::AttributeProto {
    fn from(value: i64) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_i(value);
        attributes
    }
}

impl From<String> for onnx::AttributeProto {
    fn from(value: String) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.into_bytes());
        attributes
    }
}

impl From<&str> for onnx::AttributeProto {
    fn from(value: &str) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.to_string().into_bytes());
        attributes
    }
}

impl From<onnx::AttributeProto> for Vec<i64> {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_ints().to_vec()
    }
}

impl From<onnx::AttributeProto> for Vec<f32> {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_floats().to_vec()
    }
}

impl From<onnx::AttributeProto> for f32 {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_f()
    }
}

impl From<onnx::AttributeProto> for i64 {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_i()
    }
}

impl From<onnx::AttributeProto> for String {
    fn from(value: onnx::AttributeProto) -> Self {
        from_utf8(value.get_s()).unwrap().to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{attribute, graph, initializer, model, node, tensor};

    #[test]
    fn test_model() {
        // USER INPUT

        let n = 5;
        let c = 1;
        let mut input_data = std::collections::HashMap::new();

        let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
        input_data.insert("X".to_string(), data.as_slice().into());

        // ONNX INPUTS
        let shape = vec![1, c, n, n];
        let kernel_n = 3;
        let m = 1;
        let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();
        let conv_model = model(graph(
            vec![tensor("X", &shape)],
            vec![tensor("Y", &[1, 1, 3, 3])],
            vec![tensor("W", &[2, c, 3, 3])],
            vec![initializer("W", data_w)],
            vec![node(
                vec!["X", "W"],
                vec!["Y"],
                "conv",
                "Conv",
                vec![attribute("kernel_shape", vec![3, 3])],
            )],
        ));

        // LOGIC

        let session = pollster::block_on(crate::Session::from_model(conv_model))
            .expect("Session did not create");

        let result = pollster::block_on(session.run(&input_data)).unwrap();

        assert_eq!(
            result["Y"],
            [54., 63., 72., 99., 108., 117., 144., 153., 162.]
        );
    }
}
