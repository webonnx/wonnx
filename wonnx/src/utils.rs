//! Various utilities to deal with the ONNX format structure
use protobuf::ProtobufEnum;
use protobuf::RepeatedField;
use serde::Serialize;

use crate::onnx;
use crate::onnx::OperatorSetIdProto;
use crate::onnx::TensorProto;
use crate::onnx::TensorProto_DataType;
use crate::onnx::ValueInfoProto;
use num::FromPrimitive;
use std::borrow::Cow;
use std::convert::From;
use std::convert::Into;
use std::convert::TryFrom;
use std::fmt::Display;
use std::str::from_utf8;
use thiserror::Error;

/* Minimum size of a buffer you can create with wgpu. Creating buffers smaller than this leads to panic "Validation
* error: buffer binding size X is less than minimum 64" in Device::create_bind_group */
pub(crate) const MINIMUM_BUFFER_SIZE_BYTES: u64 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Computes the shape to which all provided shapes can be broadcast (if it exists)
    /// Inspired by https://github.com/sonos/tract/blob/68db0209c9ffd1b91dff82884f4ae03b3622dd34/core/src/broadcast.rs#L5
    pub fn multi_broadcast(shapes: &[Shape]) -> Option<Shape> {
        if shapes.is_empty() {
            return None;
        }

        let max_rank = shapes.iter().map(|x| x.rank()).max().unwrap_or(0);
        let mut shape: Vec<i64> = Vec::with_capacity(max_rank);

        // Shapes must all have the same data type
        let data_type = shapes[0].data_type;
        for s in shapes {
            if s.data_type != data_type {
                return None;
            }
        }

        for i in 0..max_rank {
            let mut wanted_size = 1;
            for shape in shapes {
                let rank = shape.rank();
                let dim = if i < rank { shape.dim(rank - i - 1) } else { 1 };

                if dim != 1 {
                    if wanted_size != 1 && dim != wanted_size {
                        return None;
                    }
                    wanted_size = dim;
                }
            }
            shape.push(wanted_size as i64);
        }

        shape.reverse();
        Some(Shape::from(data_type, &shape))
    }

    pub(crate) fn left_padded_to(&self, x: u64, rank: usize) -> Shape {
        let mut dims = self.dims.clone();
        let current_rank = dims.len();
        dims.resize(rank, x);
        if rank > current_rank {
            dims.rotate_right(rank - current_rank);
        }
        Shape {
            dims,
            data_type: self.data_type,
        }
    }
}

pub enum InputTensor<'a> {
    F32(Cow<'a, [f32]>),
    I32(Cow<'a, [i32]>),
    I64(Cow<'a, [i64]>),
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

impl<'a> From<&'a [i64]> for InputTensor<'a> {
    fn from(a: &'a [i64]) -> Self {
        InputTensor::I64(Cow::Borrowed(a))
    }
}

#[derive(Error, Debug)]
pub enum TensorConversionError {
    #[error("could not convert to the requested type becaue a value could not be represented in the target type")]
    OutOfBoundsError,

    #[error("cold not return the requested type; conversions cannot be done for slices")]
    DataTypeError,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum OutputTensor {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl TryFrom<OutputTensor> for Vec<f32> {
    type Error = TensorConversionError;

    /// Convert OutputTensor into a Vec<f32>, possibly converting integer tensors if the values fit
    fn try_from(value: OutputTensor) -> Result<Self, Self::Error> {
        match value {
            OutputTensor::F32(floats) => Ok(floats),
            OutputTensor::I32(ints) => ints
                .into_iter()
                .map(|i| f32::from_i32(i).ok_or(TensorConversionError::OutOfBoundsError))
                .collect::<Result<_, _>>(),
            OutputTensor::I64(ints) => ints
                .into_iter()
                .map(|i| f32::from_i64(i).ok_or(TensorConversionError::OutOfBoundsError))
                .collect::<Result<_, _>>(),
        }
    }
}

/// Convert &OutputTensor into an &[f32]. Because we cannot store converted results, this operation does not attempt
/// to convert the tensor if the values are of a different type
impl<'a> TryFrom<&'a OutputTensor> for &'a [f32] {
    type Error = TensorConversionError;

    fn try_from(value: &'a OutputTensor) -> Result<Self, Self::Error> {
        match value {
            OutputTensor::F32(floats) => Ok(floats.as_slice()),
            OutputTensor::I32(_) | OutputTensor::I64(_) => {
                Err(TensorConversionError::DataTypeError)
            }
        }
    }
}

impl<'a> From<&InputTensor<'a>> for OutputTensor {
    fn from(input: &InputTensor<'a>) -> Self {
        match input {
            InputTensor::F32(fs) => OutputTensor::F32(fs.to_vec()),
            InputTensor::I32(fs) => OutputTensor::I32(fs.to_vec()),
            InputTensor::I64(fs) => OutputTensor::I64(fs.to_vec()),
        }
    }
}

#[derive(Error, Debug)]
pub enum DataTypeError {
    #[error("the ONNX scalar data type '{0:?}' is not supported")]
    NotSupported(TensorProto_DataType),

    #[error("the ONNX data type '{0}' is not recognized")]
    NotRecognized(i32),

    #[error("encountered parametrized dimensions '{0}'; this is not currently supported (this may be solved by running onnx-simplifier on the model first)")]
    ParametrizedDimensionUnsupported(String),

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

    pub fn to_datatype(&self) -> TensorProto_DataType {
        match self {
            ScalarType::F32 => TensorProto_DataType::FLOAT,
            ScalarType::I64 => TensorProto_DataType::INT64,
            ScalarType::I32 => TensorProto_DataType::INT32,
        }
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

    pub fn is_float(&self) -> bool {
        match self {
            ScalarType::F32 => true,
            ScalarType::I32 | ScalarType::I64 => false,
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
pub(crate) enum MultiType {
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

pub trait NodeAttributes {
    fn has_attribute(&self, attribute_name: &str) -> bool;
    fn get_attribute_value<T: std::convert::From<onnx::AttributeProto>>(
        &self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError>;
}

impl NodeAttributes for onnx::NodeProto {
    fn has_attribute(&self, attribute_name: &str) -> bool {
        self.get_attribute()
            .iter()
            .any(|attr| attr.get_name() == attribute_name)
    }

    fn get_attribute_value<T: std::convert::From<onnx::AttributeProto>>(
        &self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError> {
        match (
            self.get_attribute()
                .iter()
                .find(|attr| attr.get_name() == attribute),
            default,
        ) {
            (Some(attr), _) => Ok(attr.clone().into()),
            (None, Some(default_attr)) => Ok(default_attr),
            (None, None) => Err(AttributeNotFoundError {
                attribute: attribute.to_string(),
                node_name: self.get_name().to_string(),
            }),
        }
    }
}

/// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
pub(crate) fn ceil(num: u64, div: u64) -> u64 {
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
                        .map(|x| {
                            if x.has_dim_param() {
                                return Err(DataTypeError::ParametrizedDimensionUnsupported(
                                    x.get_dim_param().to_string(),
                                ));
                            }
                            Ok(x.get_dim_value())
                        })
                        .collect::<Result<Vec<i64>, DataTypeError>>()?
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

pub fn initializer(name: &str, data: Vec<f32>, dimensions: Vec<i64>) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    assert_eq!(
        dimensions.iter().cloned().product::<i64>() as usize,
        data.len()
    );
    initializer.set_dims(dimensions);
    initializer.set_name(name.to_string());
    initializer.set_data_type(TensorProto_DataType::FLOAT.value());
    initializer.set_float_data(data);
    initializer
}

pub fn initializer_int64(name: &str, data: Vec<i64>, dimensions: Vec<i64>) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    assert_eq!(
        dimensions.iter().cloned().product::<i64>() as usize,
        data.len()
    );
    initializer.set_name(name.to_string());
    initializer.set_dims(dimensions);
    initializer.set_data_type(TensorProto_DataType::INT64.value());
    initializer.set_int64_data(data);
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
    mut infos: Vec<onnx::ValueInfoProto>,
    initializers: Vec<onnx::TensorProto>,
    nodes: Vec<onnx::NodeProto>,
) -> onnx::GraphProto {
    let mut graph = onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(nodes));
    graph.set_input(protobuf::RepeatedField::from(inputs));
    graph.set_output(protobuf::RepeatedField::from(outputs));

    // Auto-generate tensor information for initializers so users don't have to specify those
    for i in &initializers {
        infos.push(tensor_of_type(
            i.get_name(),
            i.get_dims(),
            onnx::TensorProto_DataType::from_i32(i.get_data_type()).unwrap(),
        ));
    }

    graph.set_initializer(protobuf::RepeatedField::from(initializers));
    graph.set_value_info(protobuf::RepeatedField::from(infos));
    graph
}

pub fn model_with_opset(graph: onnx::GraphProto, opset_version: i64) -> onnx::ModelProto {
    let mut model = crate::onnx::ModelProto::new();
    let mut onnx_opset_import = OperatorSetIdProto::new();
    onnx_opset_import.set_domain("".to_string());
    onnx_opset_import.set_version(opset_version);
    model.set_opset_import(RepeatedField::from_slice(&[onnx_opset_import]));
    model.set_graph(graph);
    model
}

pub fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    model_with_opset(graph, 13)
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

impl From<TensorProto> for onnx::AttributeProto {
    fn from(value: TensorProto) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_t(value);
        attributes
    }
}

impl From<onnx::AttributeProto> for Vec<i64> {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_ints().to_vec()
    }
}

impl From<onnx::AttributeProto> for TensorProto {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_t().clone()
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
    use crate::utils::{
        attribute, graph, initializer, model, node, tensor, OutputTensor, ScalarType, Shape,
    };

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
            vec![],
            vec![initializer("W", data_w, vec![m, c, 3, 3])],
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
            OutputTensor::F32(vec![54., 63., 72., 99., 108., 117., 144., 153., 162.])
        );
    }

    // Test cases for Shape::multi_broadcast, some inspired by https://github.com/sonos/tract/blob/68db0209c9ffd1b91dff82884f4ae03b3622dd34/core/src/broadcast.rs#L31
    #[test]
    pub fn test_multi_broadcast() {
        fn shape(s: &[i64]) -> Shape {
            Shape::from(ScalarType::F32, s)
        }

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[2, 3, 4, 5]), shape(&[])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[2, 3, 4, 5]), shape(&[5])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[2, 3, 4, 5]), shape(&[4, 5])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[4, 5]), shape(&[2, 3, 4, 5])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[1, 4, 5]), shape(&[2, 3, 4, 1])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[3, 4, 5]), shape(&[2, 1, 1, 1])]),
            Some(shape(&[2, 3, 4, 5])),
        );

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[3, 4, 5]), shape(&[2, 4, 1, 1])]),
            None
        );
    }
}
