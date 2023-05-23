//! Various basic data types
use num::FromPrimitive;
use serde::Serialize;
use std::borrow::Cow;
use std::convert::From;
use std::convert::TryFrom;
use std::fmt::Display;
use thiserror::Error;

/* Minimum size of a buffer you can create with wgpu. Creating buffers smaller than this leads to panic "Validation
* error: buffer binding size X is less than minimum 64" in Device::create_bind_group */
pub(crate) const MINIMUM_BUFFER_SIZE_BYTES: u64 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
    pub data_type: ScalarType,
}

impl Shape {
    pub fn from(data_type: ScalarType, dims: &[usize]) -> Shape {
        Shape {
            data_type,
            dims: dims.to_vec(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn buffer_bytes_aligned(&self) -> usize {
        // Round buffer sizes to 16 bytes. If not, things go wrong (i.e. our shaders use vec4<f32> - if a buffer only
        // has 7 elements, the last vec4 cannot be fully written to the buffer, and the buffer ends up containing zeroes.
        fn round_to_next_multiple_of_16(n: usize) -> usize {
            (n + 15) / 16 * 16
        }

        round_to_next_multiple_of_16(self.element_count() * self.data_type.stride())
    }

    pub fn dim(&self, idx: usize) -> usize {
        self.dims[idx]
    }

    pub fn chunks(&self) -> Vec<usize> {
        let mut chunk = vec![];
        let ds = &self.dims;
        for i in 1..self.dims.len() {
            chunk.push(ds[i..].iter().product());
        }
        chunk.push(1);
        chunk
    }

    /// Computes the shape to which all provided shapes can be broadcast (if it exists)
    /// Inspired by <https://github.com/sonos/tract/blob/68db0209c9ffd1b91dff82884f4ae03b3622dd34/core/src/broadcast.rs#L5>
    pub fn multi_broadcast(shapes: &[Shape]) -> Option<Shape> {
        if shapes.is_empty() {
            return None;
        }

        let max_rank = shapes.iter().map(|x| x.rank()).max().unwrap_or(0);
        let mut shape: Vec<usize> = Vec::with_capacity(max_rank);

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
            shape.push(wanted_size);
        }

        shape.reverse();
        Some(Shape::from(data_type, &shape))
    }

    pub(crate) fn left_padded_to(&self, x: usize, rank: usize) -> Shape {
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

#[derive(Clone, Serialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum TensorData<'a> {
    F32(Cow<'a, [f32]>),
    I32(Cow<'a, [i32]>),
    I64(Cow<'a, [i64]>),
    U8(Cow<'a, [u8]>),
}

impl<'a> TensorData<'a> {
    pub fn into_static(self) -> TensorData<'static> {
        match self {
            TensorData::F32(x) => TensorData::F32(Cow::Owned(x.into_owned())),
            TensorData::I32(x) => TensorData::I32(Cow::Owned(x.into_owned())),
            TensorData::I64(x) => TensorData::I64(Cow::Owned(x.into_owned())),
            TensorData::U8(x) => TensorData::U8(Cow::Owned(x.into_owned())),
        }
    }

    pub fn scalar_type(&self) -> ScalarType {
        match self {
            TensorData::F32(_) => ScalarType::F32,
            TensorData::I32(_) => ScalarType::I32,
            TensorData::I64(_) => ScalarType::I64,
            TensorData::U8(_) => ScalarType::U8,
        }
    }
}

impl<'a> From<Vec<f32>> for TensorData<'a> {
    fn from(value: Vec<f32>) -> Self {
        TensorData::F32(Cow::Owned(value))
    }
}

impl<'a> From<&'a [f32]> for TensorData<'a> {
    fn from(a: &'a [f32]) -> Self {
        TensorData::F32(Cow::Borrowed(a))
    }
}

impl<'a> From<&'a [i32]> for TensorData<'a> {
    fn from(a: &'a [i32]) -> Self {
        TensorData::I32(Cow::Borrowed(a))
    }
}

impl<'a> From<&'a [i64]> for TensorData<'a> {
    fn from(a: &'a [i64]) -> Self {
        TensorData::I64(Cow::Borrowed(a))
    }
}

#[derive(Error, Debug)]
pub enum TensorConversionError {
    #[error("could not convert to the requested type becaue a value could not be represented in the target type")]
    OutOfBoundsError,

    #[error("cold not return the requested type; conversions cannot be done for slices")]
    DataTypeError,
}

impl<'a> TryFrom<TensorData<'a>> for Vec<f32> {
    type Error = TensorConversionError;

    /// Convert OutputTensor into a `Vec<f32>`, possibly converting integer tensors if the values fit
    fn try_from(value: TensorData) -> Result<Self, Self::Error> {
        match value {
            TensorData::F32(floats) => Ok(floats.to_vec()),
            TensorData::I32(ints) => ints
                .iter()
                .map(|i| f32::from_i32(*i).ok_or(TensorConversionError::OutOfBoundsError))
                .collect::<Result<_, _>>(),
            TensorData::I64(ints) => ints
                .iter()
                .map(|i| f32::from_i64(*i).ok_or(TensorConversionError::OutOfBoundsError))
                .collect::<Result<_, _>>(),
            TensorData::U8(ints) => ints
                .iter()
                .map(|i| f32::from_u8(*i).ok_or(TensorConversionError::OutOfBoundsError))
                .collect::<Result<_, _>>(),
        }
    }
}

/// Convert &OutputTensor into an &[f32]. Because we cannot store converted results, this operation does not attempt
/// to convert the tensor if the values are of a different type
impl<'a> TryFrom<&'a TensorData<'a>> for &'a [f32] {
    type Error = TensorConversionError;

    fn try_from(value: &'a TensorData) -> Result<Self, Self::Error> {
        match value {
            TensorData::F32(floats) => Ok(floats),
            TensorData::I32(_) | TensorData::I64(_) | TensorData::U8(_) => {
                Err(TensorConversionError::DataTypeError)
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum DataTypeError {
    #[error("the ONNX scalar data type {0:?} is not supported")]
    NotSupported(i32),

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
    U8,
}

impl ScalarType {
    pub fn stride(&self) -> usize {
        match self {
            ScalarType::F32 => 4,
            ScalarType::I32 => 4,
            ScalarType::I64 => 8,
            ScalarType::U8 => 1, // ! TODO check this
        }
    }

    pub fn wgsl_supported(&self) -> bool {
        match self {
            ScalarType::F32 => true,
            ScalarType::I32 => true,
            ScalarType::I64 => false,
            ScalarType::U8 => false, // ! TODO check this
        }
    }

    pub fn wgsl_type_name(&self) -> &'static str {
        match self {
            ScalarType::F32 => "f32",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::U8 => "u8", // ! TODO check this
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            ScalarType::F32 => true,
            ScalarType::I32 | ScalarType::I64 | ScalarType::U8 => false,
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
            "{}:{}",
            self.dims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("x"),
            self.data_type
        )
    }
}

/// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
pub(crate) fn ceil(num: usize, div: usize) -> usize {
    num / div + (num % div != 0) as usize
}

#[cfg(test)]
mod tests {
    use super::{ScalarType, Shape};

    // Test cases for Shape::multi_broadcast, some inspired by <https://github.com/sonos/tract/blob/68db0209c9ffd1b91dff82884f4ae03b3622dd34/core/src/broadcast.rs#L31>
    #[test]
    pub fn test_multi_broadcast() {
        fn shape(s: &[usize]) -> Shape {
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

        assert_eq!(
            Shape::multi_broadcast(&[shape(&[1, 255, 768]), shape(&[1, 255, 1])]),
            Some(shape(&[1, 255, 768])),
        );
    }
}
