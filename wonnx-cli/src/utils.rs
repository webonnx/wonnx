use ndarray::ArrayBase;
use std::path::Path;
use wonnx::utils::{DataTypeError, ScalarType, Shape};
use wonnx::WonnxError;
use wonnx_preprocessing::image::{load_bw_image, load_rgb_image};

use wonnx::onnx::{ModelProto, TensorShapeProto, ValueInfoProto};

use crate::types::NNXError;
pub trait ValueInfoProtoUtil {
    fn dimensions(&self) -> Vec<usize>;
    fn data_type(&self) -> Result<ScalarType, DataTypeError>;
}

pub trait TensorShapeProtoUtil {
    fn shape_dimensions(&self) -> Vec<usize>;
}

impl ValueInfoProtoUtil for ValueInfoProto {
    fn dimensions(&self) -> Vec<usize> {
        match &self.get_field_type().value {
            Some(x) => match x {
                wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => {
                    t.get_shape().shape_dimensions()
                }
                wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => vec![],
        }
    }

    fn data_type(&self) -> Result<ScalarType, DataTypeError> {
        Ok(match &self.get_field_type().value {
            Some(x) => match x {
                wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => {
                    ScalarType::from_i32(t.get_elem_type())?
                }
                wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => return Err(DataTypeError::Undefined),
        })
    }
}

impl TensorShapeProtoUtil for TensorShapeProto {
    fn shape_dimensions(&self) -> Vec<usize> {
        self.get_dim()
            .iter()
            .map(|d| match d.value {
                Some(wonnx::onnx::TensorShapeProto_Dimension_oneof_value::dim_value(i)) => {
                    i as usize
                }
                _ => 0,
            })
            .collect()
    }
}

pub trait ModelProtoUtil {
    fn get_input_shape(&self, input_name: &str) -> Result<Option<Shape>, WonnxError>;
}

impl ModelProtoUtil for ModelProto {
    fn get_input_shape(&self, input_name: &str) -> Result<Option<Shape>, WonnxError> {
        let value_info = self
            .get_graph()
            .get_input()
            .iter()
            .find(|x| x.get_name() == input_name);
        match value_info {
            Some(vi) => Ok(Some(vi.get_shape()?)),
            None => Ok(None),
        }
    }
}

pub fn load_image_input(
    input_image: &Path,
    input_shape: &Shape,
) -> Result<ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>, NNXError> {
    if input_shape.rank() == 3 {
        let mut w = input_shape.dim(1) as usize;
        let mut h = input_shape.dim(2) as usize;
        if w == 0 {
            w = 224;
        }
        if h == 0 {
            h = 224;
        }

        if input_shape.dim(0) == 3 {
            log::info!("input is (3,?,?), loading as RGB image");
            Ok(load_rgb_image(input_image, w, h).into_dyn())
        } else if input_shape.dim(0) == 1 {
            log::info!("input is (1,?,?), loading as BW image");
            Ok(load_bw_image(input_image, w, h).into_dyn())
        } else {
            Err(NNXError::InvalidInputShape)
        }
    } else if input_shape.rank() == 4 {
        let mut w = input_shape.dim(2) as usize;
        let mut h = input_shape.dim(3) as usize;
        if w == 0 {
            w = 224;
        }
        if h == 0 {
            h = 224;
        }

        if input_shape.dim(1) == 3 {
            log::info!("input is (?,3,?,?), loading as RGB image");
            Ok(load_rgb_image(input_image, w, h).into_dyn())
        } else if input_shape.dim(1) == 1 {
            log::info!("input is (?,1,?,?), loading as BW image");
            Ok(load_bw_image(input_image, w, h).into_dyn())
        } else {
            Err(NNXError::InvalidInputShape)
        }
    } else {
        Err(NNXError::InvalidInputShape)
    }
}
