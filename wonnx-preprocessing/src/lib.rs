use ndarray::ArrayBase;
use wonnx::utils::InputTensor;

pub mod image;
pub mod text;

pub enum Tensor {
    F32(ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>),
}

impl Tensor {
    pub fn input_tensor(&self) -> InputTensor {
        match self {
            Tensor::F32(a) => a.as_slice().unwrap().into(),
        }
    }
}
