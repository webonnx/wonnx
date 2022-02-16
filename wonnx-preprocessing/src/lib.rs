use ndarray::ArrayBase;
use wonnx::utils::Shape;

pub mod image;
pub mod text;

pub struct Tensor {
    pub data: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
    pub shape: Shape,
}
