use ndarray::ArrayBase;
use wonnx::utils::InputTensor;

pub mod image;
pub mod text;

pub enum Tensor {
    F32(ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>),
}

impl Tensor {
    // pub fn shape(&self) -> Shape {
    //     match self {
    //         Tensor::F32(a) => {
    //             let shape = a.shape();
    //             let shape_int: Vec<i64> = shape.iter().map(|x| *x as i64).collect();
    //             Shape::from(ScalarType::F32, &shape_int[..])
    //         }
    //     }
    // }

    pub fn input_tensor(&self) -> InputTensor {
        match self {
            Tensor::F32(a) => InputTensor::F32(a.as_slice().unwrap()),
        }
    }
}
