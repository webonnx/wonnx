pub mod builder;
pub mod ir;
pub mod onnx;
pub mod onnx_model;
pub mod tensor;

mod compiler;
mod gpu;
mod optimizer;
mod resource;

#[macro_use]
extern crate lazy_static;

pub use compiler::CompileError;
pub use gpu::GpuError;
use ir::IrError;
use onnx_model::OpsetError;
pub use optimizer::constant_of_shape_output;
use optimizer::OptimizerError;
use protobuf::{self, ProtobufError};
use std::collections::HashMap;
use std::result::Result;
use tensor::{DataTypeError, TensorData};

use crate::gpu::GpuModel;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WonnxError {
    #[error("error compiling model: {0}")]
    CompileError(#[from] CompileError),

    #[error("error executing the model: {0}")]
    SessionError(#[from] SessionError),

    #[error("error in intermediate representation: {0}")]
    IrError(#[from] IrError),

    #[error("error in data types: {0}")]
    TypeError(#[from] DataTypeError),
}

/// An inference [session](Session) represents a model that is loaded and ready to perform inference on the GPU.
pub struct Session {
    gpu_model: GpuModel,
}

#[derive(Error, Debug)]
pub enum SessionError {
    #[error("could not deserialize model: {0}")]
    ModelDeserializationError(#[from] ProtobufError),

    #[error("an error occurred reading the model file: {0}")]
    ModelReadingError(#[from] std::io::Error),

    #[error(
        "invalid input name '{0}'; inspect the file with e.g. Netron to find the correct name"
    )]
    InvalidInput(String),

    #[error(
        "invalid output name '{0}'; inspect the file with e.g. Netron to find the correct name"
    )]
    InvalidOutput(String),

    #[error("the model did not reference a specific version of the ONNX opset")]
    UnknownOnnxOpsetVersion,

    #[error("IR error: {0}")]
    IrError(#[from] IrError),

    #[error("GPU model error: {0}")]
    GpuError(#[from] GpuError),

    #[error("optimizer error: {0}")]
    OptimizerError(#[from] OptimizerError),

    #[error("opset error: {0}")]
    OpsetError(#[from] OpsetError),
}

/// Provides optional configuration when creating an inference [Session].
#[non_exhaustive]
pub struct SessionConfig {
    /// When set, only the specified outputs will be calculated, and nodes that are not inputs to these outputs may not be processed
    pub outputs: Option<Vec<String>>,
}

impl SessionConfig {
    /// Creates a new [SessionConfig] struct with the default options set.
    pub fn new() -> Self {
        Self { outputs: None }
    }

    /// Sets [`SessionConfig::outputs`] to the specified value and returns [Self].
    pub fn with_outputs(mut self, outputs: Option<Vec<String>>) -> Self {
        self.outputs = outputs;
        self
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// Perform inference given the inputs provided and return all the outputs the model was compiled to return.
    pub async fn run<'a>(
        &self,
        inputs: &HashMap<String, TensorData<'a>>,
    ) -> Result<HashMap<String, TensorData<'static>>, SessionError> {
        Ok(self.gpu_model.infer(inputs).await?)
    }
}
