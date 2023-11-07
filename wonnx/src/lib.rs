mod compiler;
mod gpu;
mod ir;
pub mod onnx;
mod optimizer;
mod resource;
pub mod utils;

pub use compiler::CompileError;
pub use gpu::GpuError;
use ir::IrError;
pub use optimizer::constant_of_shape_output;
use optimizer::{Optimizer, OptimizerError};
use protobuf::{self, Message, ProtobufError};
use std::collections::HashMap;
use std::path::Path;
use std::result::Result;
use utils::{get_opset_version, DataTypeError, InputTensor, OpsetError, OutputTensor};

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
///
/// # Examples
///
/// Basic usage:
///
/// ```ignore
/// let mut session = Session::from_path("path/to/model.onnx").await.unwrap();
/// ```
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
    // Read an ONNX model from a path and create a session, using default [session config](SessionConfig).
    pub async fn from_path<P: AsRef<Path>>(path: P) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        Session::from_model(model).await
    }

    // Read an ONNX model from a path and create a session using the specified [session config](SessionConfig).
    pub async fn from_path_with_config<P: AsRef<Path>>(
        path: P,
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        Session::from_model_with_config(model, config).await
    }

    /// Read an ONNX model from bytes and create a session, using default [session config](SessionConfig).
    pub async fn from_bytes(bytes: &[u8]) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;
        Session::from_model(model).await
    }

    /// Read an ONNX model from bytes and create a session with the specified [session config](SessionConfig).
    pub async fn from_bytes_with_config(
        bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;
        Session::from_model_with_config(model, config).await
    }

    /// Create a session using the provided [`onnx::ModelProto`] and [session config](SessionConfig).
    pub async fn from_model_with_config(
        model: onnx::ModelProto,
        config: &SessionConfig,
    ) -> Result<Session, SessionError> {
        let (device, queue) = resource::request_device_queue().await;

        // Optimize and compile the model graph to a set of buffers and 'builders' which can basically run GPU shader code referencing these buffers
        let onnx_opset_version = get_opset_version(&model)
            .map_err(SessionError::OpsetError)?
            .ok_or(SessionError::UnknownOnnxOpsetVersion)?;

        let mut optimizer = Optimizer::new(onnx_opset_version);
        let ir = optimizer
            .optimize(ir::Node::from_model(&model, config.outputs.as_deref())?)
            .await?;
        let gpu_model = GpuModel::from(ir, device, queue, onnx_opset_version)?;

        Ok(Session { gpu_model })
    }

    /// Create a Session given an ONNX model, using default configuration.
    pub async fn from_model(model: onnx::ModelProto) -> Result<Session, SessionError> {
        Self::from_model_with_config(model, &SessionConfig::new()).await
    }

    /// Perform inference given the inputs provided and return all the outputs the model was compiled to return.
    pub async fn run<'a>(
        &self,
        inputs: &HashMap<String, InputTensor<'a>>,
    ) -> Result<HashMap<String, OutputTensor>, SessionError> {
        Ok(self.gpu_model.infer(inputs).await?)
    }
}
