pub mod compiler;
mod gpu;
pub mod ir;
pub mod onnx;
pub mod optimizer;
pub mod resource;
pub mod utils;

#[macro_use]
extern crate lazy_static;

use compiler::CompileError;
use gpu::GpuError;
use ir::IrError;
use optimizer::{Optimizer, OptimizerError};
use protobuf::{self, Message, ProtobufError};
use std::collections::HashMap;
use std::path::Path;
use std::result::Result;
use utils::{DataTypeError, InputTensor};

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

/// Creates a new session connected to the GPU.
///
/// Generate a session that will translate the onnx format into WGSL instructions.
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

    #[error("more than one ONNX opset was specified: {0} and {1}")]
    DuplicateOnnxOpset(i64, i64),

    #[error("the model references an unknown opset: '{0}'")]
    UnknownOpset(String),

    #[error("the model did not reference a specific version of the ONNX opset")]
    UnknownOnnxOpsetVersion,

    #[error("IR error: {0}")]
    IrError(#[from] IrError),

    #[error("GPU model error: {0}")]
    GpuError(#[from] GpuError),

    #[error("optimizer error: {0}")]
    OptimizerError(#[from] OptimizerError),
}

impl Session {
    // Read an ONNX model from a path and create a session.
    pub async fn from_path<P: AsRef<Path>>(path: P) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        Session::from_model(model).await
    }

    pub async fn from_bytes(bytes: &[u8]) -> Result<Session, SessionError> {
        let model = onnx::ModelProto::parse_from_bytes(bytes)?;
        Session::from_model(model).await
    }

    // Create a Session given an ONNX model.
    pub async fn from_model(model: onnx::ModelProto) -> Result<Session, SessionError> {
        let (device, queue) = resource::request_device_queue().await;

        // Find the version of the ONNX operator set this model is using (this is useful because some operators' specifications change over time).
        // Note, if any other op set than the ONNX operator set is referenced, we cannot run the model.
        // See https://github.com/onnx/onnx/blob/master/docs/Versioning.md#operator-sets
        let mut onnx_opset_version = None;
        for opset_import in model.get_opset_import() {
            match opset_import.get_domain() {
                "" => {
                    // This is a reference to the ONNX specification op set
                    if let Some(onnx_version) = onnx_opset_version {
                        if opset_import.get_version() != onnx_version {
                            return Err(SessionError::DuplicateOnnxOpset(
                                onnx_version,
                                opset_import.get_version(),
                            ));
                        }
                    } else {
                        onnx_opset_version = Some(opset_import.get_version());
                    }
                }
                some_other_opset => {
                    return Err(SessionError::UnknownOpset(some_other_opset.to_string()));
                }
            }
        }

        // Optimize and compile the model graph to a set of buffers and 'builders' which can basically run GPU shader code referencing these buffers
        let onnx_opset_version = onnx_opset_version.ok_or(SessionError::UnknownOnnxOpsetVersion)?;

        let mut optimizer = Optimizer::new();
        let ir = optimizer.optimize(ir::Node::from_model(&model)?)?;
        let gpu_model = GpuModel::from(ir, device, queue, onnx_opset_version)?;

        Ok(Session { gpu_model })
    }

    /// Perform inference given the inputs provided and return all the outputs the model was compiled to return.
    pub async fn run<'a>(
        &self,
        inputs: &HashMap<String, InputTensor<'a>>,
    ) -> Result<HashMap<String, Vec<f32>>, SessionError> {
        Ok(self.gpu_model.infer(inputs).await?)
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use crate::InputTensor;
    use js_sys::Promise;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::Arc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_console_logger::DEFAULT_LOGGER;
    use wasm_bindgen_futures::future_to_promise;

    #[wasm_bindgen(start)]
    pub fn main() {
        console_error_panic_hook::set_once();
        log::set_logger(&DEFAULT_LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Info);
    }

    #[wasm_bindgen]
    pub struct Input {
        input_data: HashMap<String, Vec<f32>>,
    }

    #[wasm_bindgen]
    impl Input {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Input {
            Input {
                input_data: HashMap::new(),
            }
        }

        #[wasm_bindgen(js_name = "insert")]
        pub fn insert(&mut self, input_name: String, value: Vec<f32>) {
            self.input_data.insert(input_name, value);
        }
    }

    #[wasm_bindgen]
    #[derive(Clone)]
    pub struct Session {
        session: Arc<crate::Session>,
    }

    #[wasm_bindgen]
    pub struct SessionError(crate::SessionError);

    #[wasm_bindgen]
    impl SessionError {
        #[wasm_bindgen(js_name = toString)]
        pub fn to_string(&self) -> String {
            self.0.to_string()
        }
    }

    #[wasm_bindgen]
    impl Session {
        #[wasm_bindgen(js_name = "fromBytes")]
        pub async fn from_bytes(bytes: Vec<u8>) -> Result<Session, SessionError> {
            Ok(Session {
                session: Arc::new(
                    crate::Session::from_bytes(bytes.as_slice())
                        .await
                        .map_err(SessionError)?,
                ),
            })
        }

        pub fn run(&self, input: Input) -> Promise {
            let input_data: HashMap<String, InputTensor<'_>> = input
                .input_data
                .into_iter()
                .map(|(k, v)| (k, InputTensor::F32(Cow::Owned(v))))
                .collect();

            let engine = self.session.clone();

            future_to_promise(async move {
                let result = engine.run(&input_data).await.map_err(SessionError)?;
                Ok(JsValue::from_serde(&result).unwrap())
            })
        }
    }
}
