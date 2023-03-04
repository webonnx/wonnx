#![doc = include_str!("../README.md")]
#![allow(clippy::unused_unit)]
#![allow(clippy::inherent_to_string)]

use js_sys::Promise;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use wonnx::utils::{InputTensor, OutputTensor};

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    log::set_max_level(log::LevelFilter::Info);
    console_log::init_with_level(log::Level::Info).unwrap();
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Input {
    input_data: HashMap<String, Arc<Vec<f32>>>,
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
        self.input_data.insert(input_name, Arc::new(value));
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Session {
    session: Arc<wonnx::Session>,
}

#[wasm_bindgen]
pub struct SessionError(wonnx::SessionError);

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
                wonnx::Session::from_bytes(bytes.as_slice())
                    .await
                    .map_err(SessionError)?,
            ),
        })
    }

    pub fn run(&self, input: &Input) -> Promise {
        let input_copy = input.clone();
        let engine = self.session.clone();

        future_to_promise(async move {
            let input_data: HashMap<String, InputTensor<'_>> = input_copy
                .input_data
                .iter()
                .map(|(k, v)| (k.clone(), v.as_slice().into()))
                .collect();
            let result = engine.run(&input_data).await.map_err(SessionError)?;
            drop(input_copy);
            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        })
    }
}

/// Convert an OutputTensor to a JsValue (we cannot implement Into<JsValue> for OutputTensor here)
pub fn tensor_to_js_value(tensor: OutputTensor) -> JsValue {
    match tensor {
        OutputTensor::F32(fs) => serde_wasm_bindgen::to_value(&fs).unwrap(),
        OutputTensor::I32(ints) => serde_wasm_bindgen::to_value(&ints).unwrap(),
        OutputTensor::I64(ints) => serde_wasm_bindgen::to_value(&ints).unwrap(),
        OutputTensor::U8(ints) => serde_wasm_bindgen::to_value(&ints).unwrap(),
    }
}
