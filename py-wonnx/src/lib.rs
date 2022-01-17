use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use wonnx::Session;
#[pyclass]
#[repr(transparent)]
pub struct PySession {
    pub session: Session,
}

#[pymethods]
impl PySession {
    #[staticmethod]
    pub fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let session = pollster::block_on(wonnx::Session::from_bytes(bytes)).unwrap();
        Ok(PySession { session: session })
    }

    pub fn run(&self, dict: &PyDict) -> PyResult<HashMap<String, Vec<f32>>> {
        let map: HashMap<String, Vec<f32>> = dict.extract().unwrap();
        let mut inputs = HashMap::new();
        for (key, value) in map.iter() {
            inputs.insert(key.clone(), value.as_slice());
        }
        let result = pollster::block_on(self.session.run(inputs)).unwrap();

        Ok(result)
    }
}

/// This module is implemented in Rust.
#[pymodule]
fn wonnx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySession>().unwrap();
    Ok(())
}
