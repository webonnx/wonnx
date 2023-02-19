use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use wonnx::utils::OutputTensor;

use wonnx::Session;
#[pyclass]
#[repr(transparent)]
pub struct PySession {
    pub session: Session,
}

pub struct PyOutputTensor(OutputTensor);

impl IntoPy<PyObject> for PyOutputTensor {
    fn into_py(self, py: Python) -> PyObject {
        match self.0 {
            OutputTensor::F32(fs) => fs.into_py(py),
            OutputTensor::I32(fs) => fs.into_py(py),
            OutputTensor::I64(fs) => fs.into_py(py),
            OutputTensor::U8(fs) => fs.into_py(py),
        }
    }
}

#[pymethods]
impl PySession {
    #[staticmethod]
    pub fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let session = pollster::block_on(wonnx::Session::from_bytes(bytes)).unwrap();
        Ok(PySession { session })
    }

    #[staticmethod]
    pub fn from_path(path: &str) -> PyResult<Self> {
        let session = pollster::block_on(wonnx::Session::from_path(path)).unwrap();
        Ok(PySession { session })
    }

    pub fn run(&self, dict: &PyDict) -> PyResult<HashMap<String, PyOutputTensor>> {
        let map: HashMap<String, Vec<f32>> = dict.extract().unwrap();
        let mut inputs = HashMap::new();
        for (key, value) in map.iter() {
            inputs.insert(key.clone(), value.as_slice().into());
        }
        let result = pollster::block_on(self.session.run(&inputs)).unwrap();
        Ok(result
            .into_iter()
            .map(|(k, v)| (k, PyOutputTensor(v)))
            .collect())
    }
}

/// This module is implemented in Rust.
#[pymodule]
fn wonnx(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_class::<PySession>().unwrap();
    Ok(())
}
