use std::collections::HashMap;

use wonnx::onnx::ModelProto;
use wonnx::SessionConfig;

use async_trait::async_trait;
use wonnx::utils::OutputTensor;

use crate::types::Inferer;
use crate::types::NNXError;

pub struct GPUInferer {
    session: wonnx::Session,
}

impl GPUInferer {
    pub async fn new(
        model_path: &str,
        outputs: Option<Vec<String>>,
    ) -> Result<GPUInferer, NNXError> {
        let session_config = SessionConfig::new().with_outputs(outputs);

        Ok(GPUInferer {
            session: wonnx::Session::from_path_with_config(model_path, &session_config).await?,
        })
    }
}

#[async_trait]
impl Inferer for GPUInferer {
    async fn infer(
        &self,
        outputs: &[String],
        inputs: &HashMap<String, crate::Tensor>,
        _model: &ModelProto,
    ) -> Result<HashMap<String, OutputTensor>, NNXError> {
        let input_refs = inputs
            .iter()
            .map(|(k, v)| (k.clone(), v.input_tensor()))
            .collect();
        let mut result = self.session.run(&input_refs).await.expect("run failed");

        let mut output_tensors = HashMap::<String, OutputTensor>::new();

        for output_name in outputs {
            let result = match result.remove(output_name) {
                Some(out) => out,
                None => return Err(NNXError::OutputNotFound(output_name.to_string())),
            };
            output_tensors.insert(output_name.clone(), result);
        }
        Ok(output_tensors)
    }
}
