use std::collections::HashMap;

use wonnx::onnx::ModelProto;
use wonnx::SessionOptions;

use async_trait::async_trait;
use wonnx::utils::OutputTensor;

use crate::types::InferOptions;
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
        let session_opts = SessionOptions { outputs };

        Ok(GPUInferer {
            session: wonnx::Session::from_path_with_options(model_path, &session_opts).await?,
        })
    }
}

#[async_trait]
impl Inferer for GPUInferer {
    async fn infer(
        &self,
        infer_opt: &InferOptions,
        inputs: &HashMap<String, crate::Tensor>,
        _model: &ModelProto,
    ) -> Result<OutputTensor, NNXError> {
        let input_refs = inputs
            .iter()
            .map(|(k, v)| (k.clone(), v.input_tensor()))
            .collect();
        let mut result = self.session.run(&input_refs).await.expect("run failed");

        let result = match &infer_opt.output_name {
            Some(output_name) => match result.remove(output_name) {
                Some(out) => out,
                None => return Err(NNXError::OutputNotFound(output_name.to_string())),
            },
            None => result.values().next().unwrap().clone(),
        };

        log::info!("gpu result: {:?}", result);
        Ok(result)
    }
}
