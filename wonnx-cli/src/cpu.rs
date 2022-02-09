use std::collections::HashMap;

use crate::{InferOptions, Inferer, NNXError};
use async_trait::async_trait;
use tract_onnx::prelude::*;
use wonnx::{onnx::ModelProto, utils::Shape};

type RunnableOnnxModel =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct CPUInferer {
    model: RunnableOnnxModel,
}

impl CPUInferer {
    pub async fn new(
        model_path: &str,
        input_shapes: &HashMap<String, Shape>,
    ) -> Result<CPUInferer, NNXError> {
        let mut cpu_model = tract_onnx::onnx().model_for_path(&model_path)?;

        for (input_name, input_shape) in input_shapes {
            let input_node = cpu_model.node_by_name(&input_name)?.id;
            let fact = InferenceFact::dt_shape(f32::datum_type(), &input_shape.dims);
            log::info!(
                "set input '{}' (id {}) to shape {:?}",
                input_name,
                input_node,
                input_shape
            );
            cpu_model.set_input_fact(input_node, fact)?;
        }

        let cpu_model = cpu_model.into_optimized()?.into_runnable()?;
        Ok(CPUInferer { model: cpu_model })
    }
}

#[async_trait]
impl Inferer for CPUInferer {
    async fn infer(
        &self,
        infer_opt: &InferOptions,
        inputs: &HashMap<String, crate::Tensor>,
        model: &ModelProto,
    ) -> Result<Vec<f32>, NNXError> {
        let mut cpu_inputs: HashMap<usize, tract_onnx::prelude::Tensor> = HashMap::new();

        for (input_name, input_tensor) in inputs {
            let input_index = model
                .get_graph()
                .get_input()
                .iter()
                .enumerate()
                .find(|x| x.1.get_name() == input_name)
                .unwrap_or_else(|| panic!("input not found with name {}", input_name));
            log::info!(
                "set input fact {} for cpu model (shape: {:?})",
                input_index.0,
                input_tensor.shape
            );

            let dims: Vec<usize> = input_tensor
                .shape
                .dims
                .iter()
                .map(|x| (*x) as usize)
                .collect();

            cpu_inputs.insert(
                input_index.0,
                tract_onnx::prelude::Tensor::from_shape(
                    &dims,
                    input_tensor.data.as_slice().unwrap(),
                )?,
            );
        }

        let mut cpu_inputs_ordered = TVec::new();
        for i in 0..inputs.len() {
            cpu_inputs_ordered.push(cpu_inputs.get(&i).unwrap().clone());
        }

        let result = self.model.run(cpu_inputs_ordered)?;
        log::info!("cpu result: {:?}", result);

        let result_vector = match &infer_opt.output_name {
            Some(output_name) => {
                /* Find position of the node with the specified name in the output set. Note that Tract will suffix the
                names of the nodes in the ONNX graph with the name of the output, i.e. "Plus214_Output_0.ab.matmatmul"
                where the original name is called "Plus214_Output_0.ab.", hence the 'starts_with' hack below. */
                if let Some(idx) = self.model.outputs.iter().enumerate().find(|x| {
                    self.model.model.nodes[x.1.node]
                        .name
                        .starts_with(&format!("{}.", output_name))
                }) {
                    log::info!(
						"output node with name '{}' has idx {:?} (and tract id {}, slot {}, name '{}')",
						output_name,
						idx.0,
						idx.1.node,
						idx.1.slot,
						self.model.model.nodes[idx.1.node].name
					);
                    result[idx.0].clone()
                } else {
                    return Err(NNXError::OutputNotFound(output_name.to_string()));
                }
            }
            None => result[0].clone(),
        };

        let av = result_vector.to_array_view()?;
        Ok(av.as_slice().unwrap().to_vec())
    }
}
