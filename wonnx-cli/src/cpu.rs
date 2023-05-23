use std::collections::HashMap;

use crate::{Inferer, NNXError};
use async_trait::async_trait;
use tract_onnx::prelude::*;
use wonnx::{
    onnx::ModelProto,
    utils::{Shape, TensorData},
};

type RunnableOnnxModel =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct CPUInferer {
    model: RunnableOnnxModel,
    input_shapes: HashMap<String, Shape>,
}

impl CPUInferer {
    pub async fn new(
        model_path: &str,
        input_shapes: &HashMap<String, Shape>,
    ) -> Result<CPUInferer, NNXError> {
        let mut cpu_model = tract_onnx::onnx().model_for_path(model_path)?;

        for (input_name, input_shape) in input_shapes {
            let input_node = cpu_model.node_by_name(input_name)?.id;
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
        Ok(CPUInferer {
            model: cpu_model,
            input_shapes: input_shapes.clone(),
        })
    }
}

trait ToTract {
    fn to_tract_tensor(&self, dims: &[usize]) -> Result<Tensor, NNXError>;
}

impl ToTract for wonnx_preprocessing::Tensor {
    fn to_tract_tensor(&self, dims: &[usize]) -> Result<Tensor, NNXError> {
        match self {
            wonnx_preprocessing::Tensor::F32(d) => Ok(tract_onnx::prelude::Tensor::from_shape(
                dims,
                d.as_slice().unwrap(),
            )?),
            wonnx_preprocessing::Tensor::I32(d) => Ok(tract_onnx::prelude::Tensor::from_shape(
                dims,
                d.as_slice().unwrap(),
            )?),
            wonnx_preprocessing::Tensor::I64(d) => Ok(tract_onnx::prelude::Tensor::from_shape(
                dims,
                d.as_slice().unwrap(),
            )?),
        }
    }
}

#[async_trait]
impl Inferer for CPUInferer {
    async fn infer(
        &self,
        outputs: &[String],
        inputs: &HashMap<String, crate::Tensor>,
        model: &ModelProto,
    ) -> Result<HashMap<String, TensorData<'static>>, NNXError> {
        let mut cpu_inputs: HashMap<usize, tract_onnx::prelude::Tensor> = HashMap::new();

        for (input_name, input_tensor) in inputs {
            let input_index = model
                .get_graph()
                .get_input()
                .iter()
                .enumerate()
                .find(|x| x.1.get_name() == input_name)
                .unwrap_or_else(|| panic!("input not found with name {}", input_name));
            log::info!("set input fact {} for cpu model", input_index.0,);

            let dims: Vec<usize> = self.input_shapes[input_name].dims.to_vec();
            cpu_inputs.insert(input_index.0, input_tensor.to_tract_tensor(&dims)?);
        }

        let mut cpu_inputs_ordered = TVec::<TValue>::new();
        for i in 0..inputs.len() {
            cpu_inputs_ordered.push(TValue::Const(Arc::new(cpu_inputs.get(&i).unwrap().clone())));
        }

        let result = self.model.run(cpu_inputs_ordered)?;
        log::debug!("cpu result: {:?}", result);

        let mut output_tensors = HashMap::<String, TensorData>::new();

        for output_name in outputs {
            let result_vector = {
                // Find position of the node with the specified name in the output set.
                if let Some(idx) = self
                    .model
                    .outputs
                    .iter()
                    .enumerate()
                    .find(|x| &self.model.model.outlet_labels[x.1] == output_name)
                {
                    log::debug!(
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
            };

            let av = result_vector.to_array_view()?;
            output_tensors.insert(
                output_name.clone(),
                TensorData::F32(av.as_slice().unwrap().into()).into_static(),
            );
        }
        Ok(output_tensors)
    }
}
