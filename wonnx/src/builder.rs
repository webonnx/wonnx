use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use protobuf::ProtobufEnum;

use crate::{
    gpu::GpuModel,
    ir::{Input, Node, NodeDefinition, OperatorDefinition},
    onnx::TensorProto_DataType,
    optimizer::Optimizer,
    resource,
    utils::Shape,
    Session, SessionError,
};

pub struct GraphBuilder<'model> {
    _d: PhantomData<&'model ()>,
}

pub struct TensorRef<'model> {
    node: Arc<Node<'model>>,
    output_index: usize,
    output_shape: Shape,
}

impl<'model> From<&TensorRef<'model>> for Input<'model> {
    fn from(val: &TensorRef<'model>) -> Self {
        Input {
            source_node: val.node.clone(),
            output_index: val.output_index,
        }
    }
}

impl<'model> TensorRef<'model> {
    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.output_shape, rhs.output_shape);
        self.binary_op(rhs, "Add", self.output_shape.clone())
    }

    pub fn neg(&self) -> Self {
        self.unary_mapping_op("Neg")
    }

    fn unary_mapping_op(&self, op_type: &str) -> Self {
        let op_def = OperatorDefinition::new(
            op_type,
            vec![self.output_shape.clone()],
            format!("{}_{}", self.node.definition().get_display_name(), op_type),
        );
        TensorRef {
            node: Arc::new(Node::new(
                NodeDefinition::Operator(op_def),
                vec![Input {
                    source_node: self.node.clone(),
                    output_index: 0,
                }],
            )),
            output_index: 0,
            output_shape: self.output_shape.clone(),
        }
    }

    fn binary_op(&self, rhs: &Self, op_type: &str, output_shape: Shape) -> Self {
        let def = NodeDefinition::Operator(OperatorDefinition::new(
            op_type,
            vec![output_shape.clone()],
            format!(
                "{}_{}_{}",
                self.node.definition().get_display_name(),
                rhs.node.definition().get_display_name(),
                op_type
            ),
        ));

        TensorRef {
            node: Arc::new(Node::new(def, vec![self.into(), rhs.into()])),
            output_index: 0,
            output_shape,
        }
    }
}

impl<'model> GraphBuilder<'model> {
    pub fn new() -> GraphBuilder<'model> {
        GraphBuilder {
            _d: PhantomData::default(),
        }
    }

    pub fn tensor_f32(&mut self, name: &str, dims: &[usize], data: &[f32]) -> TensorRef<'model> {
        let mut initializer = crate::onnx::TensorProto::new();
        let udims: Vec<i64> = dims.iter().map(|x| *x as i64).collect();
        let shape = Shape::from(crate::utils::ScalarType::F32, &udims);
        assert_eq!(shape.element_count() as usize, data.len());
        initializer.set_dims(shape.dims.iter().map(|x| *x as i64).collect());
        initializer.set_name(name.to_string());
        initializer.set_data_type(TensorProto_DataType::FLOAT.value());
        initializer.set_float_data(data.to_vec());
        let definition = NodeDefinition::Tensor(Box::new(Cow::Owned(initializer)));
        TensorRef {
            node: Arc::new(Node {
                inputs: vec![],
                definition,
            }),
            output_index: 0,
            output_shape: shape,
        }
    }

    pub async fn session(
        &self,
        output_names: Vec<String>,
        outputs: &[TensorRef<'model>],
        onnx_opset_version: i64,
    ) -> Result<Session, SessionError> {
        let outputs = Arc::new(Node::new(
            NodeDefinition::Outputs {
                names: output_names,
            },
            outputs
                .iter()
                .map(|x| Input {
                    source_node: x.node.clone(),
                    output_index: x.output_index,
                })
                .collect(),
        ));

        let (device, queue) = resource::request_device_queue().await;
        let mut optimizer = Optimizer::new(onnx_opset_version);
        let ir = optimizer.optimize(outputs).await?;
        let gpu_model = GpuModel::from(ir, device, queue, onnx_opset_version)?;
        Ok(Session { gpu_model })
    }
}

impl<'model> Default for GraphBuilder<'model> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::TensorData;

    use super::GraphBuilder;
    use std::collections::HashMap;

    #[test]
    pub fn test_builder() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let mut m = GraphBuilder::new();
            let a = m.tensor_f32("x", &[1, 3], &[0.1, 0.2, 0.3]);
            let b = m.tensor_f32("y", &[1, 3], &[3.0, 2.0, 1.0]);
            let axb = a.add(&b);
            let sesh = m
                .session(vec!["result".to_string()], &[axb], 13)
                .await
                .unwrap();
            let result = sesh.run(&HashMap::new()).await.unwrap();
            assert_eq!(
                result["result"],
                TensorData::F32(vec![3.1, 2.2, 1.3].into())
            )
        });
    }
}
