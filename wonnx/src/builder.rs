use crate::{
    gpu::GpuModel,
    ir::{Input, Node, NodeDefinition, OperatorDefinition, Tensor},
    optimizer::Optimizer,
    resource, Session,
};
use std::sync::Arc;

pub use crate::tensor::{ScalarType, Shape, TensorData};
pub use crate::{SessionError, WonnxError};

#[derive(Clone)]
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

    pub fn conv(&self, weights: &Self, kernel_shape: &[usize], output_dims: &[usize]) -> Self {
        let output_shape = Shape::from(self.output_shape.data_type, output_dims);
        let mut op = OperatorDefinition::new(
            "Conv",
            vec![output_shape.clone()],
            format!("{}_conv", self.node.definition.get_display_name()),
        );
        op.set_attribute(
            "kernel_shape",
            kernel_shape.iter().map(|x| *x as i64).collect::<Vec<i64>>(),
        );
        TensorRef {
            node: Arc::new(Node::new(
                NodeDefinition::Operator(op),
                vec![self.into(), weights.into()],
            )),
            output_index: 0,
            output_shape,
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

pub fn input<'model, S: ToString>(
    name: S,
    scalar_type: ScalarType,
    dims: &[usize],
) -> TensorRef<'model> {
    let shape = Shape::from(scalar_type, dims);
    TensorRef {
        node: Arc::new(Node {
            inputs: vec![],
            definition: NodeDefinition::Input {
                name: name.to_string(),
                shape: shape.clone(),
            },
        }),
        output_index: 0,
        output_shape: shape,
    }
}

pub fn tensor<'model, S: ToString>(
    name: S,
    dims: &[usize],
    data: TensorData<'model>,
) -> TensorRef<'model> {
    let output_shape = Shape::from(data.scalar_type(), dims);
    TensorRef {
        node: Arc::new(Node {
            inputs: vec![],
            definition: NodeDefinition::Tensor(Tensor {
                data,
                dims: dims.to_vec(),
                display_name: name.to_string(),
            }),
        }),
        output_index: 0,
        output_shape,
    }
}

/// Start an inference session for calculating the outputs provided. The names will be used as keys in the resulting hashmap
pub async fn session_for_outputs<'model, S: ToString>(
    output_names: &[S],
    outputs: &[TensorRef<'model>],
    onnx_opset_version: i64,
) -> Result<Session, SessionError> {
    let output_names = output_names.iter().map(|x| x.to_string()).collect();

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

#[cfg(test)]
mod tests {
    use crate::{
        builder::{session_for_outputs, tensor},
        tensor::TensorData,
    };
    use std::collections::HashMap;

    #[test]
    pub fn test_builder() {
        let _ = env_logger::builder().is_test(true).try_init();
        pollster::block_on(async {
            let a = tensor("x", &[1, 3], vec![0.1, 0.2, 0.3].into());
            let b = tensor("y", &[1, 3], vec![3.0, 2.0, 1.0].into());
            let axb = a.add(&b);
            let sesh = session_for_outputs(&["result"], &[axb], 13).await.unwrap();
            let result = sesh.run(&HashMap::new()).await.unwrap();
            assert_eq!(
                result["result"],
                TensorData::F32(vec![3.1, 2.2, 1.3].into())
            )
        });
    }
}
