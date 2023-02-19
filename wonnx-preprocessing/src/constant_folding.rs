use std::collections::HashMap;

use protobuf::{ProtobufEnum, RepeatedField};
use thiserror::Error;

use wonnx::{
    onnx::{
        GraphProto, NodeProto, TensorProto, TensorShapeProto, TensorShapeProto_Dimension,
        TypeProto, TypeProto_Tensor, ValueInfoProto,
    },
    utils::{model_with_opset, DataTypeError, InputTensor, NodeAttributes, OutputTensor, Shape},
    CompileError, GpuError, Session, SessionError,
};

use crate::shape_inference::{dimensions_infos, infer_forward, ShapeInferenceError};

#[derive(Error, Debug)]
pub enum ConstantFoldingError {
    #[error("unsupported data type encountered: {0}")]
    #[from(DataTypeError)]
    UnsupportedDataType(DataTypeError),

    #[error("could not infer shape for folded node: {0}")]
    ShapeInferenceError(ShapeInferenceError),

    #[error("invalid node: {0}")]
    InvalidNode(String),

    #[error("error calculating constant value: {0}")]
    #[from(SessionError)]
    CalculationError(SessionError),
}

pub async fn fold_constants(
    graph: &mut GraphProto,
    opset_version: i64,
) -> Result<(), ConstantFoldingError> {
    let mut new_initializers: HashMap<String, TensorProto> = HashMap::new();
    let mut folded_node_indexes: Vec<usize> = vec![];
    let mut foldable_nodes: Vec<String> = vec![];

    {
        let mut shapes =
            dimensions_infos(graph).map_err(ConstantFoldingError::UnsupportedDataType)?;
        let initializers: HashMap<String, &TensorProto> = HashMap::from_iter(
            graph
                .initializer
                .iter()
                .map(|x| (x.get_name().to_string(), x)),
        );

        for (node_index, node) in graph.node.iter().enumerate() {
            // If nodes have all-constant inputs, their output can be folded to a constant as long as the node is deterministic
            let all_inputs_are_constant = node.input.iter().all(|input_name| {
                initializers.contains_key(input_name) || new_initializers.contains_key(input_name)
            });
            let is_known_shape_node =
                node.get_op_type() == "Shape" && shapes.contains_key(&node.input[0]);
            if all_inputs_are_constant || is_known_shape_node {
                // This node can be folded
                log::debug!("Node '{}' can be folded (all inputs constant: {all_inputs_are_constant}, is known shape node: {is_known_shape_node})", node.get_name());
                let input_shapes: Vec<&Shape> = node
                    .input
                    .iter()
                    .map(|input_name| &shapes[input_name])
                    .collect();

                let inputs: Vec<InputTensor> = node
                    .input
                    .iter()
                    .map(|input_name| {
                        if let Some(initializer) = initializers.get(input_name) {
                            (*initializer).try_into()
                        } else {
                            (&new_initializers[input_name]).try_into()
                        }
                    })
                    .collect::<Result<_, _>>()
                    .map_err(ConstantFoldingError::UnsupportedDataType)?;
                if let Some(mut constant_output) = calculate_constant_node_outputs(
                    node,
                    &shapes,
                    &inputs,
                    &initializers,
                    opset_version,
                )
                .await?
                {
                    // Infer output shapes
                    let mut all_initializers: HashMap<String, &TensorProto> = HashMap::new();
                    all_initializers.extend(initializers.iter().map(|(k, v)| (k.clone(), *v)));
                    all_initializers.extend(new_initializers.iter().map(|(k, v)| (k.clone(), v)));
                    let output_shapes = infer_forward(node, &input_shapes, &initializers)
                        .map_err(ConstantFoldingError::ShapeInferenceError)?;

                    for (output_index, output_name) in node.output.iter().enumerate().rev() {
                        let output_tensor = constant_output.remove(output_index);

                        let mut initializer: TensorProto = output_tensor.into();
                        initializer.set_name(output_name.clone());
                        new_initializers.insert(output_name.clone(), initializer);
                        shapes.insert(output_name.clone(), output_shapes[output_index].clone());
                        folded_node_indexes.push(node_index);
                    }
                } else {
                    foldable_nodes.push(node.get_name().to_string());
                }
            }
        }
    }

    // Insert initializers that we created
    graph.initializer.extend(new_initializers.into_values());

    // Remove folded nodes
    folded_node_indexes.sort();
    for index in folded_node_indexes.iter().rev() {
        graph.node.remove(*index);
    }

    if !foldable_nodes.is_empty() {
        log::info!(
            "The following nodes can likely be folded, but currently aren't due to missing support: {}",
            foldable_nodes.join(", ")
        );
    }

    Ok(())
}

async fn calculate_constant_node_outputs<'a>(
    node: &'a NodeProto,
    shapes: &HashMap<String, Shape>,
    inputs: &[InputTensor<'a>],
    initializers: &HashMap<String, &TensorProto>,
    opset_version: i64,
) -> Result<Option<Vec<OutputTensor>>, ConstantFoldingError> {
    Ok(match node.get_op_type() {
        "Identity" => Some(inputs.iter().map(OutputTensor::from).collect()),
        "Shape" => {
            let input_shape: Vec<i64> = shapes[&node.input[0]]
                .dims
                .iter()
                .map(|x| *x as i64)
                .collect();
            let mut start = node.get_attribute_value("start", Some(0)).unwrap();
            let mut end = node
                .get_attribute_value("start", Some(input_shape.len() as i64))
                .unwrap();
            if start < 0 {
                start += input_shape.len() as i64;
            }
            if end < 0 {
                end += input_shape.len() as i64;
            }
            if start > end {
                return Err(ConstantFoldingError::InvalidNode(format!("end attribute value ({}) for Shape node should be higher than start attribute ({})", end, start)));
            }

            Some(vec![OutputTensor::I64(
                (input_shape[(start as usize)..=(end as usize)]).into(),
            )])
        }
        _ => {
            // Try to run on GPU
            let input_shapes: Vec<&Shape> = node
                .input
                .iter()
                .map(|input_name| &shapes[input_name])
                .collect();

            let output_shapes = infer_forward(node, &input_shapes, initializers)
                .map_err(ConstantFoldingError::ShapeInferenceError)?;

            let mut graph = GraphProto::new();
            graph.set_input(RepeatedField::from(
                node.input
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        let shape = &shapes[input];
                        input_to_value_info(shape, &format!("input_{}", index))
                    })
                    .collect::<Vec<_>>(),
            ));

            graph.set_output(RepeatedField::from(
                node.output
                    .iter()
                    .enumerate()
                    .map(|(index, _output)| {
                        let shape = &output_shapes[index];
                        input_to_value_info(shape, &format!("output_{}", index))
                    })
                    .collect::<Vec<_>>(),
            ));

            let mut temp_node = node.clone();
            temp_node.set_output(RepeatedField::from(
                graph
                    .output
                    .iter()
                    .map(|otp| otp.get_name().to_string())
                    .collect::<Vec<String>>(),
            ));
            temp_node.set_input(RepeatedField::from(
                graph
                    .input
                    .iter()
                    .map(|otp| otp.get_name().to_string())
                    .collect::<Vec<String>>(),
            ));
            graph.set_node(RepeatedField::from(vec![temp_node]));

            let model = model_with_opset(graph, opset_version);

            let session = match Session::from_model(model).await {
                Ok(v) => v,
                Err(e) => {
                    if let SessionError::GpuError(GpuError::CompileError {
                        error: CompileError::UnimplementedOp(op_name),
                        ..
                    }) = e
                    {
                        log::info!("could not constant-fold node '{}', because op '{}' is not yet implemented", node.get_name(), op_name);
                        return Ok(None);
                    } else {
                        return Err(ConstantFoldingError::CalculationError(e));
                    }
                }
            };

            let mut named_inputs: HashMap<String, InputTensor> = HashMap::new();
            for (index, input) in inputs.iter().enumerate() {
                let input: InputTensor = input.to_owned();
                named_inputs.insert(format!("input_{}", index), input);
            }

            let mut output_values = session
                .run(&named_inputs)
                .await
                .map_err(ConstantFoldingError::CalculationError)?;

            let outputs: Vec<OutputTensor> = (0..node.output.len())
                .map(|output_index| {
                    let output_key = format!("output_{}", output_index);
                    output_values.remove(&output_key).unwrap()
                })
                .collect();

            Some(outputs)
        }
    })
}

fn input_to_value_info(shape: &Shape, name: &str) -> ValueInfoProto {
    let mut ttp = TypeProto_Tensor::new();
    ttp.set_elem_type(shape.data_type.to_datatype().value());
    let mut tsp = TensorShapeProto::new();
    tsp.set_dim(RepeatedField::from(
        shape
            .dims
            .iter()
            .map(|x| {
                let mut tdp = TensorShapeProto_Dimension::new();
                tdp.set_dim_value(*x as i64);
                tdp
            })
            .collect::<Vec<TensorShapeProto_Dimension>>(),
    ));
    ttp.set_shape(tsp);
    let mut ftp = TypeProto::new();
    ftp.set_tensor_type(ttp);
    let mut vip = ValueInfoProto::new();
    vip.set_name(name.to_string());
    vip.set_field_type(ftp);
    vip
}
