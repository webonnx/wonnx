use std::collections::HashMap;

use protobuf::ProtobufEnum;
use thiserror::Error;
use wonnx::{
    onnx::{
        GraphProto, NodeProto, TensorProto, TensorShapeProto, TensorShapeProto_Dimension,
        TypeProto, TypeProto_Tensor, TypeProto_oneof_value, ValueInfoProto,
    },
    utils::{get_attribute, AttributeNotFoundError, ScalarType, Shape},
};

pub fn apply_dynamic_dimensions(graph: &mut GraphProto, dynamic_dims: &HashMap<String, i64>) {
    // Apply to values
    for value_info in graph.mut_value_info() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }

    for value_info in graph.mut_input() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }

    for value_info in graph.mut_output() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }
}

/// Replaces dimension params with provided values
fn apply_dynamic_dimensions_value(
    value_info: &mut ValueInfoProto,
    dynamic_dims: &HashMap<String, i64>,
) {
    let name = value_info.get_name().to_string();
    let field_type = value_info.mut_field_type();

    if let Some(TypeProto_oneof_value::tensor_type(field_type_value)) = &mut field_type.value {
        let dims = field_type_value.mut_shape().mut_dim();

        for (idx, dim) in dims.iter_mut().enumerate() {
            if let Some(new_dim_value) = dynamic_dims.get(dim.get_dim_param()) {
                println!(
                    "Setting dimension param {idx} ({}) to value {new_dim_value} for {name}",
                    dim.get_dim_param()
                );
                dim.clear_dim_param();
                dim.set_dim_value(*new_dim_value);
            }
        }
    }
}

/// Retrieve all fully known value shapes
fn dimensions_infos(
    graph_proto: &GraphProto,
) -> Result<HashMap<String, Shape>, ShapeInferenceError> {
    let mut shapes_info = HashMap::new();

    for info in graph_proto.get_input() {
        if let Ok(shape) = info.get_shape() {
            shapes_info.insert(info.get_name().to_string(), shape);
        } else {
            return Err(ShapeInferenceError::IncompleteInputShape(
                info.get_name().to_string(),
            ));
        }
    }

    for info in graph_proto.get_output() {
        if let Ok(shape) = info.get_shape() {
            shapes_info.insert(info.get_name().to_string(), shape);
        }
    }

    for info in graph_proto.get_value_info() {
        if let Ok(shape) = info.get_shape() {
            shapes_info.insert(info.get_name().to_string(), shape);
        }
    }

    for info in graph_proto.get_initializer() {
        if let Ok(data_type) = ScalarType::from_i32(info.get_data_type()) {
            let shape = Shape::from(data_type, info.get_dims());
            shapes_info.insert(info.get_name().to_string(), shape);
        }
    }

    Ok(shapes_info)
}

#[derive(Error, Debug)]
pub enum ShapeInferenceError {
    #[error("missing shape for input {0}")]
    MissingInputShape(String),

    #[error("incomplete or missing shape for input {0} - be sure to specify all dynamic dimension parameters")]
    IncompleteInputShape(String),

    #[error("inference for {0} operator is not (fully) implemented yet")]
    Unsupported(String),

    #[error("node {0} is invalid: {1}")]
    InvalidNode(String, String),

    #[error("attribute {0} required for shape inference is missing")]
    #[from(AttributeNotFoundError)]
    MissingAttribute(AttributeNotFoundError),
}

pub fn infer_shapes(graph: &mut GraphProto) -> Result<(), ShapeInferenceError> {
    let mut shapes = dimensions_infos(graph)?;
    log::debug!("known shapes before shape inference: {shapes:#?}");

    for node in &mut graph.node {
        log::debug!(
            "Node: {} {} inputs {} -> outputs {}",
            node.get_op_type(),
            node.get_name(),
            node.get_input().join(", "),
            node.get_output().join(", ")
        );

        // If this node already has a shape, do not change it
        if !node
            .get_output()
            .iter()
            .any(|output_name| !shapes.contains_key(output_name.as_str()))
        {
            continue;
        }

        log::debug!("Node needs inference: {}", node.get_name());

        let input_shapes: Vec<&Shape> = node
            .get_input()
            .iter()
            .map(|name| {
                shapes
                    .get(name)
                    .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))
            })
            .collect::<Result<_, ShapeInferenceError>>()?;

        let output_shapes = infer_forward(node, &input_shapes)?;
        println!("Node {} Inferred: {:?}", node.get_name(), output_shapes);

        if output_shapes.len() != node.get_output().len() {
            panic!("number of outputs inferred does not match node output count");
        }

        // Cache the inferred shapes and write to model
        for (output_idx, output_name) in node.get_output().iter().enumerate() {
            let output_shape = &output_shapes[output_idx];
            shapes.insert(output_name.clone(), output_shape.clone());
            let mut vip = ValueInfoProto::new();
            vip.set_name(output_name.clone());

            let mut tip = TypeProto::new();
            let mut ttp = TypeProto_Tensor::new();
            ttp.set_elem_type(output_shape.data_type.to_datatype().value());

            let mut tsp = TensorShapeProto::new();
            tsp.set_dim(
                output_shape
                    .dims
                    .iter()
                    .map(|d| {
                        let mut tspd = TensorShapeProto_Dimension::new();
                        tspd.set_dim_value(*d as i64);
                        tspd
                    })
                    .collect(),
            );
            ttp.set_shape(tsp);
            tip.set_tensor_type(ttp);
            vip.set_field_type(tip);
            graph.value_info.push(vip);
        }
    }

    Ok(())
}

pub fn infer_forward(
    node: &NodeProto,
    input_shapes: &[&Shape],
) -> Result<Vec<Shape>, ShapeInferenceError> {
    match (
        node.get_op_type(),
        input_shapes.len(),
        node.get_output().len(),
    ) {
        ("Identity" | "Sqrt", 1, 1) => Ok(vec![input_shapes[0].clone()]),
        ("Gather", 2, 1) => {
            // https://github.com/onnx/onnx/blob/ceaeafa4cd2156c69dd9699bbdd2aa7d39e7c74c/onnx/defs/tensor/defs.cc#L1601
            let r = input_shapes[0].rank() as i64;
            if r < 1 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "data tensor must have rank 1 or greater".to_string(),
                ));
            }
            let q = input_shapes[1].rank() as i64;
            let mut axis = get_attribute("axis", Some(0), node)
                .map_err(ShapeInferenceError::MissingAttribute)?;
            if axis >= r || axis < -r {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "axis must be less than data tensor rank".to_string(),
                ));
            }

            if axis < 0 {
                axis += r;
            }
            let out_rank = q + r - 1;
            return Ok(vec![Shape::from(
                input_shapes[0].data_type,
                (0..out_rank)
                    .map(|idx| {
                        if idx < axis {
                            input_shapes[0].dim(idx as usize) as i64
                        } else if idx >= axis && idx < (axis + q) {
                            input_shapes[1].dim((idx - axis) as usize) as i64
                        } else {
                            input_shapes[0].dim((idx - q + 1) as usize) as i64
                        }
                    })
                    .collect::<Vec<i64>>()
                    .as_ref(),
            )]);
        }

        ("ReduceMean", 1, 1) => {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#reducemean-18
            let noop_with_empty_axes = get_attribute("noop_with_empty_axes", Some(0), node)
                .map_err(ShapeInferenceError::MissingAttribute)?;

            let input_shape = input_shapes[0];
            let input_ndim = input_shape.rank();
            let all_axes: Vec<i64> = if noop_with_empty_axes == 0 {
                (0..(input_shape.dims.len() as i64)).collect()
            } else {
                vec![]
            };
            let axes: Vec<i64> = get_attribute("axes", Some(all_axes), node)
                .map_err(ShapeInferenceError::MissingAttribute)?
                .into_iter()
                .map(|idx| {
                    if idx < 0 {
                        (input_ndim as i64) + idx
                    } else {
                        idx
                    }
                })
                .collect();
            let keep_dims = get_attribute("keepdims", Some(1), node)
                .map_err(ShapeInferenceError::MissingAttribute)?;

            Ok(vec![Shape::from(
                input_shape.data_type,
                (0..input_ndim as i64)
                    .flat_map(|i| {
                        if !axes.contains(&i) {
                            vec![input_shape.dim(i as usize) as i64]
                        } else if keep_dims == 1 {
                            vec![1]
                        } else {
                            vec![]
                        }
                    })
                    .collect::<Vec<_>>()
                    .as_ref(),
            )])
        }

        ("Sub" | "Pow" | "Add" | "Div" | "Mul", 2, 1) => {
            if let Some(output_shape) =
                Shape::multi_broadcast(&[input_shapes[0].clone(), input_shapes[1].clone()])
            {
                Ok(vec![output_shape])
            } else {
                Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "two inputs must be broadcastable".to_string(),
                ))
            }
        }

        ("Constant", 0, 1) => {
            if let Ok(values) = get_attribute::<Vec<f32>>("value_floats", None, node) {
                Ok(vec![Shape::from(ScalarType::F32, &[values.len() as i64])])
            } else if let Ok(values) = get_attribute::<Vec<i64>>("value_ints", None, node) {
                Ok(vec![Shape::from(ScalarType::I64, &[values.len() as i64])])
            } else if get_attribute::<f32>("value_float", None, node).is_ok() {
                Ok(vec![Shape::from(ScalarType::F32, &[1])])
            } else if get_attribute::<i64>("value_int", None, node).is_ok() {
                Ok(vec![Shape::from(ScalarType::I64, &[1])])
            } else if let Ok(tp) = get_attribute::<TensorProto>("value", None, node) {
                Ok(vec![Shape::from(
                    ScalarType::from_i32(tp.get_data_type()).map_err(|_| {
                        ShapeInferenceError::InvalidNode(
                            node.get_name().to_string(),
                            "invalid tensor data type".to_string(),
                        )
                    })?,
                    tp.get_dims(),
                )])
            } else {
                log::debug!("{:#?}", node);
                Err(ShapeInferenceError::Unsupported("Constant".to_string()))
            }
        }

        (
            "Sub" | "Pow" | "Add" | "Div" | "Mul" | "Identity" | "Sqrt" | "ReduceMean" | "Gather"
            | "Constant",
            _,
            _,
        ) => Err(ShapeInferenceError::InvalidNode(
            node.get_name().to_string(),
            "invalid number of in- or outputs".to_string(),
        )),

        (op_type, _inputs, _outputs) => {
            log::debug!("Shape inference unimplemented for op {op_type} with input shapes {input_shapes:#?}");
            Err(ShapeInferenceError::Unsupported(op_type.to_string()))
        }
    }
}
