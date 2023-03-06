use std::{borrow::Cow, collections::HashMap};

use protobuf::ProtobufEnum;
use thiserror::Error;
use wonnx::{
    onnx::{
        GraphProto, NodeProto, TensorProto, TensorShapeProto, TensorShapeProto_Dimension,
        TypeProto, TypeProto_Tensor, TypeProto_oneof_value, ValueInfoProto,
    },
    utils::{
        AttributeNotFoundError, DataTypeError, InputTensor, NodeAttributes, ScalarType, Shape,
    },
};

use crate::constant_folding::{calculate_constant_node_outputs, ConstantFoldingError};

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

/// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
fn div_ceil(num: i64, div: i64) -> i64 {
    num / div + (num % div != 0) as i64
}

/// Retrieve the value of the initializer with the given name as a vector if i64 values.
fn static_initializer_value_i64<'a>(
    initializers: &'a HashMap<String, Cow<'a, TensorProto>>,
    name: &str,
) -> Result<&'a [i64], ShapeInferenceError> {
    if let Some(shape_tensor) = initializers.get(name) {
        if shape_tensor.get_data_type() != ScalarType::I64.to_datatype().value() {
            return Err(ShapeInferenceError::Unsupported(format!(
                "initializer {} has data type {} and not int64, which is currently not supported",
                name,
                shape_tensor.get_data_type()
            )));
        }

        let expected_value_count: i64 = shape_tensor.get_dims().iter().product();

        // Read data from the int64_data field, except when that field's contents don't match with what we expect; then try
        // the raw_data field.
        if shape_tensor.get_int64_data().len() != expected_value_count as usize {
            let raw_data = shape_tensor.get_raw_data();
            if raw_data.len() / 8 == expected_value_count as usize {
                // Raw data has the required size, use that (raw data should be little-endian, see https://github.com/onnx/onnx/issues/2825)
                log::warn!(
                    "int64 data for initializer {name} contains {} values, expected {expected_value_count}. Raw data length ({}) matches however, using that. dims={:?}",
                    shape_tensor.get_int64_data().len(),
                    shape_tensor.get_raw_data().len(),
                    shape_tensor.get_dims()
                );
                return Ok(bytemuck::cast_slice(raw_data));
            } else {
                log::warn!(
                    "int64 data for initializer {name} contains {} values, expected {expected_value_count}. Raw data length ({}) doesn't match either! dims={:?}",
                    shape_tensor.get_int64_data().len(),
                    shape_tensor.get_raw_data().len(),
                    shape_tensor.get_dims()
                );
            }
        }

        // Get the tensor's contents
        Ok(shape_tensor.get_int64_data())
    } else {
        Err(ShapeInferenceError::Unsupported(format!(
            "input {} is dynamic (only static initializers are supported)",
            name
        )))
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

/// Retrieve all fully known value shapes from a graph
pub(crate) fn dimensions_infos(
    graph_proto: &GraphProto,
) -> Result<HashMap<String, Shape>, DataTypeError> {
    let mut shapes_info = HashMap::new();

    for info in graph_proto.get_input() {
        if let Ok(shape) = info.get_shape() {
            shapes_info.insert(info.get_name().to_string(), shape);
        }
    }

    for info in graph_proto.get_output() {
        if let Ok(shape) = info.get_shape() {
            if shapes_info
                .insert(info.get_name().to_string(), shape)
                .is_some()
            {
                log::warn!(
                    "already had shape information for '{}', replacing from outputs",
                    info.get_name()
                );
            }
        }
    }

    for info in graph_proto.get_value_info() {
        if let Ok(shape) = info.get_shape() {
            if shapes_info
                .insert(info.get_name().to_string(), shape)
                .is_some()
            {
                log::warn!(
                    "already had shape information for '{}', replacing from value_info",
                    info.get_name()
                );
            }
        }
    }

    for info in graph_proto.get_initializer() {
        if let Ok(data_type) = ScalarType::from_i32(info.get_data_type()) {
            let shape = Shape::from(data_type, info.get_dims());
            if shapes_info
                .insert(info.get_name().to_string(), shape)
                .is_some()
            {
                log::warn!(
                    "already shape information for '{}', replacing from initializer",
                    info.get_name()
                );
            }
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

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("node {0} is invalid: {1}")]
    InvalidNode(String, String),

    #[error("attribute {0} required for shape inference is missing")]
    #[from(AttributeNotFoundError)]
    MissingAttribute(AttributeNotFoundError),

    #[error("unsupported data type encountered: {0}")]
    #[from(DataTypeError)]
    UnsupportedDataType(DataTypeError),

    #[error("constant folding failed: {0}")]
    #[from(ConstantFoldingError)]
    ConstantFoldingError(ConstantFoldingError),
}

/// Replaces nodes of op type Constant with an initializer
fn replace_constant_ops_with_initializers(
    graph: &mut GraphProto,
) -> Result<(), ShapeInferenceError> {
    for node_index in (0..graph.node.len()).rev() {
        let is_constant = graph.node[node_index].get_op_type() == "Constant";

        if is_constant {
            {
                let node = &graph.node[node_index];
                if node.get_output().len() != 1 {
                    return Err(ShapeInferenceError::InvalidNode(
                        node.get_name().to_string(),
                        format!(
                            "Constant op must have one output, has {}",
                            node.get_output().len()
                        ),
                    ));
                }

                // Create an initializer
                let mut initializer = TensorProto::new();

                // Get constant value
                if let Ok(values) = node.get_attribute_value::<Vec<f32>>("value_floats", None) {
                    initializer.set_data_type(ScalarType::F32.to_datatype().value());
                    initializer.set_dims(vec![values.len() as i64]);
                    initializer.set_float_data(values);
                } else if let Ok(values) = node.get_attribute_value::<Vec<i64>>("value_ints", None)
                {
                    initializer.set_data_type(ScalarType::I64.to_datatype().value());
                    initializer.set_dims(vec![values.len() as i64]);
                    initializer.set_int64_data(values);
                } else if let Ok(values) = node.get_attribute_value::<i64>("value_int", None) {
                    initializer.set_int64_data(vec![values]);
                    initializer.set_data_type(ScalarType::I64.to_datatype().value());
                    initializer.set_dims(vec![1]);
                } else if let Ok(values) = node.get_attribute_value::<f32>("value_float", None) {
                    initializer.set_float_data(vec![values]);
                    initializer.set_data_type(ScalarType::F32.to_datatype().value());
                    initializer.set_dims(vec![1]);
                } else if let Ok(tp) = node.get_attribute_value::<TensorProto>("value", None) {
                    initializer = tp;
                    fix_raw_tensor(&mut initializer)?;
                } else {
                    log::debug!("Constant node attributes: {:?}", node.attribute);
                    return Err(ShapeInferenceError::Unsupported(
                        "Constant node with data types other than float, int".to_string(),
                    ));
                }

                log::info!(
                    "Replacing Constant node '{}' with an initializer (name='{}', shape={:?})",
                    node.get_name(),
                    node.output[0].clone(),
                    initializer.dims
                );

                initializer.set_name(node.output[0].clone()); // Needs to happen here because the name can be overwritten above when there is a tensor in the "value" attribute
                graph.initializer.push(initializer);
            }
            graph.node.remove(node_index);
        }
    }
    Ok(())
}

pub async fn infer_shapes(
    graph: &mut GraphProto,
    should_fold_constants: bool,
    opset_version: i64,
) -> Result<(), ShapeInferenceError> {
    let mut foldable_nodes: Vec<String> = vec![];
    let mut folded_node_indexes: Vec<usize> = vec![];

    if should_fold_constants {
        replace_constant_ops_with_initializers(graph)?;
    }

    let mut shapes = dimensions_infos(graph).map_err(ShapeInferenceError::UnsupportedDataType)?;

    // Needed for Reshape
    let mut initializers: HashMap<String, Cow<TensorProto>> = HashMap::from_iter(
        graph
            .initializer
            .iter()
            .map(|x| (x.get_name().to_string(), Cow::Borrowed(x))),
    );

    for (node_index, node) in graph.node.iter().enumerate() {
        log::debug!(
            "node: {} {} inputs {} -> outputs {}",
            node.get_op_type(),
            node.get_name(),
            node.get_input().join(", "),
            node.get_output().join(", ")
        );

        // Do shape inference if this node has at least one output for which the shape is not yet known
        if node
            .get_output()
            .iter()
            .any(|output_name| !shapes.contains_key(output_name.as_str()))
        {
            log::debug!("node needs shape inference: {}", node.get_name());

            let input_shapes: Vec<&Shape> = node
                .get_input()
                .iter()
                .map(|name| {
                    shapes
                        .get(name)
                        .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))
                })
                .collect::<Result<_, ShapeInferenceError>>()?;

            let output_shapes = infer_output_shapes(node, &input_shapes, &initializers)?;

            // Check inferred shapes
            for (output_index, shape) in output_shapes.iter().enumerate() {
                if shape.rank() == 0 {
                    log::warn!(
                        "inferred shape for output {output_index} of node '{}' is empty: {shape}",
                        node.get_name()
                    );
                }
            }

            log::info!(
                "node {} inferred output shapes: {}",
                node.get_name(),
                output_shapes
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| format!("{}={x}", node.output[idx]))
                    .collect::<Vec<String>>()
                    .join(", ")
            );

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

            // Can we fold the node altogether?
            let can_fold = should_fold_constants && {
                let all_inputs_are_constant = node
                    .input
                    .iter()
                    .all(|input_name| initializers.contains_key(input_name));
                let is_known_shape_node =
                    node.get_op_type() == "Shape" && shapes.contains_key(&node.input[0]);
                all_inputs_are_constant || is_known_shape_node
            };

            if can_fold {
                log::debug!("node '{}' can be folded", node.get_name());

                // Collect constant inputs
                let inputs: Vec<InputTensor> = node
                    .input
                    .iter()
                    .map(|input_name| {
                        if let Some(initializer) = initializers.get(input_name) {
                            InputTensor::try_from(initializer.as_ref())
                        } else {
                            // This should only happen when is_known_shape is true. In this case we will not do any GPU inference
                            // and the contents if this tensor don't matter
                            Ok(InputTensor::I64(Cow::Owned(vec![])))
                        }
                    })
                    .collect::<Result<_, _>>()
                    .map_err(|x| {
                        ShapeInferenceError::ConstantFoldingError(
                            ConstantFoldingError::UnsupportedDataType(x),
                        )
                    })?;

                if let Some(mut constant_output) = calculate_constant_node_outputs(
                    node,
                    &shapes,
                    &inputs,
                    &output_shapes,
                    &initializers,
                    opset_version,
                )
                .await
                .map_err(ShapeInferenceError::ConstantFoldingError)?
                {
                    // Save constant outputs as initializers
                    for (output_index, output_name) in node.output.iter().enumerate().rev() {
                        let output_tensor = constant_output.remove(output_index);

                        let output_shape = &output_shapes[output_index];
                        let mut initializer: TensorProto = output_tensor.into();
                        initializer.set_name(output_name.clone());
                        initializer.set_dims(output_shape.dims.iter().map(|x| *x as i64).collect());
                        initializers.insert(output_name.clone(), Cow::Owned(initializer));

                        assert_eq!(
                            &shapes[output_name], output_shape,
                            "output shape should be the same after folding"
                        );
                        folded_node_indexes.push(node_index);

                        log::info!(
                            "folded output '{output_name}' (#{output_index}) of node {} shape={output_shape}",
                            node.get_name(),
                        );
                    }
                } else {
                    foldable_nodes.push(node.get_name().to_string());
                }
            }
        }
    }

    // Remove folded nodes
    folded_node_indexes.sort();
    for index in folded_node_indexes.iter().rev() {
        graph.node.remove(*index);
    }

    // Save newly created initializers
    let new_initializers: Vec<TensorProto> = initializers
        .into_iter()
        .flat_map(|(_, x)| match x {
            Cow::Owned(z) => Some(z),
            Cow::Borrowed(_) => None,
        })
        .collect();

    for new_initializer in new_initializers {
        graph.initializer.push(new_initializer);
    }

    // Notify about missing fold implementations
    if !foldable_nodes.is_empty() {
        log::info!(
            "The following nodes can likely be folded, but currently aren't due to missing support: {}",
            foldable_nodes.join(", ")
        );
    }

    Ok(())
}

pub(crate) fn infer_output_shapes(
    node: &NodeProto,
    input_shapes: &[&Shape],
    initializers: &HashMap<String, Cow<TensorProto>>,
) -> Result<Vec<Shape>, ShapeInferenceError> {
    match (
        node.get_op_type(),
        input_shapes.len(),
        node.get_output().len(),
    ) {
        ("Clip", 1..=3, 1)
        | (
            "Identity" | "Sqrt" | "Relu" | "LeakyRelu" | "Abs" | "Acos" | "Acosh" | "Asin" | "Sin"
            | "Asinh" | "Atan" | "Atanh" | "Cos" | "Cosh" | "Elu" | "Erf" | "Exp" | "Log" | "Neg"
            | "Ceil" | "Floor" | "Reciprocal" | "Celu",
            1,
            1,
        ) => Ok(vec![input_shapes[0].clone()]),

        ("Cast", 1, 1) => {
            let to_value: i64 = node
                .get_attribute_value("to", None)
                .map_err(ShapeInferenceError::MissingAttribute)?;
            let to_data_type = ScalarType::from_i32(to_value as i32).map_err(|_| {
                ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!(
                        "invalid value for to attribute ({}) for Cast operator",
                        to_value
                    ),
                )
            })?;

            let mut output_shape = input_shapes[0].clone();
            output_shape.data_type = to_data_type;

            Ok(vec![output_shape])
        }

        ("GlobalAveragePool", 1, 1) => {
            let mut output_shape = input_shapes[0].clone();
            if output_shape.rank() < 2 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!("invalid input rank for GlobalAveragePool: {output_shape}",),
                ));
            }
            for a in 2..output_shape.dims.len() {
                output_shape.dims[a] = 1;
            }
            Ok(vec![output_shape])
        }

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
            let mut axis = node
                .get_attribute_value("axis", Some(0))
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

        ("Shape", 1, 1) => {
            let rank = input_shapes[0].rank() as i64;
            let start: i64 = node.get_attribute_value("start", Some(0)).unwrap();
            let end: i64 = node.get_attribute_value("end", Some(rank)).unwrap();

            Ok(vec![Shape::from(
                ScalarType::I64,
                &[rank.clamp(start, end)],
            )])
        }

        ("Slice", num_inputs @ 3..=5, 1) => {
            let data_shape = input_shapes[0];

            // All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
            // where `dims` are the dimensions of `input`.
            let mut starts: Vec<i64> =
                static_initializer_value_i64(initializers, &node.get_input()[1])?
                    .iter()
                    .enumerate()
                    .map(|(idx, s)| {
                        if *s < 0 {
                            *s + data_shape.dim(idx) as i64
                        } else {
                            *s
                        }
                    })
                    .collect();
            if starts.is_empty() {
                log::warn!(
                    "starts not set for Slice, generating it... name={}",
                    node.get_input()[1]
                );
                starts = (0..data_shape.rank()).map(|_| 1).collect();
            }
            let mut ends: Vec<i64> =
                static_initializer_value_i64(initializers, &node.get_input()[2])?
                    .iter()
                    .enumerate()
                    .map(|(idx, s)| {
                        if *s < 0 {
                            *s + data_shape.dim(idx) as i64
                        } else {
                            *s
                        }
                    })
                    .collect();
            if ends.is_empty() {
                log::warn!("ends not set for Slice, generating it...");
                ends = data_shape.dims.iter().map(|x| *x as i64).collect();
            }

            // If `axes` are omitted, they are set to `[0, ..., r-1]`.
            let axes: Vec<i64> = if num_inputs > 3 {
                let x: Vec<i64> =
                    static_initializer_value_i64(initializers, &node.get_input()[3])?.into();
                if x.is_empty() {
                    (0..(data_shape.rank() as i64)).collect()
                } else {
                    x
                }
            } else {
                (0..(data_shape.rank() as i64)).collect()
            };

            // If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
            let steps: Vec<i64> = if num_inputs > 4 {
                static_initializer_value_i64(initializers, &node.get_input()[4])?.into()
            } else {
                log::debug!(
                    "steps not set for slice, generating it (data_shape rank={})",
                    data_shape.rank()
                );
                (0..(data_shape.rank() as i64)).map(|_| 1).collect()
            };

            if axes.len() != steps.len() {
                return Err(ShapeInferenceError::InvalidNode(node.get_name().to_string(), format!("length of axes attribute ({}) must be equal to length of steps attribute ({})", axes.len(), steps.len())));
            }

            // All negative elements of `axes` are made non-negatve by adding `r` to them, where`r =rank(input)`.
            let axes: Vec<i64> = axes
                .into_iter()
                .map(|x| {
                    if x < 0 {
                        x + data_shape.rank() as i64
                    } else {
                        x
                    }
                })
                .collect();

            let mut output_shape: Vec<i64> =
                input_shapes[0].dims.iter().map(|x| *x as i64).collect();

            // https://github.com/onnx/onnx/blob/fb80e3ade84e9f406711aa41b9f3665753158371/onnx/defs/tensor/defs.cc#L969
            for (axis_index, axis) in axes.iter().enumerate() {
                let mut start = starts[axis_index];
                let mut end = ends[axis_index];
                let mut step = steps[axis_index];
                process_slice_inputs(
                    data_shape.dim(*axis as usize) as i64,
                    &mut start,
                    &mut end,
                    &mut step,
                )?;
                let temp = div_ceil(end - start, step).max(0);
                output_shape[*axis as usize] = temp;
            }

            Ok(vec![Shape::from(data_shape.data_type, &output_shape)])
        }

        (
            "ReduceMean" | "ReduceSum" | "ReduceMin" | "ReduceMax" | "ReduceSumSquare"
            | "ReduceLogSumExp" | "ReduceLogSum" | "ReduceL2" | "ReduceL1" | "ReduceProd",
            1,
            1,
        ) => {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#reducemean-18
            // Note: up to version 13 these ops take 'axes' as an attribute; from version 18 they take axes as a second (optional) attribute
            let noop_with_empty_axes = node
                .get_attribute_value("noop_with_empty_axes", Some(0))
                .map_err(ShapeInferenceError::MissingAttribute)?;

            let input_shape = input_shapes[0];
            let input_ndim = input_shape.rank();
            let all_axes: Vec<i64> = if noop_with_empty_axes == 0 {
                (0..(input_shape.dims.len() as i64)).collect()
            } else {
                vec![]
            };
            let axes: Vec<i64> = node
                .get_attribute_value("axes", Some(all_axes))
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
            let keep_dims = node
                .get_attribute_value("keepdims", Some(1))
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

        ("Sub" | "Pow" | "Add" | "Div" | "Mul" | "Mod", 2, 1) => {
            if let Some(output_shape) =
                Shape::multi_broadcast(&[input_shapes[0].clone(), input_shapes[1].clone()])
            {
                Ok(vec![output_shape])
            } else {
                Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!(
                        "two inputs (left {} shape: {}, right {} shape: {}) must be broadcastable",
                        node.get_input()[0],
                        node.get_input()[1],
                        input_shapes[0],
                        input_shapes[1]
                    ),
                ))
            }
        }

        ("Conv", 2, num_outputs @ 1)
        | ("Conv", 3, num_outputs @ 1)
        | ("MaxPool", 1, num_outputs @ 1)
        | ("MaxPool", 1, num_outputs @ 2)
        | ("AveragePool", 1, num_outputs @ 1)
        | ("AveragePool", 1, num_outputs @ 2) => {
            // https://github.com/onnx/onnx/blob/ded7e3a27449750fb429b0f88a494e10fd555be7/onnx/defs/nn/old.cc#L240
            let use_dilation = true;
            let require_kernel_shape = matches!(node.get_op_type(), "MaxPool" | "AveragePool");
            let input_shape = input_shapes[0];
            if input_shape.rank() < 2 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "input shape must have at least two dimensions".to_string(),
                ));
            }

            let num_input_dims = input_shape.rank() - 2;

            // Obtain dilations info
            let dilations: Vec<i64> = if use_dilation && node.has_attribute("dilations") {
                let dilations_attr: Vec<i64> = node
                    .get_attribute_value("dilations", None)
                    .map_err(ShapeInferenceError::MissingAttribute)?;
                if dilations_attr.len() != num_input_dims {
                    return Err(ShapeInferenceError::InvalidNode(
                        node.get_name().to_string(),
                        "attribute dilations has incorrect size".to_string(),
                    ));
                }
                dilations_attr
            } else {
                (0..num_input_dims).map(|_| 1).collect()
            };

            // Obtain stride info
            let strides: Vec<i64> = if use_dilation && node.has_attribute("strides") {
                let strides_attr: Vec<i64> = node
                    .get_attribute_value("strides", None)
                    .map_err(ShapeInferenceError::MissingAttribute)?;
                if strides_attr.len() != num_input_dims {
                    return Err(ShapeInferenceError::InvalidNode(
                        node.get_name().to_string(),
                        "attribute strides has incorrect size".to_string(),
                    ));
                }
                strides_attr
            } else {
                (0..num_input_dims).map(|_| 1).collect()
            };

            // Obtain kernel shape
            let kernel_shape = if node.has_attribute("kernel_shape") {
                node.get_attribute_value::<Vec<i64>>("kernel_shape", None)
                    .map_err(ShapeInferenceError::MissingAttribute)?
            } else if require_kernel_shape {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "node requires kernel_shape to be set".to_string(),
                ));
            } else {
                // Use second input shape to derive kernel shape
                input_shapes[1].dims[2..]
                    .iter()
                    .map(|x| *x as i64)
                    .collect()
            };

            if kernel_shape.len() != num_input_dims {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "kernel shape rank must be equal to input rank".to_string(),
                ));
            }

            // Determine effective kernel shape
            let effective_kernel_shape: Vec<i64> = kernel_shape
                .iter()
                .enumerate()
                .map(|(idx, dim)| (*dim - 1) * dilations[idx] + 1)
                .collect();

            // Obtain pads information
            let pads = if node.has_attribute("pads") {
                let p = node
                    .get_attribute_value::<Vec<i64>>("pads", None)
                    .map_err(ShapeInferenceError::MissingAttribute)?;
                if p.len() != num_input_dims * 2 {
                    return Err(ShapeInferenceError::InvalidNode(
                        node.get_name().to_string(),
                        "pads attribute has incorrect size".to_string(),
                    ));
                }
                p
            } else {
                let mut pads: Vec<i64> = (0..num_input_dims * 2).map(|_| 0).collect();
                let auto_pad = node
                    .get_attribute_value("auto_pad", Some(String::from("VALID")))
                    .unwrap();

                if auto_pad != "VALID" {
                    for i in 0..num_input_dims {
                        let mut residual: i64 = 0;
                        let stride = strides[i];

                        if stride > 1 {
                            residual = input_shape.dim(2 + i) as i64;
                            while residual >= stride {
                                residual -= stride;
                            }
                        }

                        let mut total_pad = if residual == 0 {
                            effective_kernel_shape[i] - stride
                        } else {
                            effective_kernel_shape[i] - residual
                        };
                        if total_pad < 0 {
                            total_pad = 0;
                        }

                        let half_pad_small = total_pad >> 1;
                        let half_pad_big = total_pad - half_pad_small;
                        if auto_pad == "SAME_UPPER" {
                            pads[i] = half_pad_small;
                            pads[i + num_input_dims] = half_pad_big;
                        } else if auto_pad == "SAME_LOWER" {
                            pads[i] = half_pad_big;
                            pads[i + num_input_dims] = half_pad_small;
                        }
                    }
                }
                pads
            };

            // Determine output shape
            let mut output_shape: Vec<i64> = vec![];
            output_shape.push(input_shape.dim(0) as i64);
            if require_kernel_shape {
                output_shape.push(input_shape.dim(1) as i64);
            } else {
                if input_shapes[1].rank() < 1 {
                    return Err(ShapeInferenceError::InvalidNode(
                        node.get_name().to_string(),
                        "second input has incorrect rank".to_string(),
                    ));
                }
                output_shape.push(input_shapes[1].dim(0) as i64);
            }

            let kernel_shape_size = kernel_shape.len();
            for i in 0..kernel_shape_size {
                // how big is the input, including padding
                let mut effective_input_size: i64 = input_shape.dim(2 + i) as i64;
                effective_input_size += pads[i];
                effective_input_size += pads[i + kernel_shape_size];

                // default is floor mode .i.e. ceil_mode is set to 0
                let ceil_mode = node.get_attribute_value("ceil_mode", Some(0)).unwrap();

                // how many times we can move the kernel from it's initial position, based
                // on the stride
                let strided_kernel_positions = if ceil_mode == 1 {
                    div_ceil(effective_input_size - effective_kernel_shape[i], strides[i])
                } else {
                    (effective_input_size - effective_kernel_shape[i]) / strides[i]
                };

                output_shape.push(1 + strided_kernel_positions);
            }

            // MaxPool can have two outputs
            let final_output_shape = Shape::from(input_shape.data_type, &output_shape);
            Ok((0..num_outputs)
                .map(|_| final_output_shape.clone())
                .collect())
        }

        ("ConstantOfShape", 1, 1) => {
            let shape = static_initializer_value_i64(initializers, &node.get_input()[0])?;

            let value = node
                .get_attribute_value::<TensorProto>("value", None)
                .map_err(ShapeInferenceError::MissingAttribute)?;

            let data_type = ScalarType::from_i32(value.get_data_type())
                .map_err(ShapeInferenceError::UnsupportedDataType)?;

            Ok(vec![Shape::from(data_type, shape)])
        }

        ("Constant", 0, 1) => {
            if let Ok(values) = node.get_attribute_value::<Vec<f32>>("value_floats", None) {
                Ok(vec![Shape::from(ScalarType::F32, &[values.len() as i64])])
            } else if let Ok(values) = node.get_attribute_value::<Vec<i64>>("value_ints", None) {
                Ok(vec![Shape::from(ScalarType::I64, &[values.len() as i64])])
            } else if node.get_attribute_value::<f32>("value_float", None).is_ok() {
                Ok(vec![Shape::from(ScalarType::F32, &[1])])
            } else if node.get_attribute_value::<i64>("value_int", None).is_ok() {
                Ok(vec![Shape::from(ScalarType::I64, &[1])])
            } else if let Ok(tp) = node.get_attribute_value::<TensorProto>("value", None) {
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

        ("Reshape", 2, 1) => {
            let shape_tensor_name = &node.get_input()[1];

            if let Some(shape_tensor) = initializers.get(shape_tensor_name) {
                let allow_zero = node.get_attribute_value("allowzero", Some(0)).unwrap() == 1;

                // Get the tensor's contents
                let shape_tensor_contents = shape_tensor.get_int64_data();

                // The -1 value is allowed but not supported
                for dim in shape_tensor_contents {
                    match *dim {
						-1 => return Err(ShapeInferenceError::Unsupported(
                            "Reshape with shape containing a -1 element".to_string(),
                        )),
						i64::MIN..=-1 => return Err(ShapeInferenceError::InvalidNode(
            			node.get_name().to_string(),
						format!("Reshape shape tensor cannot contain negative values except for -1 (contains {})", dim))),
						0..=i64::MAX => ()
					}
                }

                let output_shape: Vec<i64> = shape_tensor_contents
                    .iter()
                    .enumerate()
                    .map(|(idx, dim)| {
                        if *dim == 0 && !allow_zero {
                            input_shapes[0].dim(idx) as i64
                        } else {
                            *dim
                        }
                    })
                    .collect();

                if output_shape.iter().product::<i64>() != input_shapes[0].element_count() as i64 {
                    return Err(ShapeInferenceError::InvalidNode(
            			node.get_name().to_string(),
						format!("Reshape input tensor (element count={}) must have the same number of elements as specified by the new shape ({})", input_shapes[0].element_count(), output_shape.iter().product::<i64>())));
                }

                Ok(vec![Shape::from(input_shapes[0].data_type, &output_shape)])
            } else {
                Err(ShapeInferenceError::Unsupported(format!(
                    "Reshape with dynamic shape tensor (input name is {shape_tensor_name})"
                )))
            }
        }

        ("Concat", 1.., 1) => {
            let axis = node
                .get_attribute_value::<i64>("axis", None)
                .map_err(ShapeInferenceError::MissingAttribute)?;

            // All input shapes must be the same except for the dimension at the specified axis
            let mut shape: Vec<i64> = input_shapes[0].dims.iter().map(|x| *x as i64).collect();
            if axis < -(shape.len() as i64) || axis > (shape.len() - 1) as i64 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    "axis attribute needs to be smaller than input tensor rank".to_string(),
                ));
            }

            let axis_index = if axis < 0 {
                ((shape.len() as i64) + axis) as usize
            } else {
                axis as usize
            };
            shape[axis_index] = input_shapes.iter().map(|s| s.dim(axis_index) as i64).sum();
            Ok(vec![Shape::from(input_shapes[0].data_type, &shape)])
        }

        ("Dropout", 1..=3, num_outputs @ 1..=2) => {
            let shape = input_shapes[0];
            Ok((0..num_outputs).map(|_| shape.clone()).collect())
        }

        ("Unsqueeze", num_inputs @ 1..=2, 1) => {
            let axes: Vec<i64> = if num_inputs == 2 {
                let shape_tensor_name = &node.get_input()[1];
                if let Some(shape_tensor) = initializers.get(shape_tensor_name) {
                    // Get the tensor's contents
                    shape_tensor.get_int64_data().to_vec()
                } else {
                    return Err(ShapeInferenceError::Unsupported(
                        "Unsqueeze with dynamic axis inputs".to_string(),
                    ));
                }
            } else {
                node.get_attribute_value("axes", None)
                    .map_err(ShapeInferenceError::MissingAttribute)?
            };

            let output_rank = input_shapes[0].rank() + axes.len();
            let mut input_shape: Vec<i64> =
                input_shapes[0].dims.iter().map(|x| *x as i64).collect();
            for i in axes {
                let index = if i < 0 {
                    ((output_rank as i64) + i) as usize
                } else {
                    i as usize
                };
                input_shape.insert(index, 1);
            }

            Ok(vec![Shape::from(input_shapes[0].data_type, &input_shape)])
        }

        ("Range", 3, 1) => {
            // Currently only int64 ranges are supported
            let start = static_initializer_value_i64(initializers, &node.input[0])?;
            let end = static_initializer_value_i64(initializers, &node.input[1])?;
            let step = static_initializer_value_i64(initializers, &node.input[2])?;

            if start.len() != 1 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!(
                        "the start input needs to be a scalar, has {} elements",
                        start.len()
                    ),
                ));
            }

            if end.len() != 1 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!(
                        "the end input needs to be a scalar, has {} elements",
                        end.len()
                    ),
                ));
            }

            if step.len() != 1 {
                return Err(ShapeInferenceError::InvalidNode(
                    node.get_name().to_string(),
                    format!(
                        "the step input needs to be a scalar, has {} elements",
                        step.len()
                    ),
                ));
            }

            let element_count = (end[0] - start[0]) / step[0];
            Ok(vec![Shape::from(ScalarType::I64, &[element_count])])
        }

        ("Squeeze", num_inputs @ 1..=2, 1) => {
            let has_axes = num_inputs == 2;
            let axes: Vec<i64> = if has_axes {
                let shape_tensor_name = &node.get_input()[1];
                if let Some(shape_tensor) = initializers.get(shape_tensor_name) {
                    // Get the tensor's contents
                    shape_tensor.get_int64_data().to_vec()
                } else {
                    return Err(ShapeInferenceError::Unsupported(
                        "Unsqueeze with dynamic axis inputs".to_string(),
                    ));
                }
            } else {
                vec![]
            };

            let output_shape: Vec<i64> = input_shapes[0]
                .dims
                .iter()
                .enumerate()
                .flat_map(|(idx, dim)| {
                    if (has_axes && axes.contains(&(idx as i64))) || (!has_axes && *dim == 1) {
                        vec![]
                    } else {
                        vec![*dim as i64]
                    }
                })
                .collect();

            Ok(vec![Shape::from(input_shapes[0].data_type, &output_shape)])
        }

        ("Transpose", 1, 1) => {
            let input_dims: Vec<i64> = input_shapes[0].dims.iter().map(|x| *x as i64).collect();
            let output_dims: Vec<i64> = match node.get_attribute_value::<Vec<i64>>("perm", None) {
                Ok(perm) => perm.iter().map(|idx| input_dims[*idx as usize]).collect(),
                Err(_) => input_dims.iter().rev().cloned().collect(),
            };
            Ok(vec![Shape::from(input_shapes[0].data_type, &output_dims)])
        }

        ("BatchNormalization", 1.., 1) => {
            // The first output's shape is equal to the input's shape
            Ok(vec![input_shapes[0].clone()])
        }

        (
            "ReduceMean" | "ReduceSum" | "ReduceMin" | "ReduceMax" | "ReduceSumSquare"
            | "ReduceLogSumExp" | "ReduceLogSum" | "ReduceL2" | "ReduceL1" | "ReduceProd",
            2,
            1,
        ) => Err(ShapeInferenceError::Unsupported(format!(
            "{} with two inputs (axes input not supported)",
            node.get_op_type()
        ))),

        (
            "Sub" | "Pow" | "Add" | "Div" | "Mul" | "Identity" | "Sqrt" | "ReduceMean" | "Gather"
            | "Constant" | "Relu" | "LeakyRelu" | "MaxPool" | "Conv" | "AveragePool" | "Reshape"
            | "Concat" | "Unsqueeze" | "Cast" | "Squeeze" | "Shape" | "Slice" | "Range"
            | "ConstantOfShape" | "Transpose" | "Abs" | "Acos" | "Acosh" | "Asin" | "Sin" | "Asinh"
            | "Atan" | "Atanh" | "Cos" | "Cosh" | "Elu" | "Erf" | "Exp" | "Log" | "Neg" | "Ceil"
            | "Reciprocal" | "Floor" | "Mod" | "Celu" | "ReduceSum" | "ReduceMin" | "ReduceMax"
            | "ReduceSumSquare" | "ReduceLogSumExp" | "ReduceLogSum" | "ReduceL2" | "ReduceL1"
            | "ReduceProd",
            _,
            _,
        ) => Err(ShapeInferenceError::InvalidNode(
            node.get_name().to_string(),
            format!(
                "invalid number of inputs ({}) or outputs ({})",
                node.get_input().len(),
                node.get_output().len()
            ),
        )),

        (op_type, _inputs, _outputs) => {
            log::debug!("Shape inference unimplemented for op {op_type} with input shapes {input_shapes:#?}");
            Err(ShapeInferenceError::Unsupported(op_type.to_string()))
        }
    }
}

/// https://github.com/onnx/onnx/blob/fb80e3ade84e9f406711aa41b9f3665753158371/onnx/defs/tensor/defs.cc#L814
fn process_slice_inputs(
    input_rank: i64,
    start: &mut i64,
    end: &mut i64,
    step: &mut i64,
) -> Result<(), ShapeInferenceError> {
    // process step
    if *step == 0 {
        return Err(ShapeInferenceError::InvalidNode(
            "".to_string(),
            "step value must not be zero for slice".to_string(),
        ));
    }
    // process start
    if *start < 0 {
        *start += input_rank;
    }
    if *step < 0 {
        *start = (*start).clamp(0, input_rank - 1);
    } else {
        *start = (*start).clamp(0, input_rank);
    }

    // process end
    if *end < 0 {
        *end += input_rank;
    }
    if *step < 0 {
        *end = (*end).clamp(-1, input_rank - 1);
    } else {
        *end = (*end).clamp(0, input_rank);
    }
    Ok(())
}

/// Some tensors only have the raw data field filled. This function moves that data to the respective fields (i.e. int64_data)
/// depending on the data type specified.
fn fix_raw_tensor(tensor: &mut TensorProto) -> Result<(), ShapeInferenceError> {
    if tensor.has_raw_data() {
        let raw_data = tensor.take_raw_data();
        match ScalarType::from_i32(tensor.get_data_type())
            .map_err(ShapeInferenceError::UnsupportedDataType)?
        {
            ScalarType::F32 => tensor.set_float_data(bytemuck::cast_slice(&raw_data[..]).to_vec()),
            ScalarType::I64 => tensor.set_int64_data(bytemuck::cast_slice(&raw_data[..]).to_vec()),
            ScalarType::I32 => tensor.set_int32_data(bytemuck::cast_slice(&raw_data[..]).to_vec()),
            ScalarType::U8 => tensor.set_raw_data(bytemuck::cast_slice(&raw_data[..]).to_vec()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use protobuf::Message;
    use wonnx::onnx::ModelProto;

    use crate::shape_inference::infer_shapes;

    use super::dimensions_infos;

    /// Load a model, strip (and stash) all shape info for intermediate values, then re-infer shapes and compare with stashed original
    async fn test_shape_inference_for_model(path: &str, should_fold_constants: bool) {
        let mut model =
            ModelProto::parse_from_bytes(&std::fs::read(path).expect("ONNX Model path not found."))
                .unwrap();

        let graph = model.mut_graph();
        let infos = dimensions_infos(graph).unwrap();
        graph.value_info.clear();
        infer_shapes(graph, should_fold_constants, 13)
            .await
            .unwrap();
        let new_infos = dimensions_infos(graph).unwrap();

        let keys_in_old: HashSet<String> = infos.keys().cloned().collect();
        let keys_in_new: HashSet<String> = new_infos.keys().cloned().collect();
        let all_keys: HashSet<String> = keys_in_old.union(&keys_in_new).cloned().collect();

        for key in all_keys {
            if !keys_in_old.contains(&key) || !keys_in_new.contains(&key) || infos[&key].is_empty()
            {
                // Key is new after shape inference (the original model was apparently missing shapes)
                // Key missing after shape inference (this may be the result of constant folding)
                // Empty dims in source means missing... ignore
            } else {
                assert_eq!(
                    infos[&key], new_infos[&key],
                    "different shape inferred for {key}"
                )
            }
        }
    }

    #[test]
    fn test_shape_inference() {
        let _ = env_logger::builder().is_test(true).try_init();

        pollster::block_on(async {
            test_shape_inference_for_model("../data/models/opt-mnist.onnx", false).await;
            test_shape_inference_for_model("../data/models/opt-squeeze.onnx", false).await;
            test_shape_inference_for_model("../data/models/single_relu.onnx", false).await;
            test_shape_inference_for_model("../data/models/single_relu.onnx", false).await;
            test_shape_inference_for_model("../data/models/mobilenetv2-7.onnx", true).await;
        });
    }
}
