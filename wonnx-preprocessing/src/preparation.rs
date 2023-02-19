use std::collections::HashMap;

use protobuf::ProtobufEnum;
use thiserror::Error;
use wonnx::{
    onnx::{
        GraphProto, NodeProto, TensorProto, TensorShapeProto, TensorShapeProto_Dimension,
        TypeProto, TypeProto_Tensor, TypeProto_oneof_value, ValueInfoProto,
    },
    utils::{AttributeNotFoundError, NodeAttributes, ScalarType, Shape},
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

pub trait ShapeInference {
    fn infer_shapes(&mut self) -> Result<(), ShapeInferenceError>;
}

/// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
fn div_ceil(num: i64, div: i64) -> i64 {
    num / div + (num % div != 0) as i64
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

impl ShapeInference for GraphProto {
    fn infer_shapes(self: &mut GraphProto) -> Result<(), ShapeInferenceError> {
        let mut shapes = dimensions_infos(self)?;
        log::debug!("known shapes before shape inference: {shapes:#?}");

        // Needed for Reshape
        let initializers: HashMap<String, &TensorProto> = HashMap::from_iter(
            self.initializer
                .iter()
                .map(|x| (x.get_name().to_string(), x)),
        );

        for node in &mut self.node {
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

            let output_shapes = infer_forward(node, &input_shapes, &initializers)?;
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
                self.value_info.push(vip);
            }
        }

        Ok(())
    }
}

fn infer_forward(
    node: &NodeProto,
    input_shapes: &[&Shape],
    initializers: &HashMap<String, &TensorProto>,
) -> Result<Vec<Shape>, ShapeInferenceError> {
    match (
        node.get_op_type(),
        input_shapes.len(),
        node.get_output().len(),
    ) {
        ("Identity" | "Sqrt" | "Relu", 1, 1) => Ok(vec![input_shapes[0].clone()]),

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

        ("ReduceMean", 1, 1) => {
            // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#reducemean-18
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

        ("Sub" | "Pow" | "Add" | "Div" | "Mul", 2, 1) => {
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
                // Get the tensor's contents
                let shape_tensor_contents = shape_tensor.get_int64_data();
                let shape_tensor_product: i64 = shape_tensor_contents.iter().product();

                if shape_tensor_product != input_shapes[0].element_count() as i64 {
                    return Err(ShapeInferenceError::InvalidNode(
            			node.get_name().to_string(),
						format!("Reshape shape tensor (element count={}) must have the same number of elements as the input tensor's rank ({})", shape_tensor_product, input_shapes[0].element_count())));
                }

                let allow_zero = node.get_attribute_value("allowzero", Some(0)).unwrap() == 1;

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
                Ok(vec![Shape::from(input_shapes[0].data_type, &output_shape)])
            } else {
                Err(ShapeInferenceError::Unsupported(
                    "Reshape with dynamic shape tensor".to_string(),
                ))
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

        (
            "Sub" | "Pow" | "Add" | "Div" | "Mul" | "Identity" | "Sqrt" | "ReduceMean" | "Gather"
            | "Constant" | "Relu" | "MaxPool" | "Conv" | "AveragePool" | "Reshape" | "Concat"
            | "Unsqueeze" | "Cast" | "Squeeze",
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

#[cfg(test)]
mod tests {
    use protobuf::Message;
    use wonnx::onnx::ModelProto;

    use super::{dimensions_infos, ShapeInference};

    /// Load a model, strip (and stash) all shape info for intermediate values, then re-infer shapes and compare with stashed original
    fn test_shape_inference_for_model(path: &str) {
        let mut model =
            ModelProto::parse_from_bytes(&std::fs::read(path).expect("ONNX Model path not found."))
                .unwrap();

        let graph = model.mut_graph();
        let infos = dimensions_infos(graph).unwrap();
        graph.value_info.clear();
        graph.infer_shapes().unwrap();
        let new_infos = dimensions_infos(graph).unwrap();
        assert_eq!(infos, new_infos);
    }

    #[test]
    fn test_shape_inference() {
        test_shape_inference_for_model("../data/models/opt-mnist.onnx");
        test_shape_inference_for_model("../data/models/opt-squeeze.onnx");
        test_shape_inference_for_model("../data/models/single_relu.onnx");
    }
}
