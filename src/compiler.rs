use crate::utils::{ceil, get_attribute, AttributeNotFoundError, DataType, Shape};
use std::collections::HashMap;
use tera::{Context, Tera};
use thiserror::Error;

// Escaping special characters as well as adding `var_` in the beginning of the variable name to avoid collisions with wgsl syntax.
// FIXME: this has the potential to cause collisions (i.e. "a/b" and "a.b" will both translate to "ab" and hence cannot be used simultaneously)
fn to_wgsl_variable_name(input_or_output_name: &str) -> String {
    String::from("var_")
        + &input_or_output_name.replace(&['(', ')', ',', '\"', '.', ';', ':', '\'', '/'][..], "")
}

lazy_static! {
    // Templates for shader source code that we generate for nodes
    pub static ref TEMPLATES: Tera = {
        let mut tera = Tera::default();
        tera.add_raw_template(
            "endomorphism/activation.wgsl",
            include_str!("../templates/endomorphism/activation.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/arithmetic.wgsl",
            include_str!("../templates/endomorphism/arithmetic.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/batchnormalization.wgsl",
            include_str!("../templates/endomorphism/batchnormalization.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/copy.wgsl",
            include_str!("../templates/endomorphism/copy.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/softmax.wgsl",
            include_str!("../templates/endomorphism/softmax.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/map.wgsl",
            include_str!("../templates/endomorphism/map.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/concat.wgsl",
            include_str!("../templates/matrix/concat.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm_1.wgsl",
            include_str!("../templates/matrix/gemm_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm.wgsl",
            include_str!("../templates/matrix/gemm.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/resize.wgsl",
            include_str!("../templates/matrix/resize.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/split.wgsl",
            include_str!("../templates/matrix/split.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/transpose.wgsl",
            include_str!("../templates/matrix/transpose.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/aggregate.wgsl",
            include_str!("../templates/pool/aggregate.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_1.wgsl",
            include_str!("../templates/pool/conv_kernel_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_3.wgsl",
            include_str!("../templates/pool/conv_kernel_3.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv.wgsl",
            include_str!("../templates/pool/conv.wgsl"),
        )
        .unwrap();
        tera.add_raw_template("structs.wgsl", include_str!("../templates/structs.wgsl"))
            .unwrap();
        tera.add_raw_template(
            "snippets/activation_vec.wgsl",
            include_str!("../templates/snippets/activation_vec.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "snippets/activation_scalar.wgsl",
            include_str!("../templates/snippets/activation_scalar.wgsl"),
        )
        .unwrap();
        tera
    };
}

pub struct CompiledNode {
    pub shader: String,
    pub threads: (u32, u32, u32),
}

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("dimensions information missing for input/output '{0}' of node '{1}'. You may want to run onnx-simplifier on the model first.")]
    DimensionsMissing(String, String),

    #[error("attribute not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),

    #[error("operation not recognized: {0}")]
    InvalidOperation(String),

    #[error("op {0} is not implemented yet! Check the README if you want to implement it")]
    UnimplementedOp(String),

    #[error("'{variant}' is not yet implemented for op {op}")]
    UnimplementedVariant { variant: String, op: String },

    #[error("the opset version {0} is not supported")]
    UnsupportedOpsetVersion(i64),

    #[error("the value '{attribute}' is invalid for attribute '{value}' (opset version {opset_version})")]
    InvalidAttributeValue {
        attribute: String,
        value: String,
        opset_version: i64,
    },

    #[error("input {input_index} has invalid shape {input_shape}")]
    InvalidInputShape {
        input_index: usize,
        input_shape: Shape,
    },
}

pub fn compile(
    node: &crate::onnx::NodeProto,
    shape_infos: &HashMap<String, Shape>,
    opset_version: i64,
) -> Result<CompiledNode, CompileError> {
    // Escape unwanted characters
    let mut inputs = node.get_input().to_vec();
    let mut outputs = node.get_output().to_vec();

    let input_shape = inputs
        .iter()
        .map(|input| match shape_infos.get(input.as_str()) {
            Some(info) => Ok(info),
            None => Err(CompileError::DimensionsMissing(
                input.to_string(),
                node.get_name().to_string(),
            )),
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let output_shape = outputs
        .iter()
        .map(|output| match shape_infos.get(output.as_str()) {
            Some(info) => Ok(info),
            None => Err(CompileError::DimensionsMissing(
                output.to_string(),
                node.get_name().to_string(),
            )),
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let input_lengths = input_shape
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    let output_lengths = output_shape
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    // Generate variable names from the input names (which may contain special characters we don't want)
    inputs = inputs
        .iter()
        .map(|input| to_wgsl_variable_name(input))
        .collect::<Vec<_>>();

    outputs = outputs
        .iter()
        .map(|output| to_wgsl_variable_name(output))
        .collect::<Vec<_>>();

    let input_chunks: Vec<Vec<u64>> = input_shape.iter().map(|d| d.chunks()).collect();
    let output_chunks: Vec<Vec<u64>> = output_shape.iter().map(|d| d.chunks()).collect();

    let mut context = Context::new();
    context.insert("inputs", &inputs);
    context.insert("outputs", &outputs);
    context.insert("i_lens", &input_lengths);
    context.insert("o_lens", &output_lengths);
    context.insert("i_shape", &input_shape);
    context.insert("o_shape", &output_shape);
    context.insert("i_chunks", &input_chunks);
    context.insert("o_chunks", &output_chunks);
    context.insert("op_type", &node.get_op_type());
    context.insert("opset_version", &opset_version);

    let (template, x, y, z) = match node.get_op_type() {
        // Map simple function
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" => (
            "endomorphism/map.wgsl".to_string(),
            ceil(output_lengths[0], 4) as _,
            1,
            1,
        ),
        // Copy data
        "Reshape" | "Dropout" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" => (
            "endomorphism/copy.wgsl".to_string(),
            ceil(output_lengths[0], 16) as _,
            1,
            1,
        ),

        "Softmax" => {
            let default_axis = match opset_version {
                1..=10 => 1,   // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-1
                11..=12 => 1, // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-11
                13..=15 => -1, // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-13
                _ => return Err(CompileError::UnsupportedOpsetVersion(opset_version)),
            };

            /* Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely
            describes the batch_size. From version 13 onwards, counting backwards is also allowed. We only support the
            variant with [1,n] input tensors, where axis is 1 or -1 */
            let mut axis = get_attribute("axis", Some(default_axis), node)?;
            if axis < 0 {
                if opset_version >= 13 {
                    axis += input_shape[0].rank() as i64;
                } else {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "axis".to_string(),
                        value: format!("{}", axis),
                        opset_version,
                    });
                }
            }

            if axis >= (input_shape[0].rank() as i64) {
                return Err(CompileError::InvalidAttributeValue {
                    attribute: "axis".to_string(),
                    value: format!("{}", axis),
                    opset_version,
                });
            }

            if axis != 1 {
                return Err(CompileError::UnimplementedVariant {
                    variant: format!(
                        "softmax on an axis ({}) other than the second with [1,n] inputs",
                        axis,
                    ),
                    op: "Softmax".to_string(),
                });
            }

            ("endomorphism/softmax.wgsl".to_string(), 1, 1, 1)
        }

        // Arithmetic operation
        "Add" | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
        | "Mod" | "Mul" | "Or" | "Sub" => {
            let coefficient = get_attribute("coefficient", Some(1.0), node)?;
            context.insert("coefficient", &coefficient);
            context.insert(
                "op_type",
                match node.get_op_type() {
                    "Add" => "+",
                    "And" => "&",
                    "Div" => "/",
                    "Equal" => "==",
                    "Greater" => ">",
                    "GreaterOrEqual" => ">=",
                    "Less" => "<",
                    "LessOrEqual" => "<=",
                    "Mod" => "%",
                    "Mul" => "*",
                    "Or" => "|",
                    "Sub" => "-",
                    _ => {
                        return Err(CompileError::UnimplementedOp(
                            node.get_op_type().to_string(),
                        ))
                    }
                },
            );
            (
                "endomorphism/arithmetic.wgsl".to_string(),
                ceil(output_lengths[0], 1024) as _,
                1,
                1,
            )
        }
        // Not taking into account attributes
        "BatchNormalization" => {
            /* Prior to version 9, BatchNormalization supported a 'spatial' mode where input mean/variance are of shape
            [C,W,H] instead of just [C]. See https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-7.
            This mode is not supported. */
            if let Ok(spatial_value) = get_attribute::<i64>("spatial", None, node) {
                if opset_version < 9 {
                    return Err(CompileError::UnimplementedVariant {
                        op: "BatchNormalization".to_string(),
                        variant: "spatial".to_string(),
                    });
                } else {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "spatial".to_string(),
                        opset_version,
                        value: spatial_value.to_string(),
                    });
                }
            }

            // [N,C,w,h] => [N,C,w,h] where [w,h] is normalized using stats for each [N,C]
            // N and C are optional and assumed to be one for lower-rank inputs
            if input_shape[0].rank() <= 2 || input_shape[0].rank() > 4 {
                return Err(CompileError::UnimplementedVariant {
                    op: "BatchNormalization".to_string(),
                    variant: format!("with input {}", input_shape[0]),
                });
            }

            let (input_batches, input_channels, input_w, input_h) = match input_shape[0].rank() {
                2 => (1, 1, input_shape[0].dim(0), input_shape[0].dim(1)), // WxH, C=1, N=1
                3 => (
                    1,
                    input_shape[0].dim(0),
                    input_shape[0].dim(1),
                    input_shape[0].dim(2),
                ), // CxWxH, single batch N=1
                4 => (
                    input_shape[0].dim(0),
                    input_shape[0].dim(1),
                    input_shape[0].dim(2),
                    input_shape[0].dim(3),
                ), // NxCxWxH
                _ => unreachable!(),
            };

            if input_batches == 0 || input_channels == 0 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_shape[0].clone(),
                });
            }

            // If w*h is a multiple of 4, we can use vec4 in our shader
            let elem_type = DataType::for_size((input_w * input_h) as usize);

            context.insert("elem_type", &elem_type.wgsl_type_name());
            context.insert("elem_stride", &elem_type.size_bytes());

            // The default for epsilon is 1e05, see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#attributes-252
            let epsilon = get_attribute("epsilon", Some(1e-05), node)?;
            context.insert("epsilon", &epsilon);
            context.insert(
                "batch_size",
                &ceil(
                    input_channels * input_w * input_h,
                    elem_type.elements() as u64,
                ),
            );
            context.insert(
                "channel_size",
                &ceil(input_w * input_h, elem_type.elements() as u64),
            );

            (
                "endomorphism/batchnormalization.wgsl".to_string(),
                ceil(input_w * input_h, elem_type.elements() as u64) as _,
                input_channels as _,
                input_batches as _,
            )
        }
        "Relu" | "Sigmoid" | "Softsign" | "Softplus" | "Clip" | "Celu" | "Elu" | "LeakyRelu" => {
            let alpha = get_attribute("alpha", Some(1.0), node)?;
            context.insert("alpha", &alpha);
            (
                "endomorphism/activation.wgsl".to_string(),
                ceil(output_lengths[0], 4) as _,
                1,
                1,
            )
        }
        "Concat" => {
            let mut input_cumulative_len = vec![];
            let mut sum = 0;
            for len in input_lengths.iter() {
                sum += len;
                input_cumulative_len.push(sum);
            }
            context.insert("cum_len", &input_cumulative_len);
            (
                "matrix/concat.wgsl".to_string(),
                ceil(output_lengths[0], 256) as u32,
                1,
                1,
            )
        }
        op
        @
        ("MaxPool" | "AveragePool" | "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish"
        | "GlobalAveragePool") => {
            // TODO: Conv only support NxCxHxW for the moment.
            debug_assert!(input_shape[0].rank() == 4);

            // GlobalAveragePool is equivalent to AveragePool, with the kernel shape set to the size of the input tensor
            // See https://github.com/onnx/onnx/blob/main/docs/Operators.md#globalaveragepool
            // Other attributes are not supported and also not relevant, and are simply ignored
            let is_global_average_pool = op == "GlobalAveragePool";
            if is_global_average_pool {
                // Generate shader code as if this were a regular AveragePool
                context.insert("op_type", "AveragePool");
            }

            let auto_pad = get_attribute("auto_pad", Some("NOTSET".to_string()), node)?;
            let dilations = get_attribute("dilations", Some(vec![1, 1]), node)?;
            let kernel_shape = if is_global_average_pool {
                vec![input_shape[0].dim(2) as i64, input_shape[0].dim(3) as i64]
            } else {
                get_attribute::<Vec<i64>>("kernel_shape", None, node)?
            };
            let strides = get_attribute("strides", Some(vec![1, 1]), node)?;
            let pads = get_attribute("pads", Some(vec![0, 0, 0, 0]), node)?;

            let pads = match auto_pad.as_str() {
                "NOTSET" => pads.to_vec(),
                "SAME_UPPER" => {
                    let slack_0 = -strides[0] + ((kernel_shape[0] - 1) * dilations[0] + 1);
                    let slack_0_div_2 = slack_0 / 2;
                    let slack_rest_0 = slack_0 % 2;
                    let slack_1 = -strides[1] + ((kernel_shape[1] - 1) * dilations[1] + 1);
                    let slack_1_div_2 = slack_1 / 2;
                    let slack_rest_1 = slack_1 % 2;
                    vec![
                        slack_0_div_2,
                        slack_1_div_2,
                        slack_0_div_2 + slack_rest_0,
                        slack_1_div_2 + slack_rest_1,
                    ]
                }
                "SAME_LOWER" => {
                    let slack_0 = -strides[0] + ((kernel_shape[0] - 1) * dilations[0] + 1);
                    let slack_0_div_2 = slack_0 / 2;
                    let slack_rest_0 = slack_0 % 2;
                    let slack_1 = -strides[1] + ((kernel_shape[1] - 1) * dilations[1] + 1);
                    let slack_1_div_2 = slack_1 / 2;
                    let slack_rest_1 = slack_1 % 2;
                    vec![
                        slack_0_div_2 + slack_rest_0,
                        slack_1_div_2 + slack_rest_1,
                        slack_0_div_2,
                        slack_1_div_2,
                    ]
                }
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: op.to_string(),
                        variant: format!("auto_pad={}", auto_pad),
                    })
                }
            };

            let input_shape = input_shape[0];
            let output_shape = output_shape[0];
            assert!(kernel_shape.len() >= 2);
            assert!(kernel_shape[0] >= 0 && kernel_shape[1] >= 0);

            context.insert("original_width", &input_shape.dim(3));
            context.insert("width", &output_shape.dim(3));
            context.insert("original_height", &input_shape.dim(2));
            context.insert("channel", &input_shape.dim(1));
            context.insert("stride", &strides);
            context.insert("kernel_shape", &kernel_shape);
            context.insert("kernel_len", &(kernel_shape[0] * kernel_shape[1]));
            context.insert(
                "kernel_channel_len",
                &((kernel_shape[0] as u64) * (kernel_shape[1] as u64) * input_shape.dim(1)),
            );
            context.insert("pad", &pads);
            context.insert("dilation", &dilations);

            // GLSL shader for convolution computation
            match op {
                "MaxPool" | "AveragePool" | "GlobalAveragePool" => (
                    "pool/aggregate.wgsl".to_string(),
                    ceil(output_lengths[0], 1024) as _,
                    1,
                    1,
                ),
                "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish" => {
                    // Alpha is the Leaky Relu attribute
                    let alpha = get_attribute("alpha", Some(0.01), node)?;
                    context.insert("alpha", &alpha);

                    // WGSL shader for convolution computation
                    if (strides == [1, 1])
                        && (kernel_shape == [1, 1])
                        && (dilations == [1, 1] && (pads == [0, 0, 0, 0]))
                        && (input_shape.dim(1) % 16 == 0)
                        && (output_shape.dim(1) % 4 == 0)
                    {
                        (
                            "pool/conv_kernel_1.wgsl".to_string(),
                            ceil(output_lengths[0], 1024) as _,
                            1,
                            1,
                        )
                    } else if (strides == [1, 1])
                        && (kernel_shape == [3, 3])
                        && (dilations == [1, 1])
                        && (output_shape.dim(1) % 4 == 0)
                    {
                        (
                            "pool/conv_kernel_3.wgsl".to_string(),
                            ceil(output_lengths[0], 1024) as _,
                            1,
                            1,
                        )
                    } else {
                        (
                            "pool/conv.wgsl".to_string(),
                            ceil(output_lengths[0], 256) as _,
                            1,
                            1,
                        )
                    }
                }
                _ => return Err(CompileError::InvalidOperation(op.to_string())),
            }
        }
        "Gemm" | "MatMul" => {
            let alpha = get_attribute("alpha", Some(1.0), node)?;
            let beta = get_attribute("beta", Some(1.0), node)?;
            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            if input_shape[0].dim(0) == 1 {
                let threads = output_shape[0].dim(1);
                ("matrix/gemm_1.wgsl".to_string(), threads as _, 1, 1)
            } else {
                let threads = input_shape[0].dim(0) * input_shape[1].dim(1) / 16;
                ("matrix/gemm.wgsl".to_string(), threads as _, 1, 1)
            }
        }
        "Resize" => {
            let coordinate_transformation_mode = get_attribute(
                "coordinate_transformation_mode",
                Some("half_pixel".to_string()),
                node,
            )?;
            context.insert(
                "coordinate_transformation_mode",
                &coordinate_transformation_mode,
            );

            match coordinate_transformation_mode.as_str() {
                "half_pixel" => {}
                "pytorch_half_pixel" => {}
                "align_corners" => {}
                "asymmetric" => {}
                "tf_crop_and_resize" => {
                    let roi = get_attribute::<Vec<i64>>("roi", None, node)?;
                    let extrapolation_value =
                        get_attribute("extrapolation_value", Some(0.0), node)?;
                    context.insert("roi", &roi);
                    context.insert("extrapolation_value", &extrapolation_value);
                }
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: "Resize".to_string(),
                        variant: format!(
                            "coordinate_transformation_mode={}",
                            coordinate_transformation_mode
                        ),
                    })
                }
            }

            let scales = get_attribute::<Vec<f32>>("scales", Some(vec![]), node)?;
            let scale_prints = if scales.is_empty() {
                let sizes = get_attribute::<Vec<i64>>("sizes", Some(vec![]), node)?;
                sizes
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let tmp = *x as f32 / input_shape[0].dim(i) as f32;
                        format!("{:.2}", tmp)
                    })
                    .collect::<Vec<_>>()
            } else {
                scales.iter().map(|x| format!("{:.2}", x)).collect()
            };

            let mode = get_attribute("mode", Some("nearest".to_string()), node)?;
            context.insert("mode", &mode);
            context.insert("scales", &scale_prints);

            match mode.as_str() {
                "nearest" => {
                    let nearest_mode = get_attribute(
                        "nearest_mode",
                        Some("round_prefer_floor".to_string()),
                        node,
                    )?;
                    match nearest_mode.as_str() {
                        "floor" => {}
                        _ => {
                            return Err(CompileError::UnimplementedVariant {
                                op: "Resize".to_string(),
                                variant: format!("nearest_mode={}", nearest_mode.to_string()),
                            })
                        }
                    }
                }
                "cubic" => {
                    let cubic_coeff_a = get_attribute("cubic_coeff_a", Some(-0.75), node)?;
                    context.insert("cubic_coeff_a", &cubic_coeff_a);
                    return Err(CompileError::UnimplementedVariant {
                        op: String::from("Resize"),
                        variant: format!("mode={}", mode),
                    });
                }
                /* "linear" | */
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: String::from("Resize"),
                        variant: format!("mode={}", mode),
                    });
                }
            };

            let exclude_outside = get_attribute("exclude_outside", Some(0), node)?;
            context.insert("exclude_outside", &exclude_outside);

            (
                "matrix/resize.wgsl".to_string(),
                ceil(output_lengths[0], 256) as u32,
                1,
                1,
            )
        }
        "Sum" => return Err(CompileError::UnimplementedOp(String::from("Sum"))),
        "Split" => {
            let mut axis = get_attribute("axis", Some(0), node)?;
            if axis < 0 {
                axis += input_shape[0].element_count() as i64
            }
            context.insert("axis", &axis);

            let split_chunk = input_shape[0].dim(axis as usize) as usize / outputs.len();
            let default_split = (1..=outputs.len())
                .map(|x| (x * split_chunk) as _)
                .collect();

            let split = get_attribute::<Vec<i64>>("split", Some(default_split), node)?;
            context.insert("split", &split);

            (
                "matrix/split.wgsl".to_string(),
                ceil(output_lengths[0], 256) as u32,
                1,
                1,
            )
        }
        "Transpose" => {
            let default = ((input_lengths[0] as i64)..0).collect::<Vec<_>>();
            let perms: Vec<i64> = get_attribute("perm", Some(default), node)?;
            let permuted_shapes = perms
                .iter()
                .map(|p| output_shape[0].dim(*p as usize))
                .collect::<Vec<_>>();

            let mut chunks = vec![];
            for i in 1..permuted_shapes.len() {
                chunks.push(permuted_shapes[i..].iter().product::<u64>());
            }
            chunks.push(1);

            context.insert("permuted_chunks", &chunks);

            (
                "matrix/transpose.wgsl".to_string(),
                ceil(output_lengths[0], 256) as _,
                1,
                1,
            )
        }
        op => return Err(CompileError::UnimplementedOp(op.to_string())),
    };

    let shader = TEMPLATES
        .render(&template, &context)
        .expect("failed to render shader");

    debug_assert!(
        x < 16352,
        "Node {} exceeds max compute by {}",
        node.get_name(),
        x - 16352
    );
    Ok(CompiledNode {
        shader,
        threads: (x, y, z),
    })
}
