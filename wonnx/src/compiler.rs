use crate::utils::{ceil, get_attribute, AttributeNotFoundError, DataType, Shape};
use tera::{Context, Tera};
use thiserror::Error;

/// The maximum number of threads that can be spawned in each dimension, according to the WebGPU specification. See
// https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsperdimension
pub const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// The maximum workgroup size per dimension (see https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsizex)
pub const MAX_WORKGROUP_SIZE_X: u32 = 256;
pub const MAX_WORKGROUP_SIZE_Y: u32 = 256;
pub const MAX_WORKGROUP_SIZE_Z: u32 = 64;

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

    #[error("the model exceeds the limit for {0}: {1} > {2}")]
    ComputeLimitExceeded(String, u32, u32),
}

pub fn compile(
    node: &crate::onnx::NodeProto,
    input_shapes: &[&Shape],
    output_shapes: &[&Shape],
    opset_version: i64,
) -> Result<CompiledNode, CompileError> {
    let input_lengths = input_shapes
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    let output_lengths = output_shapes
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    let input_chunks: Vec<Vec<u64>> = input_shapes.iter().map(|d| d.chunks()).collect();
    let output_chunks: Vec<Vec<u64>> = output_shapes.iter().map(|d| d.chunks()).collect();

    let mut context = Context::new();
    context.insert("i_lens", &input_lengths);
    context.insert("o_lens", &output_lengths);
    context.insert("i_shape", &input_shapes);
    context.insert("o_shape", &output_shapes);
    context.insert("i_chunks", &input_chunks);
    context.insert("o_chunks", &output_chunks);
    context.insert("op_type", &node.get_op_type());
    context.insert("opset_version", &opset_version);

    context.insert("scalar_type", "f32");
    context.insert("scalar_stride", &4);
    context.insert("vec4_stride", &(4 * 4));
    context.insert("mat4x4_stride", &(4 * 4 * 4));
    context.insert("mat3x3_stride", &(48));

    let (template, x, y, z) = match node.get_op_type() {
        op @ ("Reshape" | "Dropout" | "Identity" | "Flatten" | "Squeeze" | "Unsqueeze") => {
            // These ops should all be optimized away earlier
            return Err(CompileError::InvalidOperation(op.to_string()));
        }

        // Map simple function
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" => {
            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 4),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);
            ("endomorphism/map.wgsl".to_string(), x_threads, 1, 1)
        }

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
                    axis += input_shapes[0].rank() as i64;
                } else {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "axis".to_string(),
                        value: format!("{}", axis),
                        opset_version,
                    });
                }
            }

            if axis >= (input_shapes[0].rank() as i64) {
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

            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 1024) as _,
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);

            ("endomorphism/arithmetic.wgsl".to_string(), x_threads, 1, 1)
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
            if input_shapes[0].rank() <= 2 || input_shapes[0].rank() > 4 {
                return Err(CompileError::UnimplementedVariant {
                    op: "BatchNormalization".to_string(),
                    variant: format!("with input {}", input_shapes[0]),
                });
            }

            let (input_batches, input_channels, input_w, input_h) = match input_shapes[0].rank() {
                2 => (1, 1, input_shapes[0].dim(0), input_shapes[0].dim(1)), // WxH, C=1, N=1
                3 => (
                    1,
                    input_shapes[0].dim(0),
                    input_shapes[0].dim(1),
                    input_shapes[0].dim(2),
                ), // CxWxH, single batch N=1
                4 => (
                    input_shapes[0].dim(0),
                    input_shapes[0].dim(1),
                    input_shapes[0].dim(2),
                    input_shapes[0].dim(3),
                ), // NxCxWxH
                _ => unreachable!(),
            };

            if input_batches == 0 || input_channels == 0 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_shapes[0].clone(),
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

            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 4),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;

            context.insert("workgroup_size_x", &workgroup_size_x);

            (
                "endomorphism/activation.wgsl".to_string(),
                x_threads as _,
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
        op @ ("MaxPool" | "AveragePool" | "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish"
        | "GlobalAveragePool") => {
            // TODO: Conv only support NxCxHxW for the moment.
            debug_assert!(input_shapes[0].rank() == 4);

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
                vec![input_shapes[0].dim(2) as i64, input_shapes[0].dim(3) as i64]
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

            let input_shape = &input_shapes[0];
            let output_shape = &output_shapes[0];
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
        op @ ("Gemm" | "MatMul") => {
            let alpha = get_attribute("alpha", Some(1.0), node)?;
            let beta = get_attribute("beta", Some(1.0), node)?;

            // Whether A resp. B should be transposed, or C should be broadcast (default: 0 = false)
            if op == "Gemm" {
                let transpose_a = get_attribute("transA", Some(0), node)?;
                let transpose_b = get_attribute("transB", Some(0), node)?;
                let broadcast = get_attribute("broadcast", Some(0), node)?;

                if transpose_a != 0 || transpose_b != 0 || broadcast != 0 {
                    return Err(CompileError::UnimplementedVariant {
                        variant: "Gemm with transA/transB/broadcast not equal to zero".to_string(),
                        op: op.to_string(),
                    });
                }
            }

            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            // Whether A resp. B should be transposed, or C should be broadcast (default: 0 = false)
            if op == "Gemm" {
                let transpose_a = get_attribute("transA", Some(0), node)?;
                let transpose_b = get_attribute("transB", Some(0), node)?;
                let broadcast = get_attribute("broadcast", Some(0), node)?;

                if transpose_a != 0 || transpose_b != 0 || broadcast != 0 {
                    return Err(CompileError::UnimplementedVariant {
                        variant: "Gemm with transA/transB/broadcast not equal to zero".to_string(),
                        op: op.to_string(),
                    });
                }
            }

            if input_shapes[0].dim(0) == 1 {
                let threads = output_shapes[0].dim(1);
                ("matrix/gemm_1.wgsl".to_string(), threads as _, 1, 1)
            } else {
                let threads = input_shapes[0].dim(0) * input_shapes[1].dim(1) / 16;
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
                        let tmp = *x as f32 / input_shapes[0].dim(i) as f32;
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
                                variant: format!("nearest_mode={}", nearest_mode),
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
                axis += input_shapes[0].element_count() as i64
            }
            context.insert("axis", &axis);

            let split_chunk = input_shapes[0].dim(axis as usize) as usize / output_shapes.len();
            let default_split = (1..=output_shapes.len())
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
                .map(|p| output_shapes[0].dim(*p as usize))
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

    // Check if we remain within the limits of the thread count allowed by WebGPU
    if x > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("X threads"),
            x as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }
    if y > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("Y threads"),
            y as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }
    if z > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("Z threads"),
            z as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }

    Ok(CompiledNode {
        shader,
        threads: (x, y, z),
    })
}

/// Determines the appropriate number of threads and workgroup size given a number of times the entry point of the shader should be run
fn workgroup_size(
    x: u64,
    max_threads: u32,
    max_workgroup_size: u32,
) -> Result<(u32, u32), CompileError> {
    let max_x = max_threads as u64;

    Ok(if x > max_x {
        let workgroup_size = ceil(x, max_x) as _;
        let threads = ceil(x, workgroup_size as u64) as _;
        log::info!(
            "WGS: {} > {}, so workgroup size={} x threads={}",
            x,
            max_x,
            workgroup_size,
            threads
        );

        if threads > max_threads {
            return Err(CompileError::ComputeLimitExceeded(
                String::from("threads"),
                threads,
                max_threads,
            ));
        }

        if workgroup_size > max_workgroup_size {
            return Err(CompileError::ComputeLimitExceeded(
                String::from("workgroup size"),
                workgroup_size,
                max_workgroup_size,
            ));
        }

        log::info!(
            "adjusting workgroup size to {}, threads to {} (was {})",
            workgroup_size,
            threads,
            x
        );
        (threads, workgroup_size)
    } else {
        (x as u32, 1)
    })
}
