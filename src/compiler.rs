use crate::utils::{ceil, get_attribute, AttributeNotFoundError};
use std::collections::HashMap;
use tera::{Context, Tera};
use thiserror::Error;

// Escaping special characters as well as adding `var_` in the beginning of the variable name to avoid collisions with wgsl syntax.
// FIXME: this has the potential to cause collisions (i.e. "a/b" and "a.b" will both translate to "ab" and hence cannot be used simultaneously)
fn to_wgsl_variable_name(input_or_output_name: &str) -> String {
    String::from("var_")
        + &input_or_output_name.replace(&['(', ')', ',', '\"', '.', ';', ':', '\'', '/'][..], "")
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

    #[error("the variant '{1}' is not yet implemented for op {0}")]
    UnimplementedVariant(String, String),
}

pub fn compile(
    node: &crate::onnx::NodeProto,
    dims_infos: &HashMap<String, Vec<i64>>,
    tera: &Tera,
) -> Result<CompiledNode, CompileError> {
    // Escape unwanted characters
    let mut inputs = node.get_input().to_vec();
    let mut outputs = node.get_output().to_vec();

    let input_dims = inputs
        .iter()
        .map(|input| match dims_infos.get(input.as_str()) {
            Some(info) => Ok(info),
            None => Err(CompileError::DimensionsMissing(
                input.to_string(),
                node.get_name().to_string(),
            )),
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let output_dims = outputs
        .iter()
        .map(|output| match dims_infos.get(output.as_str()) {
            Some(info) => Ok(info),
            None => Err(CompileError::DimensionsMissing(
                output.to_string(),
                node.get_name().to_string(),
            )),
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let input_lengths = input_dims
        .iter()
        .map(|dims| dims.iter().product())
        .collect::<Vec<i64>>();

    let output_lengths = output_dims
        .iter()
        .map(|dims| dims.iter().product())
        .collect::<Vec<i64>>();

    // Generate variable names from the input names (which may contain special characters we don't want)
    inputs = inputs
        .iter()
        .map(|input| to_wgsl_variable_name(input))
        .collect::<Vec<_>>();

    outputs = outputs
        .iter()
        .map(|output| to_wgsl_variable_name(output))
        .collect::<Vec<_>>();

    let mut input_chunks = vec![];
    for dims in input_dims.iter() {
        let mut chunk = vec![];
        for i in 1..dims.len() {
            chunk.push(dims[i..].iter().product::<i64>());
        }
        chunk.push(1);
        input_chunks.push(chunk);
    }

    let mut output_chunks = vec![];
    for dims in output_dims.iter() {
        let mut chunk = vec![];
        for i in 1..dims.len() {
            chunk.push(dims[i..].iter().product::<i64>());
        }
        chunk.push(1);
        output_chunks.push(chunk);
    }

    let mut context = Context::new();
    context.insert("inputs", &inputs);
    context.insert("outputs", &outputs);
    context.insert("i_lens", &input_lengths);
    context.insert("o_lens", &output_lengths);
    context.insert("i_dims", &input_dims);
    context.insert("o_dims", &output_dims);
    context.insert("i_chunks", &input_chunks);
    context.insert("o_chunks", &output_chunks);
    context.insert("op_type", &node.get_op_type());

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
        "Reshape" | "Dropout" | "Flatten" | "Squeeze" => (
            "endomorphism/copy.wgsl".to_string(),
            ceil(output_lengths[0], 16) as _,
            1,
            1,
        ),
        "Softmax" => ("endomorphism/softmax.wgsl".to_string(), 1, 1, 1),
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
            let epsilon = get_attribute("epsilon", Some(1.0), node)?;
            context.insert("epsilon", &epsilon);

            return Err(CompileError::UnimplementedOp(
                node.get_op_type().to_string(),
            ));

            //   (
            //       "endomorphism/batchnormalization.wgsl".to_string(),
            //       (length / 4) as _,
            //       1,
            //       1,
            //   )
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
        op @ ("MaxPool" | "AveragePool" | "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish") => {
            // TODO: Conv only support NxCxHxW for the moment.
            debug_assert!(input_dims[0].len() == 4usize);

            let auto_pad = get_attribute("auto_pad", Some("NOTSET".to_string()), node)?;
            let dilations = get_attribute("dilations", Some(vec![1, 1]), node)?;
            let kernel_shape = get_attribute::<Vec<i64>>("kernel_shape", None, node)?;
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
                _ => return Err(CompileError::UnimplementedVariant(op.to_string(), auto_pad)),
            };

            let input_dims = input_dims[0];
            let output_dims = output_dims[0];

            context.insert("original_width", &input_dims[3]);
            context.insert("width", &output_dims[3]);
            context.insert("original_height", &input_dims[2]);
            context.insert("channel", &input_dims[1]);
            context.insert("stride", &strides);
            context.insert("kernel_shape", &kernel_shape);
            context.insert("kernel_len", &(kernel_shape[0] * kernel_shape[1]));
            context.insert(
                "kernel_channel_len",
                &(kernel_shape[0] * kernel_shape[1] * input_dims[1]),
            );
            context.insert("pad", &pads);
            context.insert("dilation", &dilations);

            // GLSL shader for convolution computation
            match op {
                "MaxPool" | "AveragePool" => (
                    "pool/aggregate.wgsl".to_string(),
                    ceil(output_lengths[0], 1024) as _,
                    1,
                    1,
                ),
                "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish" => {
                    // Alpha is the Leaky Relu attribute
                    let alpha = get_attribute("alpha", Some(0.01), node)?;
                    context.insert("alpha", &alpha);

                    // GLSL shader for convolution computation
                    if (strides == [1, 1])
                        && (kernel_shape == [1, 1])
                        && (dilations == [1, 1] && (pads == [0, 0, 0, 0]))
                        && (input_dims[1] % 16 == 0)
                        && (output_dims[1] % 4 == 0)
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
                        && (output_dims[1] % 4 == 0)
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

            if input_dims[0][0] == 1 {
                let threads = output_dims[0][1];
                ("matrix/gemm_1.wgsl".to_string(), threads as _, 1, 1)
            } else {
                let threads = input_dims[0][0] * input_dims[1][1] / 16;
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
                    return Err(CompileError::UnimplementedVariant(
                        "Resize".to_string(),
                        coordinate_transformation_mode,
                    ))
                }
            }

            let scales = get_attribute::<Vec<f32>>("scales", Some(vec![]), node)?;
            let scale_prints = if scales.is_empty() {
                let sizes = get_attribute::<Vec<i64>>("sizes", Some(vec![]), node)?;
                sizes
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let tmp = *x as f32 / input_dims[0][i] as f32;
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
                            return Err(CompileError::UnimplementedVariant(
                                "Resize".to_string(),
                                nearest_mode.to_string(),
                            ))
                        }
                    }
                }
                "linear" => {
                    return Err(CompileError::UnimplementedVariant(
                        String::from("Resize"),
                        mode,
                    ));
                }
                "cubic" => {
                    let cubic_coeff_a = get_attribute("cubic_coeff_a", Some(-0.75), node)?;
                    context.insert("cubic_coeff_a", &cubic_coeff_a);
                    return Err(CompileError::UnimplementedVariant(
                        String::from("Resize"),
                        String::from("cubic"),
                    ));
                }
                _ => {
                    return Err(CompileError::UnimplementedVariant(
                        String::from("Resize"),
                        mode,
                    ));
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
                axis += input_dims[0].len() as i64
            }
            context.insert("axis", &axis);

            let split_chunk = input_dims[0][axis as usize] as usize / outputs.len();
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
            let default = (input_lengths[0]..0).collect::<Vec<_>>();
            let perms = get_attribute("perm", Some(default), node)?;
            let permuted_dims = perms
                .iter()
                .map(|p| output_dims[0][*p as usize])
                .collect::<Vec<_>>();

            let mut chunks = vec![];
            for i in 1..permuted_dims.len() {
                chunks.push(permuted_dims[i..].iter().product::<i64>());
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

    let shader = tera
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
