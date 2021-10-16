use crate::get_attribute;
use crate::onnx;
use std::collections::HashMap;
use std::str::from_utf8;
use tera::Context;

pub fn format_node(
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
    context: &mut Context,
) -> (String, u32, u32, u32) {
    let inputs = node.get_input();
    let outputs = node.get_output();

    context.insert("input", &inputs);
    context.insert("output", &outputs);
    context.insert("op_type", &node.get_op_type().to_lowercase());

    let input_dims = &inner_infos.get(&inputs[0]).unwrap().dims;
    let output_dims = &inner_infos.get(&outputs[0]).unwrap().dims;

    let length = crate::utils::len(input_dims);

    match node.get_op_type() {
        // Map simple function
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" => {
            ("endomorphism/map.wgsl".to_string(), (length / 4) as _, 1, 1)
        }
        // Copy data
        "Reshape" | "Dropout" | "Flatten" | "Squeeze" | "Softmax" => (
            "endomorphism/copy.wgsl".to_string(),
            (length / 4) as _,
            1,
            1,
        ),
        // Arithmetic operation
        "Add" | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
        | "Mod" | "Mul" | "Or" | "Sub" => {
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
                    _ => unimplemented!(),
                },
            );
            (
                "endomorphism/arithmetic.wgsl".to_string(),
                (length / 4) as _,
                1,
                1,
            )
        }
        // Not taking into account attributes
        "BatchNormalization" => {
            let mut epsilon_default = onnx::AttributeProto::new();
            epsilon_default.set_f(1.0);

            let epsilon = get_attribute("epsilon", Some(&epsilon_default), node).get_f();
            context.insert("epsilon", &epsilon);

            todo!();

            (
                "endomorphism/batchnormalization.wgsl".to_string(),
                (length / 4) as _,
                1,
                1,
            )
        }
        "Celu" | "Elu" => {
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();
            context.insert("alpha", &alpha);
            (
                "endomorphism/activation.wgsl".to_string(),
                (length / 4) as _,
                1,
                1,
            )
        }
        "Concat" => {
            context.insert(
                "len_0",
                &(input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]),
            );
            (
                "matrix/concat.wgsl".to_string(),
                output_dims.iter().product::<i64>() as u32,
                1,
                1,
            )
        }
        "Conv" | "MaxPool" | "AveragePool" | "ConvRelu" => {
            // TODO: Conv only support NxCxHxW for the moment.
            debug_assert!(input_dims.len() == 4usize);

            let mut auto_pad_default = onnx::AttributeProto::new();
            auto_pad_default.set_s("NOTSET".to_string().into_bytes());

            let auto_pad =
                from_utf8(get_attribute("auto_pad", Some(&auto_pad_default), node).get_s())
                    .unwrap();

            let mut dilations_default = onnx::AttributeProto::new();
            dilations_default.set_ints(vec![1, 1]);

            let dilations = get_attribute("dilations", Some(&dilations_default), node).get_ints();

            let kernel_shape = get_attribute("kernel_shape", None, node).get_ints();

            let mut strides_default = onnx::AttributeProto::new();
            strides_default.set_ints(vec![1, 1]);

            let strides = get_attribute("strides", Some(&strides_default), node).get_ints();

            let mut pads_default = onnx::AttributeProto::new();
            pads_default.set_ints(vec![0, 0, 0, 0]);

            let pads = get_attribute("pads", Some(&pads_default), node).get_ints();

            let pads = match auto_pad {
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
                _ => unimplemented!(),
            };

            context.insert(
                "M_x_H_x_W",
                &(output_dims[1] * output_dims[2] * output_dims[3]),
            );
            context.insert("H_x_W", &(output_dims[2] * output_dims[3]));
            context.insert(
                "original_C_x_H_x_W",
                &(input_dims[1] * input_dims[2] * input_dims[3]),
            );
            context.insert("original_H_x_W", &(input_dims[2] * input_dims[3]));
            context.insert("original_width", &input_dims[3]);
            context.insert("width", &output_dims[3]);
            context.insert("original_height", &input_dims[2]);
            context.insert("channel", &input_dims[1]);
            context.insert("stride", strides);
            context.insert("kernel_shape", kernel_shape);
            context.insert("kernel_len", &(kernel_shape[0] * kernel_shape[1]));
            context.insert(
                "kernel_channel_len",
                &(kernel_shape[0] * kernel_shape[1] * input_dims[1]),
            );
            context.insert("pad", &pads);
            context.insert("dilation", &dilations);

            if node.get_op_type() == "ConvRelu" {
                context.insert("conv_relu", &true);
            }
            // GLSL shader for convolution computation
            (
                "pool/conv.wgsl".to_string(),
                (output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]) as _,
                1,
                1,
            )
        }
        "Gemm" | "MatMul" => {
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();

            let mut beta_default = onnx::AttributeProto::new();
            beta_default.set_f(1.0);

            let beta = get_attribute("beta", Some(&beta_default), node).get_f();

            let left_columns = &input_dims[1];
            let right_columns = &inner_infos.get(&inputs[1]).unwrap().dims[1];

            context.insert("left_columns", &left_columns);

            context.insert("right_columns", &right_columns);
            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            if input_dims[0] == 1 {
                let threads = output_dims[1];
                ("matrix/gemm_1.wgsl".to_string(), threads as _, 1, 1)
            } else {
                let threads = (&input_dims[0] / 4) * right_columns / 4;
                ("matrix/gemm.wgsl".to_string(), threads as _, 1, 1)
            }
        }
        "Relu" | "Sigmoid" | "Softsign" | "Softplus" | "Clip" => {
            ("endomorphism/activation.wgsl".to_string(), 1, 1, 1)
        }
        "Sum" => {
            unimplemented!()
        }
        "Transpose" => {
            let len_0 = input_dims[0];
            let len_1 = input_dims[1] / 4;

            let perm = get_attribute("perm", None, node).get_ints();
            context.insert("len_1", &len_1);
            context.insert("len_0", &len_0);
            ("matrix/transpose.wgsl".to_string(), (length / 4) as _, 1, 1)
        }
        _ => unimplemented!(),
    }
}
