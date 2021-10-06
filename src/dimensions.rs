use crate::get_attribute;
use crate::onnx;
use crate::resource;
use crate::InnerInfo;
use std::collections::HashMap;
use std::str::from_utf8;

fn get_dimension(inputs_onnx: &[onnx::ValueInfoProto], input_name: &str) -> Vec<i64> {
    inputs_onnx
        .iter()
        .find(|x| x.get_name() == input_name)
        .unwrap_or_else(|| panic!("Dimensions for input: {} was not found", input_name))
        .get_field_type()
        .get_tensor_type()
        .get_shape()
        .get_dim()
        .iter()
        .map(|x| x.get_dim_value())
        .collect()
}

pub fn generate_buffer<'a>(
    node: &onnx::NodeProto,
    inputs_onnx: &[onnx::ValueInfoProto],
    device: &wgpu::Device,
    inner_infos: &'a mut HashMap<std::string::String, InnerInfo>,
    initializers: &[onnx::TensorProto],
) {
    let inputs = node.get_input();
    let outputs = node.get_output();
    let attributes = node.get_attribute();
    let input_dims = if let Some(inner_info) = inner_infos.get(&inputs[0]) {
        inner_info.dims.clone()
    } else {
        get_dimension(inputs_onnx, &inputs[0])
    };

    match node.get_op_type() {
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" | "Add" | "And" | "Div"
        | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Mod" | "Mul"
        | "Or" | "Sub" | "Celu" | "Elu" | "Relu" | "Sigmoid" | "Softsign" | "Softplus"
        | "Dropout" | "Softmax" | "BatchNormalization" => {
            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&input_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: input_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Conv" => {
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

            let mut output_dims = input_dims.clone();

            {
                let mut inner_info = inner_infos.get_mut(&inputs[1]).unwrap_or_else(|| {
                    panic!("Did not find initializer for input Conv {}", inputs[1])
                });

                inner_info.inner_type = crate::compute::InnerType::Array;
            }

            if inputs.len() == 3 {
                let mut inner_info = inner_infos.get_mut(&inputs[2]).unwrap_or_else(|| {
                    panic!("Did not find initializer for input Conv {}", inputs[2])
                });

                inner_info.inner_type = crate::compute::InnerType::Array;
            }

            let w_dims = &inner_infos.get(&inputs[1]).unwrap().dims;

            match auto_pad {
                "NOTSET" => {
                    output_dims[0] = input_dims[0];
                    output_dims[1] = w_dims[0];
                    output_dims[2] = (input_dims[2] - ((kernel_shape[0] - 1) * dilations[0] + 1)
                        + pads[0]
                        + pads[2])
                        / strides[0]
                        + 1;
                    output_dims[3] = (input_dims[3] - ((kernel_shape[1] - 1) * dilations[1] + 1)
                        + pads[1]
                        + pads[3])
                        / strides[1]
                        + 1;
                }
                "SAME_UPPER" => {
                    output_dims[0] = input_dims[0];
                    output_dims[1] = w_dims[0];
                    output_dims[2] = input_dims[2] / strides[0];
                    output_dims[3] = input_dims[3] / strides[1];
                }
                "SAME_LOWER" => {
                    output_dims[0] = input_dims[0];
                    output_dims[1] = w_dims[0];
                    output_dims[2] = input_dims[2] / strides[0];
                    output_dims[3] = input_dims[3] / strides[1];
                }
                _ => unimplemented!(),
            }

            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        // Experimental
        "Concat" => {
            let mut output_dims = input_dims.clone();
            let input_right_dims = if let Some(inner_info) = inner_infos.get(&inputs[1]) {
                inner_info.dims.clone()
            } else {
                get_dimension(inputs_onnx, &inputs[1])
            };
            output_dims[1] += input_right_dims[1];
            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "MaxPool" | "AveragePool" => {
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

            let mut output_dims = input_dims.clone();

            match auto_pad {
                "NOTSET" => {
                    output_dims[2] = (input_dims[2] - ((kernel_shape[0] - 1) * dilations[0] + 1)
                        + pads[0]
                        + pads[2])
                        / strides[0]
                        + 1;
                    output_dims[3] = (input_dims[3] - ((kernel_shape[1] - 1) * dilations[1] + 1)
                        + pads[1]
                        + pads[3])
                        / strides[1]
                        + 1;
                }
                "SAME_UPPER" => {
                    output_dims[2] = input_dims[2] / strides[0];
                    output_dims[3] = input_dims[3] / strides[1];
                }
                "SAME_LOWER" => {
                    output_dims[2] = input_dims[2] / strides[0];
                    output_dims[3] = input_dims[3] / strides[1];
                }
                _ => unimplemented!(),
            }
            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Transpose" => {
            let perm = attributes
                .iter()
                .find(|attr| attr.get_name() == "perm")
                .unwrap_or_else(|| panic!("Required attribute '{}' not found", "perm"))
                .get_ints();

            let mut output_dims = input_dims.clone();
            for (i, j) in input_dims.iter().zip(perm) {
                output_dims[*j as usize] = *i;
            }

            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Gemm" => {
            let mut trans_a_default = onnx::AttributeProto::new();
            trans_a_default.set_i(0);

            let trans_a = get_attribute("transA", Some(&trans_a_default), node).get_i();

            let mut trans_b_default = onnx::AttributeProto::new();
            trans_b_default.set_i(0);

            let trans_b = get_attribute("transB", Some(&trans_b_default), node).get_i();

            let mut output_dims = input_dims.clone();

            let input_right_dims = if let Some(inner_info) = inner_infos.get(&inputs[1]) {
                inner_info.dims.clone()
            } else {
                get_dimension(inputs_onnx, &inputs[1])
            };

            output_dims[0] = if trans_a == 0 {
                input_dims[0]
            } else {
                input_dims[1]
            };
            output_dims[1] = if trans_b == 0 {
                input_right_dims[1]
            } else {
                input_right_dims[0]
            };

            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "MatMul" => {
            let mut output_dims = input_dims.clone();
            let input_right_dims = if let Some(inner_info) = inner_infos.get(&inputs[1]) {
                inner_info.dims.clone()
            } else {
                get_dimension(inputs_onnx, &inputs[1])
            };
            output_dims[1] = input_right_dims[1];
            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Clip" => {
            {
                let mut inner_info = inner_infos.get_mut(&inputs[1]).unwrap_or_else(|| {
                    panic!("Did not find initializer for input Clip {}", inputs[1])
                });

                inner_info.inner_type = crate::compute::InnerType::Array;
            }
            {
                let mut inner_info = inner_infos.get_mut(&inputs[2]).unwrap_or_else(|| {
                    panic!("Did not find initializer for input Clip {}", inputs[1])
                });

                inner_info.inner_type = crate::compute::InnerType::Array;
            }
        }
        "Reshape" => {
            let reshape = initializers
                .iter()
                .find(|x| x.get_name() == inputs[1].as_str())
                .unwrap_or_else(|| {
                    panic!("Did not find initializer for input Reshape {}", inputs[1])
                });

            let output_dims = if reshape.get_int64_data().to_vec().contains(&-1) {
                vec![input_dims[0], input_dims[1..].iter().product()]
            } else {
                reshape.get_int64_data().to_vec()
            };

            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims,
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Squeeze" => {
            let axis = initializers
                .iter()
                .find(|x| x.get_name() == inputs[1].as_str())
                .unwrap_or_else(|| {
                    panic!("Did not find initializer for input Reshape {}", inputs[1])
                })
                .get_int64_data()
                .to_vec();

            let mut output_dims = input_dims.clone();
            for i in axis {
                output_dims.remove(i as usize);
            }
            inner_infos.insert(
                outputs[0].clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        resource::size(&output_dims.to_vec()) as _,
                        outputs[0].as_str(),
                    ),
                    dims: output_dims.to_vec(),
                    inner_type: crate::compute::InnerType::ArrayVector,
                },
            );
        }
        "Flatten" => {
            let mut axis_default = onnx::AttributeProto::new();

            axis_default.set_f(1.0);

            let axis = get_attribute("axis", Some(&axis_default), node).get_i();

            if axis == axis_default.get_i() {
                let mut output_dims = [0; 2];

                output_dims[0] = input_dims[0];
                output_dims[1] = input_dims[1..].iter().product();
                inner_infos.insert(
                    outputs[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&output_dims.to_vec()) as _,
                            outputs[0].as_str(),
                        ),
                        dims: output_dims.to_vec(),
                        inner_type: crate::compute::InnerType::ArrayVector,
                    },
                );
            } else {
                unimplemented!()
            }
        }
        _ => unimplemented!("The Op: {} is not yet implemented!", node.get_op_type()),
    }
}
