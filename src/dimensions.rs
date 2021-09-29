use crate::get_attribute;
use crate::onnx;
use crate::resource;
use crate::InnerInfo;
use std::collections::HashMap;
use std::str::from_utf8;

use log::debug;
pub fn generate_buffer<'a>(
    input_data: HashMap<String, (&[f32], &[i64])>,
    graph: &onnx::GraphProto,
    device: &wgpu::Device,
    inner_infos: &'a mut HashMap<std::string::String, InnerInfo>,
) -> &'a HashMap<String, InnerInfo> {
    let initializers = graph.get_initializer();
    for (input, (data, dims)) in input_data.iter() {
        inner_infos.insert(
            input.to_string(),
            InnerInfo {
                buffer: resource::create_buffer_init(device, data, input),
                dims: dims.to_vec(),
                inner_type: crate::compute::InnerType::ArrayVector,
            },
        );
    }

    for node in graph.get_node().iter() {
        let inputs = node.get_input();
        let outputs = node.get_output();
        let attributes = node.get_attribute();
        let input_dims = inner_infos
            .get(&inputs[0])
            .expect(format!("Did not find initializer for input: {}", &inputs[0]).as_str())
            .dims
            .clone();

        match node.get_op_type() {
            "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor"
            | "Log" | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" | "Add"
            | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
            | "Mod" | "Mul" | "Or" | "Sub" | "Celu" | "Elu" | "Relu" | "Sigmoid" | "Softsign"
            | "Softplus" | "Dropout" => {
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

                let dilations =
                    get_attribute("dilations", Some(&dilations_default), node).get_ints();

                let kernel_shape = get_attribute("kernel_shape", None, node).get_ints();

                let mut strides_default = onnx::AttributeProto::new();
                strides_default.set_ints(vec![1, 1]);

                let strides = get_attribute("strides", Some(&strides_default), node).get_ints();

                let mut pads_default = onnx::AttributeProto::new();
                pads_default.set_ints(vec![0, 0, 0, 0]);

                let pads = get_attribute("pads", Some(&pads_default), node).get_ints();

                let mut output_dims = input_dims.clone();

                {
                    let mut inner_info = inner_infos.get_mut(&inputs[1]).expect(
                        format!("Did not find initializer for input Conv {}", inputs[1]).as_str(),
                    );

                    inner_info.inner_type = crate::compute::InnerType::Array;
                }

                if inputs.len() == 3 {
                    let mut inner_info = inner_infos.get_mut(&inputs[2]).expect(
                        format!("Did not find initializer for input Conv {}", inputs[1]).as_str(),
                    );

                    inner_info.inner_type = crate::compute::InnerType::Array;
                }

                let w_dims = &inner_infos.get(&inputs[1]).unwrap().dims;

                match auto_pad {
                    "NOTSET" => {
                        output_dims[0] = input_dims[0];
                        output_dims[1] = w_dims[0];
                        output_dims[2] = (input_dims[2]
                            - ((kernel_shape[0] - 1) * dilations[0] + 1)
                            + pads[0]
                            + pads[2])
                            / strides[0]
                            + 1;
                        output_dims[3] = (input_dims[3]
                            - ((kernel_shape[1] - 1) * dilations[1] + 1)
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

            "MaxPool" => {
                // TODO: Conv only support NxCxHxW for the moment.
                debug_assert!(input_dims.len() == 4usize);

                let mut auto_pad_default = onnx::AttributeProto::new();
                auto_pad_default.set_s("NOTSET".to_string().into_bytes());

                let auto_pad =
                    from_utf8(get_attribute("auto_pad", Some(&auto_pad_default), node).get_s())
                        .unwrap();

                let mut dilations_default = onnx::AttributeProto::new();
                dilations_default.set_ints(vec![1, 1]);

                let dilations =
                    get_attribute("dilations", Some(&dilations_default), node).get_ints();

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
                        output_dims[2] = (input_dims[2]
                            - ((kernel_shape[0] - 1) * dilations[0] + 1)
                            + pads[0]
                            + pads[2])
                            / strides[0]
                            + 1;
                        output_dims[3] = (input_dims[3]
                            - ((kernel_shape[1] - 1) * dilations[1] + 1)
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
                    .expect(format!("Required attribute '{}' not found", "perm").as_str())
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
                let mut alpha_default = onnx::AttributeProto::new();
                alpha_default.set_f(1.0);

                let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();

                let mut beta_default = onnx::AttributeProto::new();
                beta_default.set_f(1.0);

                let beta = get_attribute("beta", Some(&beta_default), node).get_f();

                let mut transA_default = onnx::AttributeProto::new();
                transA_default.set_i(0);

                let transA = get_attribute("transA", Some(&transA_default), node).get_i();

                let mut transB_default = onnx::AttributeProto::new();
                transB_default.set_i(0);

                let transB = get_attribute("transB", Some(&transB_default), node).get_i();

                let mut output_dims = input_dims.clone();

                let input_right_dims = inner_infos
                    .get(&inputs[1])
                    .expect(format!("Input: {} has not been provided", inputs[1]).as_str())
                    .dims
                    .clone();

                output_dims[0] = if transA == 0 {
                    input_dims[0]
                } else {
                    input_dims[1]
                };
                output_dims[1] = if transB == 0 {
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
                let input_right_dims = inner_infos
                    .get(&inputs[1])
                    .expect(format!("Input: {} has not been provided", inputs[1]).as_str())
                    .dims
                    .clone();
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
                    let mut inner_info = inner_infos.get_mut(&inputs[1]).expect(
                        format!("Did not find initializer for input Conv {}", inputs[1]).as_str(),
                    );

                    inner_info.inner_type = crate::compute::InnerType::Array;
                }
                {
                    let mut inner_info = inner_infos.get_mut(&inputs[2]).expect(
                        format!("Did not find initializer for input Conv {}", inputs[1]).as_str(),
                    );

                    inner_info.inner_type = crate::compute::InnerType::Array;
                }
            }
            "Reshape" => {
                let reshape = initializers
                    .iter()
                    .find(|x| x.get_name() == inputs[1].as_str())
                    .expect(
                        format!("Did not find initializer for input Reshape {}", inputs[1])
                            .as_str(),
                    );
                let output_dims = reshape.get_int64_data().to_vec();
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
                    .expect(
                        format!("Did not find initializer for input Reshape {}", inputs[1])
                            .as_str(),
                    )
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

    inner_infos
}
