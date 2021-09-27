use crate::get_attribute;
use crate::onnx;
use crate::resource;
use crate::InnerInfo;
use std::collections::HashMap;
use std::str::from_utf8;

use log::debug;
pub fn generate_buffer(
    input_data: HashMap<String, (&[f32], &[i64])>,
    graph: &onnx::GraphProto,
    device: &wgpu::Device,
) -> HashMap<String, InnerInfo> {
    let mut inner_infos = HashMap::new();
    let initializers = graph.get_initializer();
    for input in graph.get_input().iter() {
        let name = input.get_name();
        let (data, dim) = input_data
            .get(name)
            .unwrap_or_else(|| panic!("Input: {name} was not found in user HashMap.", name = name));
        inner_infos.insert(
            name.to_string(),
            InnerInfo {
                buffer: resource::create_buffer_init(device, data, name),
                dims: dim.to_vec(),
                inner_type: crate::compute::InnerType::ArrayVector,
            },
        );
    }

    for node in graph.get_node().iter() {
        let input = node.get_input();
        let output = node.get_output();
        let attributes = node.get_attribute();

        let input_dims = inner_infos
            .get(&input[0])
            .expect(format!("Input: {} has not been provided", input[0]).as_str())
            .dims
            .clone();
        debug!(
            "resource::size(input_dims): {:#?}",
            resource::size(&input_dims)
        );
        match node.get_op_type() {
            "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor"
            | "Log" | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" | "Add"
            | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
            | "Mod" | "Mul" | "Or" | "Sub" | "Celu" | "Elu" | "Relu" | "Sigmoid" | "Softsign"
            | "Softplus" => {
                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&input_dims) as _,
                            output[0].as_str(),
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

                let w = initializers
                    .iter()
                    .find(|x| x.get_name() == input[1].as_str())
                    .expect(
                        format!("Did not find initializer for input Conv {}", input[1]).as_str(),
                    );
                let w_data = w.get_float_data();
                let w_dims = w.get_dims();

                inner_infos.insert(
                    input[1].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, w_data, input[1].as_str()),
                        dims: w_dims.to_vec(),
                        inner_type: crate::compute::InnerType::Array,
                    },
                );

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
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&input_dims) as _,
                            output[0].as_str(),
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
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&input_dims) as _,
                            output[0].as_str(),
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
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&output_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: output_dims,
                        inner_type: crate::compute::InnerType::ArrayVector,
                    },
                );
            }
            "MatMul" => {
                let mut output_dims = input_dims.clone();
                let input_right_dims = inner_infos
                    .get(&input[1])
                    .expect(format!("Input: {} has not been provided", input[1]).as_str())
                    .dims
                    .clone();
                output_dims[1] = input_right_dims[1];
                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&output_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: output_dims,
                        inner_type: crate::compute::InnerType::ArrayVector,
                    },
                );
            }
            "Flatten" => {
                let mut axis_default = onnx::AttributeProto::new();

                axis_default.set_f(1.0);

                let axis = get_attribute("axis", Some(&axis_default), node).get_i();

                unimplemented!()
            }
            _ => unimplemented!(),
        }
    }

    inner_infos
}
