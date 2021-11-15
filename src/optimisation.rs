use std::collections::HashMap;

use crate::{
    onnx::{self, NodeProto},
    resource,
    resource::padding,
    utils::node,
    utils::rename_attribute,
    utils::{self, get_dimension, len},
    InnerInfo,
};

use log::debug;

const MAX_OPTIMIZATION_LEN: usize = 14;
pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(Vec<NodeProto>, HashMap<String, InnerInfo>), wgpu::Error> {
    let initializers = graph.get_initializer();
    let mut hash = HashMap::new();
    for initializer in initializers {
        let input = initializer.get_name().to_string();
        let dims = initializer.get_dims().to_vec();
        let data = initializer.get_float_data();
        let raw_data = if !data.is_empty() {
            bytemuck::cast_slice(data)
        } else {
            initializer.get_raw_data()
        };

        hash.insert(input, (dims, raw_data));
    }
    let mut initializers = hash;

    let value_info = graph.get_value_info();
    let output_info = &graph.get_output();
    let original_nodes = graph.get_node();

    let mut inner_infos = HashMap::new();
    let n = original_nodes.iter().count();

    let mut node_index = 0;

    let mut optimised_nodes = vec![];

    while node_index < n {
        let nodes = &original_nodes[node_index..(usize::min(node_index + MAX_OPTIMIZATION_LEN, n))];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<&str>>();
        let mut optimisation_length = 1;

        let runnable_node = match names.as_slice() {
            ["Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Concat", "Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Concat", ..] =>
            {
                optimisation_length = 14;
                let raw_inputs = nodes[0].get_input();

                let mut inputs = nodes[0]
                    .get_input()
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>();

                inputs.append(
                    &mut nodes[2].get_input()[1..=2]
                        .iter()
                        .map(|x| x.as_str())
                        .collect::<Vec<&str>>(),
                );

                inputs.append(
                    &mut nodes[4].get_input()[1..=2]
                        .iter()
                        .map(|x| x.as_str())
                        .collect::<Vec<&str>>(),
                );

                let mut attributes = nodes[0]
                    .get_attribute()
                    .iter()
                    .map(|x| rename_attribute(x, x.get_name().to_string() + "_0"))
                    .collect::<Vec<onnx::AttributeProto>>();

                attributes.append(
                    &mut nodes[2]
                        .get_attribute()
                        .iter()
                        .map(|x| rename_attribute(x, x.get_name().to_string() + "_1"))
                        .collect::<Vec<onnx::AttributeProto>>(),
                );
                attributes.append(
                    &mut nodes[4]
                        .get_attribute()
                        .iter()
                        .map(|x| rename_attribute(x, x.get_name().to_string() + "_2"))
                        .collect::<Vec<onnx::AttributeProto>>(),
                );
                let (mut w_0_dims, w_0_data) = initializers.remove(inputs[1]).unwrap();
                let (mut b_0_dims, b_0_data) = initializers.remove(inputs[2]).unwrap();
                let (mut w_1_dims, w_1_data) = initializers.remove(inputs[3]).unwrap();
                let (mut b_1_dims, b_1_data) = initializers.remove(inputs[4]).unwrap();
                let (w_2_dims, w_2_data) = initializers.remove(inputs[5]).unwrap();
                let (mut b_2_dims, b_2_data) = initializers.remove(inputs[6]).unwrap();

                let mut w_0 = w_0_data.to_vec();
                w_0.extend(w_1_data);
                w_0_dims.append(&mut w_1_dims);

                inner_infos.insert(
                    inputs[1].to_string(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, &w_0, &raw_inputs[1]),
                        dims: w_0_dims,
                    },
                );

                let mut b_0 = b_0_data.to_vec();
                b_0.extend(b_1_data);
                b_0.extend(b_2_data);
                b_0_dims.append(&mut b_1_dims);
                b_0_dims.append(&mut b_2_dims);

                inner_infos.insert(
                    inputs[2].to_string(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, &b_0, &inputs[2]),
                        dims: b_0_dims,
                    },
                );
                let w_2_data = padding(w_2_data, 12, 4);

                inner_infos.insert(
                    inputs[5].to_string(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, &w_2_data, &inputs[5]),
                        dims: w_2_dims,
                    },
                );

                node(
                    vec![inputs[0], inputs[1], inputs[2]],
                    nodes[13].get_output().iter().map(|x| x.as_str()).collect(),
                    "SqueezenetConvGroup",
                    "SqueezenetConvGroup",
                    attributes,
                )
            }
            ["Conv", "Relu", ..] => {
                optimisation_length = 2;
                let inputs = nodes[0].get_input();
                for input in inputs {
                    if let Some((dims, data)) = initializers.remove(input) {
                        let data = if input == &inputs[1]
                            && utils::get_attribute::<Vec<i64>>("kernel_shape", None, &nodes[0])
                                == [3, 3]
                            && utils::get_attribute("pads", Some(vec![0, 0, 0, 0]), &nodes[0])
                                == [1, 1, 1, 1]
                            && utils::get_attribute("strides", Some(vec![1, 1]), &nodes[0])
                                == [1, 1]
                        {
                            padding(data, 12, 4)
                        } else {
                            data.to_vec()
                        };

                        if !data.is_empty() {
                            inner_infos.insert(
                                input.to_string(),
                                InnerInfo {
                                    buffer: resource::create_buffer_init(
                                        device,
                                        data.as_slice(),
                                        input,
                                    ),
                                    dims,
                                },
                            );
                        } else {
                            debug!("Not inserting input: {} with shape: {:?}", input, dims);
                        };
                    }
                }

                node(
                    nodes[0].get_input().iter().map(|x| x.as_str()).collect(),
                    nodes[1].get_output().iter().map(|x| x.as_str()).collect(),
                    "ConvRelu",
                    "ConvRelu",
                    nodes[0].get_attribute().to_vec(),
                )
            }
            [..] => {
                let inputs = nodes[0].get_input();
                for input in inputs {
                    if let Some((dims, data)) = initializers.remove(input) {
                        if !data.is_empty() {
                            inner_infos.insert(
                                input.to_string(),
                                InnerInfo {
                                    buffer: resource::create_buffer_init(device, data, input),
                                    dims,
                                },
                            );
                        } else {
                            debug!("Not inserting input: {} with shape: {:?}", input, dims);
                        };
                    }
                }

                nodes[0].clone()
            }
        };

        node_index += optimisation_length;

        // Initalialising Output
        let output = &nodes[optimisation_length - 1].get_output()[0];
        if let Some(output_dims) = get_dimension(value_info, &output) {
            inner_infos.insert(
                output.clone(),
                InnerInfo {
                    buffer: resource::create_buffer(
                        device,
                        len(&output_dims) as _,
                        output.as_str(),
                    ),
                    dims: output_dims,
                },
            );
        } else if let Some(_) = get_dimension(output_info, &output) {
        } else {
            panic!("output dims was not provided. You can use python's onnx-simplifier to generate implied dimensions.")
        }
        optimised_nodes.push(runnable_node);
    }

    //  graph.set_node(RepeatedField::from(optimised_nodes));

    Ok((optimised_nodes, inner_infos))
}
