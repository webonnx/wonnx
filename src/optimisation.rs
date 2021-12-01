use std::collections::HashMap;

use crate::{
    onnx::{self, NodeProto},
    resource,
    resource::padding,
    utils::node,
    utils::{self, len},
    utils::{attribute, rename_attribute},
};

use log::debug;
use tera::Tera;

const MAX_OPTIMIZATION_LEN: usize = 7;
pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(Vec<NodeProto>, HashMap<String, wgpu::Buffer>), wgpu::Error> {
    let tera = match Tera::new("templates/**/*.wgsl") {
        Ok(t) => t,
        Err(e) => {
            panic!("Parsing error(s): {}", e);
        }
    };

    let mut initializers = HashMap::new();
    for initializer in graph.get_initializer() {
        let input = initializer.get_name().to_string();
        let data = initializer.get_float_data();
        let raw_data = if !data.is_empty() {
            bytemuck::cast_slice(data)
        } else {
            initializer.get_raw_data()
        };

        initializers.insert(input, raw_data);
    }

    let mut value_info = HashMap::new();

    for info in graph.get_input() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph.get_output() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph.get_value_info() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        value_info.insert(info.get_name().to_string(), dims);
    }

    let base_nodes = graph.get_node();

    let mut inner_infos = HashMap::new();
    let n = base_nodes.len();

    let mut node_index = 0;

    let mut optimised_nodes = vec![];

    while node_index < n {
        let nodes = &base_nodes[node_index..(usize::min(node_index + MAX_OPTIMIZATION_LEN, n))];
        let names = nodes
            .iter()
            .map(|node| node.get_op_type())
            .collect::<Vec<&str>>();
        let mut optimisation_length = 1;

        let mut runnable_node = match names.as_slice() {
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
                let w_0_data = initializers.remove(inputs[1]).unwrap();
                let b_0_data = initializers.remove(inputs[2]).unwrap();
                let w_1_data = initializers.remove(inputs[3]).unwrap();
                let b_1_data = initializers.remove(inputs[4]).unwrap();
                let w_2_data = initializers.remove(inputs[5]).unwrap();
                let b_2_data = initializers.remove(inputs[6]).unwrap();

                let mut w_0 = w_0_data.to_vec();
                w_0.extend(w_1_data);

                inner_infos.insert(
                    inputs[1].to_string(),
                    resource::create_buffer_init(device, &w_0, &raw_inputs[1]),
                );

                let mut b_0 = b_0_data.to_vec();
                b_0.extend(b_1_data);
                b_0.extend(b_2_data);

                inner_infos.insert(
                    inputs[2].to_string(),
                    resource::create_buffer_init(device, &b_0, inputs[2]),
                );
                let w_2_data = padding(w_2_data, 12, 4);

                inner_infos.insert(
                    inputs[5].to_string(),
                    resource::create_buffer_init(device, &w_2_data, inputs[5]),
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
                    if let Some(data) = initializers.remove(input) {
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
                                resource::create_buffer_init(device, data.as_slice(), input),
                            );
                        } else {
                            debug!("Not inserting input: {}", input);
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
                    if let Some(data) = initializers.remove(input) {
                        if !data.is_empty() {
                            inner_infos.insert(
                                input.to_string(),
                                resource::create_buffer_init(device, data, input),
                            );
                        } else {
                            debug!("Not inserting input: {}", input);
                        };
                    }
                }

                nodes[0].clone()
            }
        };

        node_index += optimisation_length;

        // Initalialising Output
        let output = &nodes[optimisation_length - 1].get_output()[0];
        if let Some(output_dims) = value_info.get(output) {
            inner_infos.insert(
                output.clone(),
                resource::create_buffer(device, len(output_dims) as _, output.as_str()),
            );
        } else {
            panic!("output dims was not provided. You can use python's onnx-simplifier to generate implied dimensions.")
        }

        let (shader, x, y, z) = crate::compiler::format_node(&runnable_node, &value_info, &tera);
        debug!("shader: {}", shader);
        let attributes = runnable_node.mut_attribute();
        attributes.push(attribute("WGSL", shader));
        attributes.push(attribute("threads", vec![x as i64, y as i64, z as i64]));

        optimised_nodes.push(runnable_node);
    }

    //  graph.set_node(RepeatedField::from(optimised_nodes));

    Ok((optimised_nodes, inner_infos))
}

pub fn load_sequentially(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(Vec<NodeProto>, HashMap<String, wgpu::Buffer>), wgpu::Error> {
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

    let original_nodes = graph.get_node();

    let mut inner_infos = HashMap::new();

    let mut optimised_nodes = vec![];

    for node in original_nodes {
        let inputs = node.get_input();
        for input in inputs {
            if let Some((dims, data)) = initializers.remove(input) {
                if !data.is_empty() {
                    inner_infos.insert(
                        input.to_string(),
                        resource::create_buffer_init(device, data, input),
                    );
                } else {
                    debug!("Not inserting input: {} with shape: {:?}", input, dims);
                };
            }
        }

        let runnable_node = node.clone();
        optimised_nodes.push(runnable_node);
    }

    //  graph.set_node(RepeatedField::from(optimised_nodes));

    Ok((optimised_nodes, inner_infos))
}
