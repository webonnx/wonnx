use std::collections::HashMap;

use log::debug;
use wgpu::{Buffer, Device};

use crate::{
    onnx::{self, NodeProto},
    resource::{self, padding},
    utils::{self, node, rename_attribute},
};

pub fn sequence(
    names: &[&str],
    nodes: &[NodeProto],
    device: &Device,
    initializers: &mut HashMap<String, &[u8]>,
    inner_infos: &mut HashMap<String, Buffer>,
) -> (NodeProto, usize) {
    let mut optimisation_length = 1;
    let result = match names {
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
                        && utils::get_attribute("strides", Some(vec![1, 1]), &nodes[0]) == [1, 1]
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
                &(nodes[0].get_name().to_string() + nodes[1].get_name()),
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

    (result, optimisation_length)
}
