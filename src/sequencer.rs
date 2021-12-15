use std::collections::HashMap;

use bytemuck::cast_slice;
use protobuf::RepeatedField;
use wgpu::{Buffer, BufferUsages, Device};

use crate::{
    onnx::NodeProto,
    resource::{self, padding},
    utils::{attribute, get_attribute, node},
};

pub fn sequence(
    names: &[&str],
    nodes: &[NodeProto],
    device: &Device,
    initializers: &HashMap<String, &[u8]>,
    inner_infos: &mut HashMap<String, Buffer>,
) -> (NodeProto, usize) {
    let mut optimisation_length = 1;
    let inputs = nodes[0].get_input();

    let result = match names {
        ["Conv", "Exp", "Add", "Log", "Tanh", "Mul", ..] => {
            optimisation_length = 6;
            node(
                inputs.iter().map(|x| x.as_str()).collect(),
                nodes[6].get_output().iter().map(|x| x.as_str()).collect(),
                &(nodes[0].get_name().to_string() + nodes[1].get_name()),
                "ConvMish",
                nodes[0].get_attribute().to_vec(),
            )
        }
        ["Conv", "Relu", ..] | ["Conv", "LeakyRelu", ..] => {
            optimisation_length = 2;
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    let data = if input == &inputs[1]
                        && get_attribute::<Vec<i64>>("kernel_shape", None, &nodes[0]) == [3, 3]
                        && get_attribute("pads", Some(vec![0, 0, 0, 0]), &nodes[0]) == [1, 1, 1, 1]
                        && get_attribute("strides", Some(vec![1, 1]), &nodes[0]) == [1, 1]
                    {
                        padding(data, 12, 4)
                        // data.to_vec()
                    } else {
                        data.to_vec()
                    };
                    let data = data.to_vec();

                    // debug_assert!(!data.is_empty(), "Not inserting input: {}", input);

                    inner_infos.insert(
                        input.to_string(),
                        resource::create_buffer_init(
                            device,
                            data.as_slice(),
                            input,
                            BufferUsages::STORAGE,
                        ),
                    );
                }
            }

            let mut attributes = nodes[0].get_attribute().to_vec();
            for attribute in nodes[1].get_attribute() {
                attributes.push(attribute.clone());
            }

            node(
                inputs.iter().map(|x| x.as_str()).collect(),
                nodes[1].get_output().iter().map(|x| x.as_str()).collect(),
                &(nodes[0].get_name().to_string() + nodes[1].get_name()),
                "ConvRelu",
                attributes,
            )
        }
        op
        @ (["Reshape", ..] | ["Clip", ..] | ["Squeeze", ..] | ["Split", ..] | ["Resize", ..]) => {
            // Remove non binding related input for those Op
            let mut inputs = inputs.iter();

            // Remove the first input.
            let input = inputs.next().unwrap();
            if let Some(data) = initializers.get(input) {
                inner_infos.insert(
                    input.to_string(),
                    resource::create_buffer_init(device, data, input, BufferUsages::STORAGE),
                );
            }

            let mut node = nodes[0].clone();
            node.set_input(RepeatedField::from(vec![input.clone()]));
            // Transform some intput into attributes for optimisation.
            let mut attributes = node.take_attribute();

            for input in inputs {
                match (op, input.as_str()) {
                    (["Split", ..], "split") => {
                        let value: Vec<i64> = cast_slice(initializers.get(input).unwrap()).to_vec();
                        attributes.push(attribute(input, value));
                    }
                    (["Resize", ..], "roi") => {
                        let value: Vec<i64> = cast_slice(initializers.get(input).unwrap()).to_vec();
                        attributes.push(attribute(input, value));
                    }
                    (["Resize", ..], "scales") => {
                        let value: Vec<f32> = cast_slice(initializers.get(input).unwrap()).to_vec();
                        attributes.push(attribute(input, value));
                    }
                    (["Resize", ..], "sizes") => {
                        let value: Vec<i64> = cast_slice(initializers.get(input).unwrap()).to_vec();
                        attributes.push(attribute(input, value));
                    }
                    _ => (),
                }
            }

            node.set_attribute(attributes);

            node
        }
        op @ (["Mul", ..] | ["Add", ..]) => {
            let mut ending_input = vec![];
            let mut attributes = vec![];
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    // debug_assert!(!data.is_empty(), "Not inserting input: {}", input);

                    match (data.len(), op) {
                        (4.., _) => {
                            inner_infos.insert(
                                input.to_string(),
                                resource::create_buffer_init(
                                    device,
                                    data,
                                    input,
                                    BufferUsages::STORAGE,
                                ),
                            );
                            ending_input.push(input.clone());
                        }
                        (1, ["Mul", ..]) => {
                            let coeff: Vec<f32> = bytemuck::cast_slice(data).to_vec();
                            attributes.push(attribute("coefficient", coeff[0]));
                        }
                        (3, ["Add", ..]) => {
                            if data == &[0, 0, 0] {
                                attributes.push(attribute("coefficient", 0));
                            } else {
                                unimplemented!()
                            }
                        }
                        _ => {
                            unimplemented!()
                        }
                    }
                }
            }

            let mut node = nodes[0].clone();
            node.set_input(RepeatedField::from(ending_input));
            node
        }
        [..] => {
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    // debug_assert!(!data.is_empty(), "Not inserting input: {}", input);

                    inner_infos.insert(
                        input.to_string(),
                        resource::create_buffer_init(device, data, input, BufferUsages::STORAGE),
                    );
                }
            }

            nodes[0].clone()
        }
    };

    (result, optimisation_length)
}
