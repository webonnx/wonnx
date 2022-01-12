use std::collections::HashMap;

use bytemuck::cast_slice;
use protobuf::RepeatedField;
use thiserror::Error;
use wgpu::{Buffer, BufferUsages, Device};

use crate::{
    onnx::NodeProto,
    resource::{self, padding},
    utils::{attribute, get_attribute, node, AttributeNotFoundError, Shape},
};

#[derive(Error, Debug)]
pub enum SequenceError {
    #[error("a required attribute was not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),

    #[error("{0} is not implemented yet")]
    NotImplemented(String),

    #[error("could not cull identity node")]
    CullFailed,
}

pub struct Sequence {
    pub node: NodeProto,
    pub nodes_consumed: usize,
}

/* This function will take in a list of nodes and their op names, and will consume one or more nodes from the front of
the list and return a single node representing the removed node(s), as well as the number of nodes consumed from the
sequence. This function can coalesce or even skip certain nodes. */
pub fn sequence(
    names: &[&str],
    nodes: &[NodeProto],
    device: &Device,
    initializers: &HashMap<String, &[u8]>,
    inner_infos: &mut HashMap<String, Buffer>,
    shapes_info: &mut HashMap<String, Shape>,
) -> Result<Sequence, SequenceError> {
    assert_eq!(names.len(), nodes.len());
    let inputs = nodes[0].get_input();

    Ok(match names {
        ["Conv", "Exp", "Add", "Log", "Tanh", "Mul", ..] => Sequence {
            node: node(
                inputs.iter().map(|x| x.as_str()).collect(),
                nodes[6].get_output().iter().map(|x| x.as_str()).collect(),
                &(nodes[0].get_name().to_string() + nodes[1].get_name()),
                "ConvMish",
                nodes[0].get_attribute().to_vec(),
            ),
            nodes_consumed: 6,
        },
        ["Conv", "Relu", ..] | ["Conv", "LeakyRelu", ..] => {
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    let data = if input == &inputs[1]
                        && get_attribute::<Vec<i64>>("kernel_shape", None, &nodes[0])? == [3, 3]
                        && (get_attribute("pads", Some(vec![0, 0, 0, 0]), &nodes[0])?
                            == [1, 1, 1, 1]
                            || get_attribute(
                                "auto_pad",
                                Some("SAME_UPPER".to_string()),
                                &nodes[0],
                            )? == "SAME_UPPER")
                        && get_attribute("strides", Some(vec![1, 1]), &nodes[0])? == [1, 1]
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

            Sequence {
                nodes_consumed: 2,
                node: node(
                    inputs.iter().map(|x| x.as_str()).collect(),
                    nodes[1].get_output().iter().map(|x| x.as_str()).collect(),
                    &(nodes[0].get_name().to_string() + nodes[1].get_name()),
                    "ConvRelu",
                    attributes,
                ),
            }
        }

        /* These ops can simply be skipped if they are followed by another op. This is efficiently done by replacing the
        input of the next op (which is the output of a skippable op) with the input of the skippable op itself. */
        ["Identity", _next_op, ..]
        | ["Squeeze", _next_op, ..]
        | ["Unsqueeze", _next_op, ..]
        | ["Flatten", _next_op, ..]
        | ["Dropout", _next_op, ..]
        | ["Reshape", _next_op, ..] => {
            // Replace the input received from the identity op with the input the identity op takes
            let identity_input_name = &nodes[0].get_input()[0];
            let identity_output_name = &nodes[0].get_output()[0];

            // The input to the identity op can still be an initializer
            if let Some(data) = initializers.get(identity_input_name) {
                inner_infos.insert(
                    identity_input_name.to_string(),
                    resource::create_buffer_init(
                        device,
                        data,
                        identity_input_name,
                        BufferUsages::STORAGE,
                    ),
                );
            }

            let mut node = nodes[1].clone();

            let mut found = false;
            for input in node.mut_input() {
                if input == identity_output_name {
                    if found {
                        return Err(SequenceError::CullFailed);
                    }
                    *input = identity_input_name.clone();
                    found = true;
                } else if let Some(data) = initializers.get(input) {
                    inner_infos.insert(
                        input.to_string(),
                        resource::create_buffer_init(device, data, input, BufferUsages::STORAGE),
                    );
                }
            }

            if found {
                // Patch the dims of the re-used input
                shapes_info.insert(
                    identity_input_name.clone(),
                    shapes_info.get(identity_output_name).unwrap().clone(),
                );
            } else {
                return Err(SequenceError::CullFailed);
            }

            Sequence {
                node,
                nodes_consumed: 2,
            }
        }
        op @ (["Clip", ..] | ["Split", ..] | ["Resize", ..] | ["Reshape", ..]) => {
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
                    (["Reshape", ..], "shape") => {
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

            Sequence {
                node,
                nodes_consumed: 1,
            }
        }
        op @ (["Mul", ..] | ["Add", ..]) => {
            let mut ending_input = vec![];
            let mut attributes = vec![];
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    match (data.len(), op) {
                        (4, ["Mul", ..]) => {
                            let coeff: Vec<f32> = bytemuck::cast_slice(data).to_vec();
                            attributes.push(attribute("coefficient", coeff[0]));
                        }
                        (12, ["Add", ..]) => {
                            if data == &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
                                attributes.push(attribute("coefficient", 0));
                            } else {
                                return Err(SequenceError::NotImplemented(String::from(
                                    "Add with non-zero data",
                                )));
                            }
                        }
                        _ => {
                            inner_infos.insert(
                                input.to_string(),
                                resource::create_buffer_init(
                                    device,
                                    data,
                                    input,
                                    BufferUsages::STORAGE,
                                ),
                            );
                        }
                    }
                } else {
                    ending_input.push(input.clone());
                }
            }

            let mut node = nodes[0].clone();
            node.set_input(RepeatedField::from(ending_input));
            Sequence {
                node,
                nodes_consumed: 1,
            }
        }
        [..] => {
            for input in inputs {
                if let Some(data) = initializers.get(input) {
                    // debug_assert!(!data.is_empty(), "Not inserting input: {}", input);
                    let mut data = data.to_vec();

                    // Prevent issue with minimum buffer size for binding enforced by wgpu
                    if data.len() < 16 {
                        data.resize(16, 0);
                    }
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

            Sequence {
                node: nodes[0].clone(),
                nodes_consumed: 1,
            }
        }
    })
}
