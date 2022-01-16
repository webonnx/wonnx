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
    _shapes_info: &mut HashMap<String, Shape>,
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
    })
}
