use std::collections::HashMap;

use protobuf::RepeatedField;
use wgpu::{Buffer, BufferUsages, Device};

use crate::{
    onnx::NodeProto,
    resource::{self, padding},
    utils::{get_attribute, node},
};

pub fn sequence(
    names: &[&str],
    nodes: &[NodeProto],
    device: &Device,
    initializers: &HashMap<String, &[u8]>,
    inner_infos: &mut HashMap<String, Buffer>,
) -> (NodeProto, usize) {
    let mut optimisation_length = 1;
    let result = match names {
        ["Conv", "Relu", ..] => {
            optimisation_length = 2;
            let inputs = nodes[0].get_input();
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

            node(
                nodes[0].get_input().iter().map(|x| x.as_str()).collect(),
                nodes[1].get_output().iter().map(|x| x.as_str()).collect(),
                &(nodes[0].get_name().to_string() + nodes[1].get_name()),
                "ConvRelu",
                nodes[0].get_attribute().to_vec(),
            )
        }
        ["Reshape", ..] | ["Clip", ..] | ["Squeeze", ..] => {
            // Remove non binding related input for those Op
            let input = &nodes[0].get_input()[0];
            if let Some(data) = initializers.get(input) {
                inner_infos.insert(
                    input.to_string(),
                    resource::create_buffer_init(device, data, input, BufferUsages::STORAGE),
                );
            }

            let mut node = nodes[0].clone();
            node.set_input(RepeatedField::from(vec![input.clone()]));
            node
        }
        [..] => {
            let inputs = nodes[0].get_input();
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
