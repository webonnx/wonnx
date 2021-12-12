use crate::{
    onnx::{NodeProto},
    resource,
    sequencer::sequence,
    utils::{len},
    utils::{attribute},
    Result,
};

use std::collections::HashMap;

use log::debug;
use tera::Tera;

const MAX_OPTIMIZATION_LEN: usize = 7;

lazy_static! {
    pub static ref TEMPLATES: Tera = {
        let mut tera = Tera::default();
        tera.add_raw_template("endomorphism/activation.wgsl", include_str!("../templates/endomorphism/activation.wgsl")).unwrap();
        tera.add_raw_template("endomorphism/arithmetic.wgsl", include_str!("../templates/endomorphism/arithmetic.wgsl")).unwrap();
        tera.add_raw_template("endomorphism/batchnormalization.wgsl", include_str!("../templates/endomorphism/batchnormalization.wgsl")).unwrap();
        tera.add_raw_template("endomorphism/copy.wgsl", include_str!("../templates/endomorphism/copy.wgsl")).unwrap();
        tera.add_raw_template("endomorphism/map.wgsl", include_str!("../templates/endomorphism/map.wgsl")).unwrap();
        tera.add_raw_template("containers/conv.wgsl", include_str!("../templates/containers/conv.wgsl")).unwrap();
        tera.add_raw_template("containers/squeezenet_conv_group.wgsl", include_str!("../templates/containers/squeezenet_conv_group.wgsl")).unwrap();
        tera.add_raw_template("matrix/concat.wgsl", include_str!("../templates/matrix/concat.wgsl")).unwrap();
        tera.add_raw_template("matrix/gemm_1.wgsl", include_str!("../templates/matrix/gemm_1.wgsl")).unwrap();
        tera.add_raw_template("matrix/gemm.wgsl", include_str!("../templates/matrix/gemm.wgsl")).unwrap();
        tera.add_raw_template("matrix/transpose.wgsl", include_str!("../templates/matrix/transpose.wgsl")).unwrap();
        tera.add_raw_template("pool/aggregate.wgsl", include_str!("../templates/pool/aggregate.wgsl")).unwrap();
        tera.add_raw_template("pool/conv_kernel_1.wgsl", include_str!("../templates/pool/conv_kernel_1.wgsl")).unwrap();
        tera.add_raw_template("pool/conv_kernel_3.wgsl", include_str!("../templates/pool/conv_kernel_3.wgsl")).unwrap();
        tera.add_raw_template("pool/conv.wgsl", include_str!("../templates/pool/conv.wgsl")).unwrap();
        tera.add_raw_template("base.wgsl", include_str!("../templates/base.wgsl")).unwrap();
        tera.add_raw_template("structs.wgsl", include_str!("../templates/structs.wgsl")).unwrap();
        tera
    };
}

pub fn load(
    graph: &crate::onnx::GraphProto,
    device: &wgpu::Device,
) -> Result<(Vec<NodeProto>, HashMap<String, wgpu::Buffer>)> {
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
        let (mut current_node, optimisation_length) =
            sequence(&names, nodes, device, &mut initializers, &mut inner_infos);

        // Exit node if necessary.
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

        let (shader, x, y, z) = crate::compiler::format_node(&current_node, &value_info, &TEMPLATES);
        debug!("shader: {}", shader);
        let attributes = current_node.mut_attribute();
        attributes.push(attribute("WGSL", shader));
        attributes.push(attribute("threads", vec![x as i64, y as i64, z as i64]));

        optimised_nodes.push(current_node);

        node_index += optimisation_length;
    }

    Ok((optimised_nodes, inner_infos))
}
