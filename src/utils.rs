use crate::onnx;
use std::collections::HashMap;
use std::convert::From;
use std::convert::Into;
use std::str::from_utf8;

pub fn len(dims: &[i64]) -> i64 {
    dims.iter().product::<i64>()
}

pub fn get_attribute<T: std::convert::From<onnx::AttributeProto>>(
    attribute: &str,
    default: Option<T>,
    node: &onnx::NodeProto,
) -> T {
    match (
        node.get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute),
        default,
    ) {
        (Some(attr), _) => attr.clone().into(),
        (None, Some(default_attr)) => default_attr,
        (None, None) => panic!(
            "Did not find attribute: {} for node: {}",
            attribute,
            node.get_name()
        ),
    }
}

pub fn ceil(num: i64, div: i64) -> i64 {
    num / div + (num % div != 0) as i64
}

pub fn rename_attribute(
    attribute: &onnx::AttributeProto,
    new_name: String,
) -> onnx::AttributeProto {
    let mut attr = attribute.clone();
    attr.set_name(new_name);
    attr
}

pub fn dimensions_infos(graph_proto: &onnx::GraphProto) -> HashMap<String, Vec<i64>> {
    let mut dims_info = HashMap::new();

    for info in graph_proto.get_input() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        dims_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph_proto.get_output() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        dims_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph_proto.get_value_info() {
        let dims = info
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .map(|x| x.get_dim_value())
            .collect::<Vec<i64>>();
        dims_info.insert(info.get_name().to_string(), dims);
    }

    for info in graph_proto.get_initializer() {
        let dims = info.get_dims().to_vec();
        dims_info.insert(info.get_name().to_string(), dims);
    }

    dims_info
}

pub fn initializers(graph_proto: &onnx::GraphProto) -> HashMap<String, &[u8]> {
    let mut initializers = HashMap::new();
    for initializer in graph_proto.get_initializer() {
        let input = initializer.get_name().to_string();
        let data = initializer.get_float_data();
        let raw_data = if !data.is_empty() {
            bytemuck::cast_slice(data)
        } else {
            initializer.get_raw_data()
        };

        initializers.insert(input, raw_data);
    }
    initializers
}
// TODO: Make dimension optional
pub fn tensor(name: &str, dimensions: &[i64]) -> onnx::ValueInfoProto {
    let mut dim_value = vec![];
    for dimension in dimensions {
        let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
        dim_channel.set_dim_value(*dimension);
        dim_value.push(dim_channel);
    }

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(dim_value));

    let mut type_proto_tensor = onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(1);
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut tensor = onnx::ValueInfoProto::new();
    tensor.set_name(name.to_string());
    tensor.set_field_type(type_proto);

    tensor
}

// Remove dimensions
pub fn initializer(name: &str, data: Vec<f32>) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    initializer.set_name(name.to_string());
    initializer.set_float_data(data);
    initializer
}

pub fn attribute(name: &str, inputs: impl Into<onnx::AttributeProto>) -> onnx::AttributeProto {
    let mut attributes: onnx::AttributeProto = inputs.into();
    attributes.set_name(name.to_string());
    attributes
}

pub fn node(
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    name: &str,
    op_type: &str,
    attributes: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    let mut node = crate::onnx::NodeProto::new();

    node.set_op_type(op_type.to_string());
    node.set_name(name.to_string());
    node.set_input(protobuf::RepeatedField::from(
        inputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_output(protobuf::RepeatedField::from(
        outputs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    ));
    node.set_attribute(protobuf::RepeatedField::from(attributes));
    node
}

pub fn graph(
    inputs: Vec<onnx::ValueInfoProto>,
    outputs: Vec<onnx::ValueInfoProto>,
    infos: Vec<onnx::ValueInfoProto>,
    initializers: Vec<onnx::TensorProto>,
    nodes: Vec<onnx::NodeProto>,
) -> onnx::GraphProto {
    let mut graph = onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(nodes));
    graph.set_input(protobuf::RepeatedField::from(inputs));
    graph.set_value_info(protobuf::RepeatedField::from(infos));
    graph.set_output(protobuf::RepeatedField::from(outputs));
    graph.set_initializer(protobuf::RepeatedField::from(initializers));
    graph
}

pub fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);
    model
}

impl From<Vec<i64>> for onnx::AttributeProto {
    fn from(value: Vec<i64>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_ints(value);
        attributes
    }
}

impl From<Vec<f32>> for onnx::AttributeProto {
    fn from(value: Vec<f32>) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_floats(value);
        attributes
    }
}

impl From<f32> for onnx::AttributeProto {
    fn from(value: f32) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_f(value);
        attributes
    }
}

impl From<i64> for onnx::AttributeProto {
    fn from(value: i64) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_i(value);
        attributes
    }
}

impl From<String> for onnx::AttributeProto {
    fn from(value: String) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.into_bytes());
        attributes
    }
}

impl From<&str> for onnx::AttributeProto {
    fn from(value: &str) -> Self {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(value.to_string().into_bytes());
        attributes
    }
}

impl From<onnx::AttributeProto> for Vec<i64> {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_ints().to_vec()
    }
}

impl From<onnx::AttributeProto> for Vec<f32> {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_floats().to_vec()
    }
}

impl From<onnx::AttributeProto> for f32 {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_f()
    }
}

impl From<onnx::AttributeProto> for i64 {
    fn from(value: onnx::AttributeProto) -> Self {
        value.get_i()
    }
}

impl From<onnx::AttributeProto> for String {
    fn from(value: onnx::AttributeProto) -> Self {
        from_utf8(value.get_s()).unwrap().to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{attribute, graph, initializer, model, node, tensor};

    #[test]
    fn test_model() {
        // USER INPUT

        let n = 5;
        let c = 1;
        let mut input_data = std::collections::HashMap::new();

        let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
        input_data.insert("X".to_string(), data.as_slice());

        // ONNX INPUTS
        let dims = vec![1, c as i64, n as i64, n as i64];
        let kernel_n = 3;
        let m = 1;
        let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();
        let conv_model = model(graph(
            vec![tensor("X", &dims)],
            vec![tensor("Y", &[1, 1, 3, 3])],
            vec![tensor("W", &[2, c, 3, 3])],
            vec![initializer("W", data_w)],
            vec![node(
                vec!["X", "W"],
                vec!["Y"],
                "conv",
                "Conv",
                vec![attribute("kernel_shape", vec![3, 3])],
            )],
        ));

        // LOGIC

        let session = pollster::block_on(crate::Session::from_model(conv_model))
            .expect("Session did not create");

        let result = pollster::block_on(session.run(input_data)).unwrap();

        assert_eq!(
            result["Y"],
            [54., 63., 72., 99., 108., 117., 144., 153., 162.]
        );
    }
}
