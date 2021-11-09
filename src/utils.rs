use crate::onnx;
use std::collections::HashMap;

pub fn len(dims: &[i64]) -> i64 {
    dims.iter().product::<i64>()
}

pub fn get_attribute<'a>(
    attribute: &'a str,
    defaults: Option<&'a onnx::AttributeProto>,
    node: &'a onnx::NodeProto,
) -> &'a onnx::AttributeProto {
    match defaults {
        Some(default) => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .unwrap_or(default),
        None => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .unwrap_or_else(|| {
                panic!(
                    "Did not find required attribute: {}, for node: {}",
                    attribute,
                    node.get_name()
                )
            }),
    }
}

pub fn get_dimension(value_info: &[onnx::ValueInfoProto], input_name: &str) -> Option<Vec<i64>> {
    if let Some(info) = value_info.iter().find(|x| x.get_name() == input_name) {
        Some(
            info.get_field_type()
                .get_tensor_type()
                .get_shape()
                .get_dim()
                .iter()
                .map(|x| x.get_dim_value())
                .collect(),
        )
    } else {
        None
    }
}
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

pub fn initializer(name: &str, data: Vec<f32>, dimensions: &[i64]) -> onnx::TensorProto {
    let mut initializer = crate::onnx::TensorProto::new();
    initializer.set_name(name.to_string());
    initializer.set_float_data(data);
    initializer.set_data_type(1);
    initializer.set_dims(dimensions.to_vec());
    initializer
}

pub trait ToAttribute {
    fn to_attribute(self) -> onnx::AttributeProto
    where
        Self: Sized;
}

impl ToAttribute for Vec<i64> {
    fn to_attribute(self) -> onnx::AttributeProto {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_ints(self);
        attributes
    }
}

impl ToAttribute for String {
    fn to_attribute(self) -> onnx::AttributeProto {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(self.into_bytes());
        attributes
    }
}

impl ToAttribute for &str {
    fn to_attribute(self) -> onnx::AttributeProto {
        let mut attributes = crate::onnx::AttributeProto::new();
        attributes.set_s(self.to_string().into_bytes());
        attributes
    }
}

pub fn attribute(name: &str, inputs: impl ToAttribute) -> onnx::AttributeProto {
    let mut attributes = inputs.to_attribute();
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
    initializers: Vec<onnx::TensorProto>,
    nodes: Vec<onnx::NodeProto>,
) -> onnx::GraphProto {
    let mut graph = onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(nodes));
    graph.set_input(protobuf::RepeatedField::from(inputs));
    graph.set_output(protobuf::RepeatedField::from(outputs));
    graph.set_initializer(protobuf::RepeatedField::from(initializers));
    graph
}

pub fn model(graph: onnx::GraphProto) -> onnx::ModelProto {
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);
    model
}

#[test]
fn test_model() {
    // USER INPUT

    let n = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..50).map(|x| x as f32).collect();
    let dims = vec![2, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let data_w: Vec<f32> = (0..2 * c * 3 * 3).map(|_| 1 as f32).collect();

    let conv_model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Y", &[2, 2, n, n])],
        vec![initializer("W", data_w, &[2, c, 3, 3])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "conv",
            "Conv",
            vec![
                attribute("kernel_shape", vec![3, 3]),
                attribute("auto_pad", "SAME_UPPER"),
            ],
        )],
    ));

    // LOGIC

    let mut session =
        pollster::block_on(crate::Session::from_model(conv_model)).expect("Session did not create");

    let result = pollster::block_on(crate::run(&mut session, input_data)).unwrap();

    assert_eq!(
        result,
        [
            12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0,
            81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 12.0, 21.0,
            27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0, 81.0, 93.0,
            144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 112.0, 171.0, 177.0,
            183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0, 231.0,
            243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0, 112.0, 171.0,
            177.0, 183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0,
            231.0, 243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0
        ]
    );
}
