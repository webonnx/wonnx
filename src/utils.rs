use crate::onnx;

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
