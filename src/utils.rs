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

pub fn get_dimension(inputs_onnx: &[onnx::ValueInfoProto], input_name: &str) -> Vec<i64> {
    inputs_onnx
        .iter()
        .find(|x| x.get_name() == input_name)
        .unwrap_or_else(|| panic!("Dimensions for input: {} was not found", input_name))
        .get_field_type()
        .get_tensor_type()
        .get_shape()
        .get_dim()
        .iter()
        .map(|x| x.get_dim_value())
        .collect()
}
