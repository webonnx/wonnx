use protobuf;
use serde::de::Deserialize;
use serde_protobuf::de::Deserializer;
use serde_protobuf::descriptor::Descriptors;
use serde_value::Value;
use std::fs;

use serde_protobuf;

fn main() {
    // Load a descriptor registry (see descriptor module)
    let mut file = crate::onnx::ModelProto::new();
    open("src/single_relu.onnx").unwrap();
    let proto = protobuf::Message::parse_from_reader(&mut file).unwrap();
    let descriptors = Descriptors::from_proto(&proto);
}
