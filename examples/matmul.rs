use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value

async fn run() {
    let steps = execute_gpu().await.unwrap();

    assert_eq!(steps[0..5], [1.0, 1.0, 1.0, 1.0, 1.0]);
    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
    assert_eq!(steps[0..5], [1.0, 1.0, 1.0, 1.0, 1.0]);
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data_a = vec![-1.0f32; n];
    let dims_a = vec![n as i64, n as i64];
    let data_b = vec![-1.0f32; n];
    let dims_b = vec![n as i64, n as i64];
    input_data.insert("A".to_string(), (data_a.as_slice(), dims_a.as_slice()));
    input_data.insert("B".to_string(), (data_b.as_slice(), dims_b.as_slice()));

    // ONNX INPUTS

    let mut shape_tensor_proto_dim = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim.set_dim_value(n as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![shape_tensor_proto_dim]));

    let mut type_proto_tensor = crate::onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(1);
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = crate::onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut input = crate::onnx::ValueInfoProto::new();
    input.set_name("A".to_string());
    input.set_field_type(type_proto.clone());

    let mut input = crate::onnx::ValueInfoProto::new();
    input.set_name("B".to_string());
    input.set_field_type(type_proto.clone());

    let mut output = crate::onnx::ValueInfoProto::new();
    output.set_name("C".to_string());
    output.set_field_type(type_proto.clone());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("MatMul".to_string());
    node.set_name("node".to_string());
    node.set_input(protobuf::RepeatedField::from(vec!["X".to_string(),"X".to_string()]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));

    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    session.run(input_data).await
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
