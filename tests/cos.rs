use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value
use wonnx::utils::tensor;

async fn run() {
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
}

#[test]
fn test_cos() {
    // USER INPUT

    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // ONNX INPUTS

    let input = tensor("X", &dims);

    let output = tensor("Y", &dims);

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Cos".to_string());
    node.set_name("node".to_string());
    node.set_input(protobuf::RepeatedField::from(vec!["X".to_string()]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));

    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let mut session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();
    assert_eq!(result, [1.0; 16]);
}

// #[wasm_bindgen_test]
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
