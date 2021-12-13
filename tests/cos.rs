use std::collections::HashMap;
// use wasm_bindgen_test::*;
// Indicates a f32 overflow in an intermediate Collatz value

use wonnx::utils::{graph, model, node, tensor};

#[test]
fn test_cos() {
    // USER INPUT

    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // ONNX INPUTS
    let model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Y", &dims)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "cos", "Cos", vec![])],
    ));

    // LOGIC

    let mut session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();
    assert_eq!(result["Y"], [1.0; 16]);
}
