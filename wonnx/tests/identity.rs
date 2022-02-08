use std::collections::HashMap;
// use wasm_bindgen_test::*;
// Indicates a f32 overflow in an intermediate Collatz value

use wonnx::utils::{graph, model, node, tensor};

#[test]
fn test_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // Model: X -> Identity -> Y; Y==Z
    let model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Y", &dims)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "a", "Identity", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], data);
}

#[test]
fn test_double_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // Model: X -> Identity -> Y -> Identity -> Z. X==Z
    let model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Z", &dims)],
        vec![tensor("Y", &dims)],
        vec![],
        vec![
            node(vec!["X"], vec!["Y"], "a", "Identity", vec![]),
            node(vec!["Y"], vec!["Z"], "b", "Identity", vec![]),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Z"], data);
}
