use std::collections::HashMap;
use wonnx::utils::{graph, model, node, tensor};
mod common;

#[test]
fn test_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

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
    common::assert_eq_vector(result["Y"].unwrap_f32_slice(), &data);
}

#[test]
fn test_double_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

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
    common::assert_eq_vector(result["Z"].unwrap_f32_slice(), &data);
}
