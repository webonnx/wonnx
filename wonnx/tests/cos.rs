use std::collections::HashMap;
use wonnx::utils::{graph, model, node, tensor};

#[test]
fn test_cos() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // Model: X -> Cos -> Y
    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "cos", "Cos", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], [1.0; 16]);
}
