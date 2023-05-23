use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx_model::{onnx_graph, onnx_model, onnx_node, onnx_tensor},
    tensor::TensorData,
};

mod common;

#[test]
fn test_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Identity -> Y; Y==Z
    let model = onnx_model(onnx_graph(
        vec![onnx_tensor("X", &dims)],
        vec![onnx_tensor("Y", &dims)],
        vec![],
        vec![],
        vec![onnx_node(vec!["X"], vec!["Y"], "Identity", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), &data);
}

#[test]
fn test_double_identity() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let dims = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Identity -> Y -> Identity -> Z. X==Z
    let model = onnx_model(onnx_graph(
        vec![onnx_tensor("X", &dims)],
        vec![onnx_tensor("Z", &dims)],
        vec![onnx_tensor("Y", &dims)],
        vec![],
        vec![
            onnx_node(vec!["X"], vec!["Y"], "Identity", vec![]),
            onnx_node(vec!["Y"], vec!["Z"], "Identity", vec![]),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &data);
}

#[test]
fn test_buffer_readability() {
    let _ = env_logger::builder().is_test(true).try_init();
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Cos -> Y -> Flatten -> Z -> Flatten -> W
    let model = onnx_model(onnx_graph(
        vec![onnx_tensor("X", &shape)],
        vec![onnx_tensor("W", &shape)],
        vec![onnx_tensor("Y", &shape), onnx_tensor("Z", &shape)],
        vec![],
        vec![
            onnx_node(vec!["X"], vec!["Y"], "Cos", vec![]),
            onnx_node(vec!["Y"], vec!["Z"], "Reshape", vec![]),
            onnx_node(vec!["Z"], vec!["W"], "Reshape", vec![]),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["W"], TensorData::F32(vec![1.0; 16].into()));
}
