use std::collections::HashMap;
use wonnx::{
    onnx::TensorProto_DataType,
    utils::{graph, model, node, tensor, tensor_of_type, InputTensor},
};

mod common;

#[test]
fn test_cos() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), InputTensor::F32(data.as_slice()));

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

#[test]
fn test_reciprocal() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let reciprocal_data: Vec<f32> = (1..=n).map(|x| 1.0 / (x as f32)).collect();
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), InputTensor::F32(data.as_slice()));

    // Model: X -> Reciprocal -> Y
    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "rec", "Reciprocal", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector(result["Y"].as_slice(), &reciprocal_data);
}

#[test]
fn test_integer() {
    let _ = env_logger::builder().is_test(true).try_init();
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![21i32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), InputTensor::I32(data.as_slice()));

    // Model: X -> Cos -> Y
    let model = model(graph(
        vec![tensor_of_type("X", &shape, TensorProto_DataType::INT32)],
        vec![tensor_of_type("Y", &shape, TensorProto_DataType::INT32)],
        vec![],
        vec![],
        vec![node(vec!["X", "X"], vec!["Y"], "add_ints", "Add", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], vec![42.0; n]);
}
