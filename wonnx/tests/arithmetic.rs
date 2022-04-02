use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::TensorProto_DataType,
    utils::{
        graph, initializer_int64, model, node, tensor, tensor_of_type, InputTensor, OutputTensor,
    },
};

mod common;

#[test]
fn test_cos() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![0.0f32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

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
    assert_eq!(result["Y"], OutputTensor::F32(vec![1.0; 16]));
}

#[test]
fn test_reciprocal() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let reciprocal_data: Vec<f32> = (1..=n).map(|x| 1.0 / (x as f32)).collect();
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

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
    common::assert_eq_vector(
        (&result["Y"]).try_into().unwrap(),
        reciprocal_data.as_slice(),
    );
}

#[test]
fn test_integer() {
    let _ = env_logger::builder().is_test(true).try_init();
    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data = vec![21i32; n];
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), InputTensor::I32(data.as_slice().into()));

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
    assert_eq!(result["Y"], OutputTensor::I32(vec![42; n]));
}

#[test]
fn test_int64_initializers() {
    let _ = env_logger::builder().is_test(true).try_init();
    let n: usize = 16;
    let left: Vec<i64> = (0..n).map(|x| x as i64).collect();
    let right: Vec<i64> = (0..n).map(|x| (x * 2) as i64).collect();
    let sum: Vec<i64> = (0..n).map(|x| (x * 3) as i64).collect();
    let dims = vec![n as i64];

    let model = model(graph(
        vec![tensor_of_type("X", &dims, TensorProto_DataType::INT64)],
        vec![tensor_of_type("Z", &dims, TensorProto_DataType::INT64)],
        vec![],
        vec![initializer_int64("Y", right)],
        vec![node(vec!["X", "Y"], vec!["Z"], "adder", "Add", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let mut input_data: HashMap<String, InputTensor> = HashMap::new();
    input_data.insert("X".to_string(), left.as_slice().into());
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    assert_eq!(result["Z"], OutputTensor::I64(sum))
}

#[test]
fn test_pow() {
    let n: usize = 16;
    let mut input_data = HashMap::new();

    // Output should be 1^0, 2^1, 3^2, 4^3, 5^0, ..
    let x: Vec<f32> = (0..n).map(|x| (x + 1) as f32).collect();
    let y: Vec<f32> = (0..n).map(|x| (x % 4) as f32).collect();

    let shape = vec![n as i64];
    input_data.insert("X".to_string(), x.as_slice().into());
    input_data.insert("Y".to_string(), y.as_slice().into());

    // Model: X,Y -> Pow -> Z
    let model = model(graph(
        vec![tensor("X", &shape), tensor("Y", &shape)],
        vec![tensor("Z", &shape)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "pow", "Pow", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let expected = vec![
        1.0, 2.0, 8.999998, 64.0, 1.0, 6.0, 48.999985, 512.0, 1.0, 10.0, 120.99996, 1727.9989, 1.0,
        14.0, 224.99994, 4096.0,
    ];
    assert_eq!(result["Z"], OutputTensor::F32(expected));
}
