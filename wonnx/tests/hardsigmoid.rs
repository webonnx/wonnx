use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{attribute, graph, model, node, tensor};
mod common;

/// Test HardSigmoid node with default alpha and beta
/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-68
#[test]
fn test_hardsigmoid_default() {
    let input_data = [-2.0, -1.0, 1.0, 2.0];
    let shape = vec![2, 2];

    let (default_alpha, default_beta) = (0.2, 0.5);
    let expected_output_data: Vec<f32> = input_data
        .iter()
        .map(|x| x * default_alpha + default_beta)
        .collect();

    let mut model_input = HashMap::new();
    model_input.insert("X".to_string(), input_data.as_slice().into());

    let node = node(vec!["X"], vec!["Y"], "hard_sigmoid", "HardSigmoid", vec![]);

    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let output = pollster::block_on(session.run(&model_input)).unwrap();
    let output_data: &[f32] = (&output["Y"]).try_into().unwrap();

    common::assert_eq_vector(output_data, expected_output_data.as_slice());
}

/// Test HardSigmoid node with predefined alpha and beta
/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-68
#[test]
fn test_hardsigmoid() {
    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let shape = vec![1, 3];

    let mut model_input = HashMap::new();
    model_input.insert("X".to_string(), input_data.as_slice().into());

    let alpha = attribute("alpha", 0.5);
    let beta = attribute("beta", 0.6);

    let node = node(
        vec!["X"],
        vec!["Y"],
        "hard_sigmoid",
        "HardSigmoid",
        vec![alpha, beta],
    );

    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let output = pollster::block_on(session.run(&model_input)).unwrap();
    println!("{:?}", output);

    let expected_output = &[0.1, 0.6, 1.0];
    let output_data: &[f32] = (&output["Y"]).try_into().unwrap();
    common::assert_eq_vector(output_data, expected_output);
}
