// use approx::assert_relative_eq;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::utils::{attribute, graph, model, node, tensor};
// Indicates a f32 overflow in an intermediate Collatz value
use std::time::Instant;

#[test]
fn test_matmul_square_matrix() {
    // USER INPUT

    let n = 16;
    let mut input_data = HashMap::new();

    let data_a = ndarray::Array2::eye(n);
    let mut data_b = ndarray::Array2::<f32>::zeros((n, n));
    data_b[[0, 0]] = 0.0;
    data_b[[0, 1]] = 0.5;
    data_b[[0, 2]] = -0.5;
    data_b[[1, 0]] = 1.0;
    data_b[[1, 1]] = 1.5;
    data_b[[1, 2]] = -1.5;
    data_b[[2, 0]] = 2.0;
    data_b[[2, 1]] = 2.5;
    data_b[[2, 2]] = -2.5;

    let sum = data_a.dot(&data_b);

    input_data.insert("A".to_string(), data_a.as_slice().unwrap());
    input_data.insert("B".to_string(), data_b.as_slice().unwrap());

    let n = n as i64;
    let model = model(graph(
        vec![tensor("A", &[n, n]), tensor("B", &[n, n])],
        vec![tensor("C", &[n, n])],
        vec![],
        vec![],
        vec![node(vec!["A", "B"], vec!["C"], "MatMul", "MatMul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(input_data)).unwrap();

    assert_eq!(result["C"].as_slice(), sum.as_slice().unwrap());
}

#[test]
fn test_two_transposes() {
    // USER INPUT

    let mut input_data = HashMap::new();
    let data = (0..2 * 3 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice());

    let model = model(graph(
        vec![tensor("X", &[2, 3, 4])],
        vec![tensor("Z", &[2, 3, 4])],
        vec![tensor("Y", &[4, 3, 2])],
        vec![],
        vec![
            node(
                vec!["X"],
                vec!["Y"],
                "Transpose",
                "Transpose",
                vec![attribute("perm", vec![2, 1, 0])],
            ),
            node(
                vec!["Y"],
                vec!["Z"],
                "Transpose",
                "Transpose",
                vec![attribute("perm", vec![2, 1, 0])],
            ),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(input_data)).unwrap();

    assert_eq!(result["Z"], data);
}
