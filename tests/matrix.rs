// use approx::assert_relative_eq;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::utils::{attribute, graph, initializer, model, node, tensor};
// Indicates a f32 overflow in an intermediate Collatz value

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

#[test]
fn test_split() {
    // USER INPUT

    let mut input_data = HashMap::new();
    let data = (1..=2 * 6).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice());

    let model = model(graph(
        vec![tensor("X", &[2, 6])],
        vec![tensor("Y", &[2, 3]), tensor("W", &[2, 3])],
        vec![],
        vec![],
        vec![node(
            vec!["X"],
            vec!["Y", "W"],
            "Split",
            "Split",
            vec![attribute("axis", 1)],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(input_data)).unwrap();

    let test_y = vec![1., 2., 3., 7., 8., 9.];
    assert_eq!(result["Y"], test_y);
    let test_w = vec![4., 5., 6., 10., 11., 12.];
    assert_eq!(result["W"], test_w);
}

#[test]
fn test_resize() {
    // USER INPUT

    let mut input_data = HashMap::new();
    let data = (1..=2 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice());

    let downsampling_model = model(graph(
        vec![tensor("X", &[1, 1, 2, 4])],
        vec![tensor("Y", &[1, 1, 1, 2])],
        vec![],
        vec![initializer("scales", vec![1., 1., 0.6, 0.6])],
        vec![node(
            vec!["X", "scales"],
            vec!["Y"],
            "Resize",
            "Resize",
            vec![attribute("nearest_mode", "floor")],
        )],
    ));

    let session = pollster::block_on(wonnx::Session::from_model(downsampling_model))
        .expect("session did not create");
    let result = pollster::block_on(session.run(input_data)).unwrap();

    let test_y = vec![1., 3., 0., 0.];
    assert_eq!(result["Y"], test_y);

    let mut input_data = HashMap::new();
    let data = (1..=4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice());

    let upsampling_model = model(graph(
        vec![tensor("X", &[1, 1, 2, 2])],
        vec![tensor("Y", &[1, 1, 4, 6])],
        vec![],
        vec![initializer("scales", vec![1., 1., 2., 3.])],
        vec![node(
            vec!["X", "scales"],
            vec!["Y"],
            "Resize",
            "Resize",
            vec![attribute("nearest_mode", "floor")],
        )],
    ));

    let session = pollster::block_on(wonnx::Session::from_model(upsampling_model))
        .expect("session did not create");
    let _result = pollster::block_on(session.run(input_data)).unwrap();

    //let test_y = vec![
    //    1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4., 3., 3., 3., 4., 4.,
    //    4.,
    //];
    //assert_eq!(result["Y"], test_y);
}
