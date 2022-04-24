use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{attribute, graph, initializer, model, node, tensor};
mod common;

#[test]
fn test_matmul_square_matrix() {
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

    input_data.insert("A".to_string(), data_a.as_slice().unwrap().into());
    input_data.insert("B".to_string(), data_b.as_slice().unwrap().into());

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
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["C"]).try_into().unwrap(), sum.as_slice().unwrap());
}

#[test]
fn test_two_transposes() {
    let mut input_data = HashMap::new();
    let data = (0..2 * 3 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Transpose -> Y -> Transpose -> Z; X==Z
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
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &data);
}

#[test]
fn test_split() {
    let mut input_data = HashMap::new();
    let data = (1..=2 * 6).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

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
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let test_y = vec![1., 2., 3., 7., 8., 9.];
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), &test_y);
    let test_w = vec![4., 5., 6., 10., 11., 12.];
    common::assert_eq_vector((&result["W"]).try_into().unwrap(), &test_w);
}

#[test]
fn test_resize() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut input_data = HashMap::new();
    let data = (1..=2 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

    let downsampling_model = model(graph(
        vec![tensor("X", &[1, 1, 2, 4])],
        vec![tensor("Y", &[1, 1, 1, 2])],
        vec![],
        vec![initializer("scales", vec![1., 1., 0.6, 0.6], vec![4])],
        vec![node(
            vec!["X", "" /* roi */, "scales"],
            vec!["Y"],
            "Resize",
            "Resize",
            vec![attribute("nearest_mode", "floor")],
        )],
    ));

    let session = pollster::block_on(wonnx::Session::from_model(downsampling_model))
        .expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let test_y = vec![1., 3.];
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), &test_y);

    let mut input_data = HashMap::new();
    let data = (1..=4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

    let upsampling_model = model(graph(
        vec![tensor("X", &[1, 1, 2, 2])],
        vec![tensor("Y", &[1, 1, 4, 6])],
        vec![],
        vec![initializer("scales", vec![1., 1., 2., 3.], vec![4])],
        vec![node(
            vec!["X", "" /* roi */, "scales"],
            vec!["Y"],
            "Resize",
            "Resize",
            vec![attribute("nearest_mode", "floor")],
        )],
    ));

    let session = pollster::block_on(wonnx::Session::from_model(upsampling_model))
        .expect("session did not create");
    let _result = pollster::block_on(session.run(&input_data)).unwrap();

    //let test_y = vec![
    //    1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4., 3., 3., 3., 4., 4.,
    //    4.,
    //];
    //assert_eq!(result["Y"], test_y);
}

// Multiply a 2x2 matrix with an identity matrix of size 2x2.
#[test]
fn test_matmul_square_matrix_small() {
    let n = 2;
    let mut input_data = HashMap::new();

    let data_a = ndarray::Array2::eye(n);
    let mut data_b = ndarray::Array2::<f32>::zeros((n, n));
    data_b[[0, 0]] = 0.0;
    data_b[[0, 1]] = 0.5;
    data_b[[1, 0]] = 1.0;
    data_b[[1, 1]] = 1.5;

    let sum = data_a.dot(&data_b);

    input_data.insert("A".to_string(), data_a.as_slice().unwrap().into());
    input_data.insert("B".to_string(), data_b.as_slice().unwrap().into());

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
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["C"]).try_into().unwrap(), sum.as_slice().unwrap());
}

// Multiply a 4x4 matrix with a ones matrix of size 4x2.
// a = np.arange(0,16).reshape((4,4))
// b = np.ones((4,2))
// c = np.matmul(a, b)
// array([[ 6.,  6.], [22., 22.], [38., 38.], [54., 54.]])
#[test]
fn test_matmul_nonsquare_matrix_small() {
    let a_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..8).map(|_| 1.0).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());

    let model = model(graph(
        vec![tensor("A", &[4, 4]), tensor("B", &[4, 2])],
        vec![tensor("C", &[4, 2])],
        vec![],
        vec![],
        vec![node(vec!["A", "B"], vec!["C"], "MatMul", "MatMul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let out = &[6., 6., 22., 22., 38., 38., 54., 54.];
    println!("Result: {:?}", result["C"]);
    common::assert_eq_vector((&result["C"]).try_into().unwrap(), out);
}
