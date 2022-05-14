use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{attribute, graph, initializer, initializer_int64, model, node, tensor};
mod common;

#[test]
fn test_matmul_square_matrix() {
    let _ = env_logger::builder().is_test(true).try_init();
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

fn test_transpose_4d_perm(transpose_first: &[i64], transpose_second: &[i64]) {
    let mut input_data = HashMap::new();
    let data = (0..2 * 3 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

    let x_dims = vec![1, 2, 3, 4];
    let intermediate_dims: Vec<i64> = transpose_first
        .iter()
        .map(|i| x_dims[*i as usize])
        .collect();

    // Model: X -> Transpose -> Y -> Transpose -> Z; X==Z
    let model = model(graph(
        vec![tensor("X", &x_dims)],
        vec![tensor("Z", &x_dims)],
        vec![tensor("Y", &intermediate_dims)],
        vec![],
        vec![
            node(
                vec!["X"],
                vec!["Y"],
                "Transpose",
                "Transpose",
                vec![attribute("perm", transpose_first.to_vec())],
            ),
            node(
                vec!["Y"],
                vec!["Z"],
                "Transpose",
                "Transpose",
                vec![attribute("perm", transpose_second.to_vec())],
            ),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &data);
}

/* This tests the equivalent of the following Python code:
a = np.arange(0,24).reshape((1,2,3,4));
a == a.transpose(a).transpose(inverse of a)
*/
#[test]
fn test_two_transposes_4d() {
    // a == a.transpose([0,2,1,3]).transpose([0,2,1,3])
    test_transpose_4d_perm(&[0, 2, 1, 3], &[0, 2, 1, 3]);

    // a == a.transpose([0,2,3,1]).transpose([0,3,1,2])
    test_transpose_4d_perm(&[0, 2, 3, 1], &[0, 3, 1, 2]);

    // a == a.transpose([0,3,2,1]).transpose([0,3,2,1])
    test_transpose_4d_perm(&[0, 3, 2, 1], &[0, 3, 2, 1]);
}

// When no 'perm' attribute is specified, Transpose should reverse axes (e.g. for 4D input, assume perm is [3,2,1,0])
#[test]
fn test_two_transposes_default_4d() {
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
            node(vec!["X"], vec!["Y"], "Transpose", "Transpose", vec![]),
            node(vec!["Y"], vec!["Z"], "Transpose", "Transpose", vec![]),
        ],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &data);
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
    let _ = env_logger::builder().is_test(true).try_init();
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
            vec![attribute("axis", -1)],
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
fn test_pad_example() {
    let mut input_data = HashMap::new();
    #[rustfmt::skip]
    let data = [
        1.0, 1.2,
        2.3, 3.4,
        4.5, 5.7,
    ].to_vec();
    input_data.insert("X".to_string(), data.as_slice().into());

    let model = model(graph(
        vec![tensor("X", &[3, 2])],
        vec![tensor("Y", &[3, 4])],
        vec![],
        vec![initializer_int64("pads", vec![0, 2, 0, 0], vec![4])],
        vec![node(vec!["X", "pads"], vec!["Y"], "Pad", "Pad", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    #[rustfmt::skip]
    let test_y = vec![
        0.0, 0.0, 1.0, 1.2,
        0.0, 0.0, 2.3, 3.4,
        0.0, 0.0, 4.5, 5.7,
    ];
    let actual: &[_] = (&result["Y"]).try_into().unwrap();
    // No arithmetic is done, so `assert_eq!` can be used.
    assert_eq!(actual, &test_y);
}

#[test]
fn test_pad_complex() {
    let mut input_data = HashMap::new();
    let data = (1..=3 * 2).map(|x| x as f32).collect::<Vec<f32>>();
    input_data.insert("X".to_string(), data.as_slice().into());

    let kv = 0.5;
    let model = model(graph(
        vec![tensor("X", &[1, 3, 2])],
        vec![tensor("Y", &[2, 4, 5])],
        vec![],
        vec![],
        vec![node(
            vec!["X"],
            vec!["Y"],
            "Pad",
            "Pad",
            vec![
                attribute(
                    "pads",
                    vec![
                        0, // x1_begin
                        1, // x2_begin
                        2, // x3_begin
                        1, // x1_end
                        0, // x2_end
                        1, // x3_end
                    ],
                ),
                attribute("constant_value", kv),
            ],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    /* [[[1,2], [3,4], [5,6]]] ->
    [
        [[0,0,0,0,0],
        [0,0,1,2,0],
        [0,0,3,4,0],
        [0,0,5,6,0]],
        [[0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]]
    ]
    */

    #[rustfmt::skip]
    let test_y = vec![
        kv, kv, kv, kv, kv,
        kv, kv, 1., 2., kv,
        kv, kv, 3., 4., kv,
        kv, kv, 5., 6., kv,

        kv, kv, kv, kv, kv,
        kv, kv, kv, kv, kv,
        kv, kv, kv, kv, kv,
        kv, kv, kv, kv, kv,
    ];
    let actual: &[_] = (&result["Y"]).try_into().unwrap();

    // No arithmetic is done, so `assert_eq!` can be used.
    assert_eq!(actual, &test_y);
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
    let _ = env_logger::builder().is_test(true).try_init();
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
// b = np.arange(-8,0).reshape((4,2))
// c = np.matmul(a, b)
//array([[ -20,  -14], [-100,  -78], [-180, -142], [-260, -206]])
#[test]
fn test_matmul_nonsquare_matrix_small() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..8).map(|x| -8.0 + (x as f32)).collect();

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

    let out = &[-20., -14., -100., -78., -180., -142., -260., -206.];
    common::assert_eq_vector((&result["C"]).try_into().unwrap(), out);
}

// Multiply two stacks of 2x2 matrixes
// a = np.arange(0,16).reshape((4,2,2))
// b = np.arange(16,32).reshape((4,2,2))
// c = np.matmul(a, b)
#[test]
fn test_matmul_stacks() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (16..32).map(|x| x as f32).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());

    let model = model(graph(
        vec![tensor("A", &[4, 2, 2]), tensor("B", &[4, 2, 2])],
        vec![tensor("C", &[4, 2, 2])],
        vec![],
        vec![],
        vec![node(vec!["A", "B"], vec!["C"], "MatMul", "MatMul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let out: Vec<f32> = vec![
        18, 19, 86, 91, 190, 199, 274, 287, 426, 443, 526, 547, 726, 751, 842, 871,
    ]
    .iter()
    .map(|x| *x as f32)
    .collect();
    common::assert_eq_vector((&result["C"]).try_into().unwrap(), &out);
}

// Multiply a 1x4 matrix with a ones matrix of size 4x2.
// a = np.arange(0,4).reshape((1,4))
// b = np.arange(-8,0).reshape((4,2))
// c = np.matmul(a, b)
//array([[ -20,  -14]])
#[test]
fn test_matmul_1d() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..4).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..8).map(|x| -8.0 + (x as f32)).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());

    let model = model(graph(
        vec![tensor("A", &[1, 4]), tensor("B", &[4, 2])],
        vec![tensor("C", &[1, 2])],
        vec![],
        vec![],
        vec![node(vec!["A", "B"], vec!["C"], "MatMul", "MatMul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let out = &[-20., -14.];
    common::assert_eq_vector((&result["C"]).try_into().unwrap(), out);
}

// Test Gemm with matrix bias
// a = np.arange(24).reshape((4,6))
// b = np.arange(24).reshape((6,4))
// c = np.arange(16).reshape((4,4))
// d = np.dot(a,b) + c
// d = array([[ 220,  236,  252,  268], [ 584,  636,  688,  740], [ 948, 1036, 1124, 1212], [1312, 1436, 1560, 1684]])
#[test]
fn test_gemm_matrix_bias() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let c_data: Vec<f32> = (0..16).map(|x| x as f32).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());
    input_data.insert("C".to_string(), c_data.as_slice().into());

    let model = model(graph(
        vec![
            tensor("A", &[4, 6]),
            tensor("B", &[6, 4]),
            tensor("C", &[4, 4]),
        ],
        vec![tensor("D", &[4, 4])],
        vec![],
        vec![],
        vec![node(vec!["A", "B", "C"], vec!["D"], "Gemm", "Gemm", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let out = &[
        220., 236., 252., 268., 584., 636., 688., 740., 948., 1036., 1124., 1212., 1312., 1436.,
        1560., 1684.,
    ];
    common::assert_eq_vector((&result["D"]).try_into().unwrap(), out);
}

// Test Gemm with broadcasting bias
// a = np.arange(24).reshape((4,6))
// b = np.arange(24).reshape((6,4))
// c = np.arange(4).reshape((1,4))
// d = np.dot(a,b) + c
// d = array([[ 220,  236,  252,  268], [ 580,  632,  684,  736], [ 940, 1028, 1116, 1204], [1300, 1424, 1548, 1672]])
#[test]
fn test_gemm_broadcasting_bias() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let c_data: Vec<f32> = (0..4).map(|x| x as f32).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());
    input_data.insert("C".to_string(), c_data.as_slice().into());

    let model = model(graph(
        vec![
            tensor("A", &[4, 6]),
            tensor("B", &[6, 4]),
            tensor("C", &[1, 4]),
        ],
        vec![tensor("D", &[4, 4])],
        vec![],
        vec![],
        vec![node(vec!["A", "B", "C"], vec!["D"], "Gemm", "Gemm", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let out = &[
        220., 236., 252., 268., 580., 632., 684., 736., 940., 1028., 1116., 1204., 1300., 1424.,
        1548., 1672.,
    ];
    common::assert_eq_vector((&result["D"]).try_into().unwrap(), out);
}

// Test Gemm with broadcasting bias (same as above but bias is shaped (4,1) instead of (1,4))
// a = np.arange(24).reshape((4,6))
// b = np.arange(24).reshape((6,4))
// c = np.arange(4).reshape((1,4))
// d = np.dot(a,b) + c
// d = array([[ 220,  235,  250,  265], [ 581,  632,  683,  734], [ 942, 1029, 1116, 1203], [1303, 1426, 1549, 1672]])
#[test]
fn test_gemm_broadcasting_second_bias() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let c_data: Vec<f32> = (0..4).map(|x| x as f32).collect();

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());
    input_data.insert("C".to_string(), c_data.as_slice().into());

    let model = model(graph(
        vec![
            tensor("A", &[4, 6]),
            tensor("B", &[6, 4]),
            tensor("C", &[4, 1]),
        ],
        vec![tensor("D", &[4, 4])],
        vec![],
        vec![],
        vec![node(vec!["A", "B", "C"], vec!["D"], "Gemm", "Gemm", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let out = &[
        220., 235., 250., 265., 581., 632., 683., 734., 942., 1029., 1116., 1203., 1303., 1426.,
        1549., 1672.,
    ];
    common::assert_eq_vector((&result["D"]).try_into().unwrap(), out);
}

// Test Gemm with broadcasting bias
// a = np.arange(24).reshape((4,6))
// b = np.arange(24).reshape((6,4))
// c = np.array(-1000)
// d = np.dot(a,b) + c
// d = array([[-780, -765, -750, -735], [-420, -369, -318, -267], [ -60,   27,  114,  201], [ 300,  423,  546,  669]])
#[test]
fn test_gemm_scalar_bias() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let c_data: Vec<f32> = vec![-1000.0];

    let mut input_data = HashMap::new();
    input_data.insert("A".to_string(), a_data.as_slice().into());
    input_data.insert("B".to_string(), b_data.as_slice().into());
    input_data.insert("C".to_string(), c_data.as_slice().into());

    let model = model(graph(
        vec![
            tensor("A", &[4, 6]),
            tensor("B", &[6, 4]),
            tensor("C", &[1]),
        ],
        vec![tensor("D", &[4, 4])],
        vec![],
        vec![],
        vec![node(vec!["A", "B", "C"], vec!["D"], "Gemm", "Gemm", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let out = &[
        -780., -765., -750., -735., -420., -369., -318., -267., -60., 27., 114., 201., 300., 423.,
        546., 669.,
    ];
    common::assert_eq_vector((&result["D"]).try_into().unwrap(), out);
}
