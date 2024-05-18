use approx::assert_abs_diff_eq;
use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::TensorProto_DataType,
    utils::{
        graph, initializer, initializer_int64, model, node, tensor, tensor_of_type, InputTensor,
        OutputTensor,
    },
};

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

mod common;
#[test]
fn test_erf() {
    let n: usize = 16;
    let mut input_data = HashMap::new();
    let data = ndarray::Array1::<f32>::random(n, Uniform::new(-100f32, 100f32));

    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().unwrap().into());

    // Model: X -> Cos -> Y
    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "erf", "Erf", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    if let OutputTensor::F32(vec) = &result["Y"] {
        for e in vec {
            assert_ne!(*e, f32::NAN);
        }
    }
}
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

    // Model: X -> Add -> Y
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
        vec![initializer_int64("Y", right, dims.clone())],
        vec![node(vec!["X", "Y"], vec!["Z"], "adder", "Add", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let mut input_data: HashMap<String, InputTensor> = HashMap::new();
    input_data.insert("X".to_string(), left.as_slice().into());
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    assert_eq!(result["Z"], OutputTensor::I64(sum))
}

pub fn assert_eq_vector_weak(xs: &[f32], ys: &[f32]) {
    assert_eq!(xs.len(), ys.len());
    for i in 0..xs.len() {
        assert_abs_diff_eq!(xs[i], ys[i], epsilon = 0.5);
    }
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

    // The pow(x,y) function in WGSL appears to be rather imprecise. Therefore we use a weaker comparison here (for now).
    assert_eq_vector_weak((&result["Z"]).try_into().unwrap(), expected.as_slice());
}

#[test]
fn test_mul_broadcast() {
    let _ = env_logger::builder().is_test(true).try_init();
    let x = vec![2.0, 3.0];
    let y = vec![10.0, 20.0, 30.0];
    let shape_x = [1, 2];
    let shape_y = [1, 3, 1];
    let shape_z = [1, 3, 2];

    let mut input_data = HashMap::new();
    input_data.insert("X".to_string(), x.as_slice().into());
    input_data.insert("Y".to_string(), y.as_slice().into());

    // Model: X,Y -> Mul -> Z
    let model = model(graph(
        vec![tensor("X", &shape_x), tensor("Y", &shape_y)],
        vec![tensor("Z", &shape_z)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "mul", "Mul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    // np.array([[2,3]]) * np.array([[10],[20],[30]]) => [[20, 30], [40, 60], [60, 90]]
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let expected = vec![20.0, 30.0, 40.0, 60.0, 60.0, 90.0];
    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), expected.as_slice());
}

#[test]
fn test_prelu() {
    fn test(
        data: &[f32],
        data_shape: &[i64],
        slope: &[f32],
        slope_shape: &[i64],
        expected: &[f32],
    ) {
        let mut input_data = HashMap::new();
        input_data.insert("X".to_string(), data.into());
        input_data.insert("Y".to_string(), slope.into());

        let model = model(graph(
            vec![tensor("X", data_shape), tensor("Y", slope_shape)],
            vec![tensor("Z", data_shape)],
            vec![],
            vec![],
            vec![node(vec!["X", "Y"], vec!["Z"], "prelu", "PRelu", vec![])],
        ));

        let session =
            pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
        let result = pollster::block_on(session.run(&input_data)).unwrap();
        common::assert_eq_vector((&result["Z"]).try_into().unwrap(), expected);
    }

    test(&[1.0], &[1], &[1.0], &[1], &[1.0]);
    test(&[-1.0], &[1], &[0.5], &[1], &[-0.5]);
    test(
        &[-1.0, -0.5, 0.0, 1.0],
        &[4],
        &[0.5, 2.0, 100.0, 100.0],
        &[4],
        &[-0.5, -1.0, 0.0, 1.0],
    );

    // Broadcast tests:
    test(
        &[-1.0, -0.5, 0.0, 1.0],
        &[4],
        &[0.5],
        &[1],
        &[-0.5, -0.25, 0.0, 1.0],
    );
    test(
        &[-1.0, -1.0, -1.0, -1.0],
        &[2, 2],
        &[0.5, 2.0],
        &[1, 2],
        &[-0.5, -2.0, -0.5, -2.0],
    );
    test(
        &[-1.0, -1.0, -1.0, -1.0],
        &[2, 2],
        &[0.5, 2.0],
        &[2, 1],
        &[-0.5, -0.5, -2.0, -2.0],
    );
}

#[test]
fn test_sign() {
    let _ = env_logger::builder().is_test(true).try_init();
    let n: usize = 7;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..=(n - 1))
        .map(|x| ((x as i64) - (n / 2) as i64) as f32)
        .collect();
    let shape = vec![n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> Cos -> Y
    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "sign", "Sign", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let expected: Vec<f32> = data
        .iter()
        .map(|x| {
            if *x == 0.0 {
                0.0
            } else if *x < 0.0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], OutputTensor::F32(expected));
}

#[test]
fn test_clip() {
    // Model: X -> Clip -> Y
    let shape = vec![1, 1, 2, 2];
    let model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![
            initializer("min", vec![0.0], vec![]),
            initializer("max", vec![1.0], vec![]),
        ],
        vec![node(
            vec!["X", "min", "max"],
            vec!["Y"],
            "clip",
            "Clip",
            vec![],
        )],
    ));
    let mut input_data = HashMap::new();
    input_data.insert(
        "X".to_string(),
        InputTensor::F32([-1.0, 0.0, 1.0, 2.0].as_slice().into()),
    );

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], OutputTensor::F32(vec![0.0, 0.0, 1.0, 1.0]));
}
