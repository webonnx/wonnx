use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::AttributeProto,
    utils::{attribute, graph, initializer_int64, model, node, tensor},
};
mod common;

fn test_reduce(
    data: &[f32],
    data_shape: &[i64],
    axes: Option<Vec<i64>>,
    op_name: &str,
    keep_dims: bool,
    output: &[f32],
    output_shape: &[i64],
) {
    let mut input_data = HashMap::new();

    input_data.insert("X".to_string(), data.into());

    #[allow(clippy::bool_to_int_with_if)]
    let mut attributes: Vec<AttributeProto> =
        vec![attribute("keepdims", if keep_dims { 1 } else { 0 })];
    if let Some(axes) = axes {
        attributes.push(attribute("axes", axes))
    }

    // Model: X -> ReduceMean -> Y
    let model = model(graph(
        vec![tensor("X", data_shape)],
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(vec!["X"], vec!["Y"], "myReduce", op_name, attributes)],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    log::info!("OUT: {:?}", result["Y"]);
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), output);
}

fn sum_square(a: f32, b: f32) -> f32 {
    ((a * a) + (b * b)).sqrt()
}

#[test]
fn reduce() {
    let _ = env_logger::builder().is_test(true).try_init();

    #[rustfmt::skip]
    let data = [
        5.0, 1.0, 
        20.0, 2.0, 
        
        30.0, 1.0, 
        40.0, 2.0, 
        
        55.0, 1.0,
        60.0, 2.0,
    ];

    #[rustfmt::skip]
    let data_two = [
        20.0, 1.0, 
        5.0, 2.0, 
        
        30.0, 1.0, 
        40.0, 2.0, 
        
        60.0, 1.0,
        55.0, 2.0,
    ];

    // ReduceSum: sum all
    test_reduce(
        &data,
        &[3, 2, 2],
        None, // all
        "ReduceSum",
        false,
        &[219.],
        &[1],
    );

    // ReduceLogSumExp
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![0]),
        "ReduceLogSumExp",
        false,
        &[55., 2.0986123, 60.],
        &[3],
    );

    // ReduceLogSumExp
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![0]),
        "ReduceLogSum",
        false,
        &[4.499_809_7, 1.098_612_4, 4.787_492_3],
        &[3],
    );

    // ONNX test case: do_not_keepdims with ReduceL2
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceL2",
        false,
        &[
            sum_square(20.0, 5.0),
            sum_square(1., 2.),
            sum_square(30.0, 40.0),
            sum_square(1.0, 2.0),
            sum_square(55., 60.),
            sum_square(1., 2.),
        ],
        &[3, 2],
    );

    // ReduceL2 all axes
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![0, 1, 2]),
        "ReduceL2",
        false,
        &[97.800_82],
        &[1],
    );

    // ONNX test case: do_not_keepdims with ReduceL1
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceL1",
        false,
        &[25.0, 3.0, 70., 3., 115., 3.],
        &[3, 2],
    );

    // ONNX test case: do_not_keepdims with ReduceProd
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceProd",
        false,
        &[100., 2., 1200., 2., 3300., 2.],
        &[3, 2],
    );

    // ONNX test case: default_axes_keepdims
    test_reduce(
        &data,
        &[3, 2, 2],
        None,
        "ReduceMean",
        true,
        &[18.25],
        &[1, 1, 1],
    );

    // ONNX test case: do_not_keepdims
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMean",
        false,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 2],
    );

    // ONNX test case: keepdims
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMean",
        true,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 1, 2],
    );

    // ONNX test case: negative_axes_keepdims
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![-2]),
        "ReduceMean",
        true,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 1, 2],
    );

    // ONNX test case: do_not_keepdims with ReduceSum
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceSum",
        false,
        &[25.0, 3.0, 70., 3., 115., 3.],
        &[3, 2],
    );

    // ONNX test case: do_not_keepdims with ReduceMin
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMin",
        false,
        &[5., 1., 30., 1., 55., 1.],
        &[3, 2],
    );

    // ONNX test case: do_not_keepdims with ReduceMax
    test_reduce(
        &data,
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMax",
        false,
        &[20., 2., 40., 2., 60., 2.],
        &[3, 2],
    );

    // ONNX test case: do_not_keepdims with ArgMax
    test_reduce(
        &data_two,
        &[3, 2, 2],
        Some(vec![1]),
        "ArgMax",
        false,
        &[0., 1., 1., 1., 0., 1.],
        &[3, 2],
    );

    // ONNX test case for ReduceSumSquare (https://github.com/onnx/onnx/blob/94e2f64551ded652df53a7e9111031e8aabddaee/onnx/backend/test/case/node/reducesumsquare.py#L27)
    test_reduce(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceSumSquare",
        false,
        &[10., 20., 74., 100., 202., 244.],
        &[3, 2],
    );
}

// Separate test for the case where ReduceSum takes an axes input
// Test case adapted from https://github.com/onnx/onnx/blob/94e2f64551ded652df53a7e9111031e8aabddaee/onnx/backend/test/case/node/reducesum.py#L92
#[test]
fn test_reduce_sum_with_axes_as_input() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut input_data = HashMap::new();

    #[rustfmt::skip]
    let data: &[f32] = &[
       1., 2.,
       3., 4.,

       5., 6.,
       7., 8.,

       9., 10.,
       11., 12.,
    ];

    input_data.insert("X".to_string(), data.into());
    let attributes: Vec<AttributeProto> = vec![attribute("keepdims", 1)];

    // Model: X -> ReduceMean -> Y
    let model = model(graph(
        vec![tensor("X", &[3, 2, 2])],
        vec![tensor("Y", &[3, 2])],
        vec![],
        vec![initializer_int64("A", vec![-2], vec![1])],
        vec![node(
            vec!["X", "A"],
            vec!["Y"],
            "myReduce",
            "ReduceSum",
            attributes,
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    log::info!("OUT: {:?}", result["Y"]);
    common::assert_eq_vector(
        (&result["Y"]).try_into().unwrap(),
        &[4., 6., 12., 14., 20., 22.],
    );
}
