use std::collections::HashMap;
use wonnx::{
    onnx::AttributeProto,
    utils::{attribute, graph, model, node, tensor},
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
    common::assert_eq_vector(result["Y"].as_slice(), output);
}

#[test]
fn reduce() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ONNX test case: default_axes_keepdims
    test_reduce(
        &[
            5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
        ],
        &[3, 2, 2],
        None,
        "ReduceMean",
        true,
        &[18.25],
        &[1, 1, 1],
    );

    // ONNX test case: do_not_keepdims
    test_reduce(
        &[
            5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
        ],
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMean",
        false,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 2],
    );

    // ONNX test case: keepdims
    test_reduce(
        &[
            5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
        ],
        &[3, 2, 2],
        Some(vec![1]),
        "ReduceMean",
        true,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 1, 2],
    );

    // ONNX test case: negative_axes_keepdims
    test_reduce(
        &[
            5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
        ],
        &[3, 2, 2],
        Some(vec![-2]),
        "ReduceMean",
        true,
        &[12.5, 1.5, 35., 1.5, 57.5, 1.5],
        &[3, 1, 2],
    );
}
