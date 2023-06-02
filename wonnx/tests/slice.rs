use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::AttributeProto,
    utils::{attribute, graph, model, node, tensor},
};
mod common;

fn assert_slice(
    data: &[f32],
    data_shape: &[i64],
    output: &[f32],
    output_shape: &[i64],
    starts: &[i64],
    ends: &[i64],
    axes: Option<Vec<i64>>,
    steps: Option<Vec<i64>>
) {
    let mut input_data = HashMap::new();

    input_data.insert("X".to_string(), data.into());

    let mut attributes: Vec<AttributeProto> = vec![
        attribute("starts", starts.to_vec()),
        attribute("ends", ends.to_vec())
    ];
    if let Some(axes) = axes {
        attributes.push(attribute("axes", axes));
    }
    if let Some(steps) = steps {
        attributes.push(attribute("steps", steps));
    }

    // Model: (X) -> Slice -> Y
    let bn_model = model(graph(
        vec![tensor("X", data_shape)],
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            vec!["X"],
            vec!["Y"],
            "mySlice",
            "Slice",
            attributes
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(bn_model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), output);
}

#[test]
fn slice() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Example 1 from https://onnx.ai/onnx/operators/onnx__Slice.html#slice.
    assert_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &[5.0, 7.0],
        &[2 ,1],
        &[1, 0],
        &[2, 3],
        Some(vec![0, 1]),
        Some(vec![1, 2])
    );
}
