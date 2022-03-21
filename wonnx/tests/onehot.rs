use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::AttributeProto,
    utils::{attribute, graph, model, node, tensor, InputTensor},
};
mod common;

fn test_onehot(
    indexes: &[i32],
    indexes_shape: &[i64],
    values: &[f32],
    depth: i32,
    axis: Option<i64>,
    output: &[f32],
    output_shape: &[i64],
) {
    let mut input_data = HashMap::<String, InputTensor>::new();

    let depth_tensor: &[i32] = &[depth];
    input_data.insert("I".to_string(), indexes.into());
    input_data.insert("D".to_string(), depth_tensor.into());
    input_data.insert("V".to_string(), values.into());

    let mut attributes: Vec<AttributeProto> = vec![];
    if let Some(axis) = axis {
        attributes.push(attribute("axis", axis))
    }

    // Model: I, D, V -> OneHot -> Y
    let model = model(graph(
        vec![
            tensor("I", indexes_shape),
            tensor("D", &[]),
            tensor("V", &[values.len() as i64]),
        ],
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            vec!["I", "D", "V"],
            vec!["Y"],
            "oneHot",
            "OneHot",
            attributes,
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    log::info!("OUT: {:?}", result["Y"]);
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), output);
}

#[test]
fn onehot() {
    let _ = env_logger::builder().is_test(true).try_init();

    let values = [2., 5.];
    let depth = 5;

    #[rustfmt::skip]
    test_onehot(&[0, 3, 1], &[3], &values, depth, None, &[
		5., 2., 2., 2., 2.,
		2., 2., 2., 5., 2., 
		2., 5., 2., 2., 2.,
	], &[3, depth as i64]);

    #[rustfmt::skip]
    test_onehot(&[-1, 0, -2], &[3], &values, depth, None, &[
		2., 2., 2., 2., 5.,
		5., 2., 2., 2., 2., 
		2., 2., 2., 5., 2.,
	], &[3, depth as i64]);
}
