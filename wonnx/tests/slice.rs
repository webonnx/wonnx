use std::{collections::HashMap, convert::TryInto};
use wonnx::{
    onnx::ValueInfoProto,
    utils::{graph, model, node, tensor},
};
mod common;

fn assert_slice(
    data: &[f32],
    data_shape: &[i64],
    output: &[f32],
    output_shape: &[i64],
    starts: &[f32],
    ends: &[f32],
    axes: Option<Vec<f32>>,
    steps: Option<Vec<f32>>
) {
    let mut input_data = HashMap::new();
    let mut input_shapes: Vec<ValueInfoProto> = vec![];
    let mut input_names: Vec<&str> = vec![];

    let starts_lengths = vec![starts.len() as i64];
    let ends_lengths = vec![ends.len() as i64];

    input_data.insert("X".to_string(), data.into());
    input_shapes.push(tensor("X", data_shape));
    input_names.push("X");

    input_data.insert("S".to_string(), starts.into());
    input_shapes.push(tensor("S", &starts_lengths[..]));
    input_names.push("S");

    input_data.insert("E".to_string(), ends.into());
    input_shapes.push(tensor("E", &ends_lengths[..]));
    input_names.push("E");

    if let Some(axes) = axes {
        let axes_lengths = vec![axes.len() as i64];
        input_data.insert("A".to_string(), (&axes[..]).into()); // TODO: Lifetime issues
        input_shapes.push(tensor("A", &axes_lengths[..]));
        input_names.push("A");
    }

    if let Some(steps) = steps {
        let steps_lengths = vec![steps.len() as i64];
        input_data.insert("P".to_string(), (&steps[..]).into()); // TODO: Lifetime issues
        input_shapes.push(tensor("P", &steps_lengths[..]));
        input_names.push("P");
    }

    // Model: (X, S, E, A?, P?) -> Slice -> Y
    let bn_model = model(graph(
        input_shapes,
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            input_names,
            vec!["Y"],
            "mySlice",
            "Slice",
            vec![]
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
        &[1., 2., 3., 4., 5., 6., 7., 8.],
        &[4, 2],
        &[5., 7.],
        &[2, 1],
        &[1., 0.],
        &[2., 3.],
        Some(vec![0., 1.]),
        Some(vec![1., 2.])
    );
}
