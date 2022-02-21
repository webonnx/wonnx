use std::collections::HashMap;
use wonnx::utils::{attribute, graph, model, node, tensor, InputTensor};
mod common;

fn assert_gather(
    data: &[f32],
    data_shape: &[i64],
    indices: &[i32],
    indices_shape: &[i64],
    output: &[f32],
    output_shape: &[i64],
    axis: i64,
) {
    let mut input_data = HashMap::new();

    input_data.insert("X".to_string(), InputTensor::F32(data));
    input_data.insert("I".to_string(), InputTensor::I32(indices));

    // Model: (X, I) -> Gather -> Y
    let bn_model = model(graph(
        vec![tensor("X", data_shape), tensor("I", indices_shape)],
        vec![tensor("Y", output_shape)],
        vec![],
        vec![],
        vec![node(
            vec!["X", "I"],
            vec!["Y"],
            "myGather",
            "Gather",
            vec![attribute("axis", axis)],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(bn_model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector(result["Y"].as_slice(), output);
}

#[test]
fn gather() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Very simple test case that just does simple selection from a 1D array
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[6],
        &[3, 2, 3, 1],
        &[4],
        &[3.4, 2.3, 3.4, 1.2],
        &[4],
        0,
    );

    // Very simple test case that just does simple selection from a 1D array, with negative indexing
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[6],
        &[-3, -2, -3, -1],
        &[4],
        &[3.4, 4.5, 3.4, 5.7],
        &[4],
        0,
    );

    // Test case for axis=0 from https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[3, 2],
        &[0, 1, 1, 2],
        &[2, 2],
        &[1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7],
        &[2, 2, 2],
        0,
    );

    // Same as the above, but now with larger chunks to copy (so we test the shader's batching capability)
    assert_gather(
        &[1.0, 1.2, 2.3, 3.4, 4.5, 5.7, 1.0, 1.2, 2.3, 3.4, 4.5, 5.7],
        &[3, 4],
        &[0, 1, 1, 2],
        &[2, 2],
        &[
            1.0, 1.2, 2.3, 3.4, 4.5, 5.7, 1.0, 1.2, 4.5, 5.7, 1.0, 1.2, 2.3, 3.4, 4.5, 5.7,
        ],
        &[2, 2, 4],
        0,
    );
}
