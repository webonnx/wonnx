use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{graph, model, node, tensor};
mod common;

#[test]
fn global_average_pool() {
    let mut input_data = HashMap::new();

    let batches = 1;
    let width_height: usize = 2;
    let channels: usize = 4;
    // FIXME: we are testing with 4 channels because the AveragePool op doesn't support output tensors with total length non divisible by 4
    let data: Vec<f32> = (0..(batches * width_height * width_height * channels))
        .map(|x| x as f32)
        .collect();
    let shape = vec![
        batches as i64,
        channels as i64,
        width_height as i64,
        width_height as i64,
    ];
    let output_shape = vec![batches as i64, channels as i64, 1, 1];
    input_data.insert("X".to_string(), data.as_slice().into());

    // Model: X -> GlobalAveragePool -> Y
    let bn_model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &output_shape)],
        vec![],
        vec![],
        vec![node(
            vec!["X"],
            vec!["Y"],
            "gap",
            "GlobalAveragePool",
            vec![],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(bn_model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let out_y = &result["Y"];

    // The GlobalAveragePool op simply averages all pixels in an image (NxCxWxH becomes NxCx1x1). In our test data pixels
    // range from 0..16, so the test data and output are given as:
    // Channel 1: [[0,1], [2,3]] => average is 1,5
    // Channel 2: [[4,5], [6,7]] => average is 5,5
    // Channel 3: [[8,9], [10, 11]] => average is 9,5
    // Channel 4: [[12,13], [14, 15]] => average is 13,5
    common::assert_eq_vector(out_y.try_into().unwrap(), &[1.5, 5.5, 9.5, 13.5]);
}
