use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{attribute, graph, model, node, tensor};
mod common;

#[test]
fn local_response_normalization() {
    let mut input_data = HashMap::new();

    let batches = 1;
    let width_height: usize = 3;
    let channels: usize = 4;
    let data: Vec<f32> = [
        1., 1., 2., 4., 2., 2., 1., 2., 3., 1., 2., 1., 4., 2., 3., 5., 3., 3., 2., 2., 6., 2., 3.,
        1., 7., 3., 4., 2., 8., 4., 3., 2., 9., 3., 4., 4.,
    ]
    .to_vec();

    let shape = vec![
        batches as i64,
        channels as i64,
        width_height as i64,
        width_height as i64,
    ];
    input_data.insert("X".to_string(), data.as_slice().into());

    let bn_model = model(graph(
        vec![tensor("X", &shape)], // input
        vec![tensor("Y", &shape)], // output
        vec![],                    // infos
        vec![],                    // intializers
        // nodes
        vec![node(
            vec!["X"],
            vec!["Y"],
            "lrn",
            "LRN",
            vec![
                attribute("alpha", 1.0),
                attribute("beta", 1.0),
                attribute("bias", 0.0),
                attribute("size", 2),
            ],
        )],
    ));

    // LOGIC
    let session =
        pollster::block_on(wonnx::Session::from_model(bn_model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();
    let out_y = &result["Y"];

    common::assert_eq_vector(
        out_y.try_into().unwrap(),
        &[
            1.0, 0.4, 0.2, 0.5, 0.5, 0.8, 0.4, 1.0, 0.6, 0.4, 0.8, 2.0, 0.4, 0.30769232, 0.1764706,
            0.39999998, 0.33333334, 0.4615385, 0.5, 1.0, 0.3, 0.30769232, 0.6, 2.0, 0.2413793,
            0.24, 0.4, 1.0, 0.2, 0.32, 0.4615385, 1.0, 0.2, 0.24, 0.25, 0.5,
        ],
    );
}
