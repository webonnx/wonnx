use std::{collections::HashMap, convert::TryInto};
use wonnx::tensor::{attribute, graph, initializer, model, node, tensor};
mod common;

#[test]
fn batch_normalization() {
    let mut input_data = HashMap::new();

    let batches = 1;
    let width_height: usize = 3;
    let channels: usize = 2;
    let data: Vec<f32> = (0..(batches * width_height * width_height * channels))
        .map(|x| x as f32)
        .collect();
    let shape = vec![
        batches as i64,
        channels as i64,
        width_height as i64,
        width_height as i64,
    ];
    input_data.insert("X".to_string(), data.as_slice().into());

    let mean: Vec<f32> = vec![100.0, -100.0];
    let var: Vec<f32> = vec![10.0, 10.0];
    let b: Vec<f32> = vec![1000.0, -1000.0];
    let scale: Vec<f32> = vec![1.0, 2.0];

    assert_eq!(mean.len(), channels);
    assert_eq!(var.len(), channels);
    assert_eq!(b.len(), channels);
    assert_eq!(scale.len(), channels);

    let bn_model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![
            initializer("scale", scale, vec![channels as i64]),
            initializer("B", b, vec![channels as i64]),
            initializer("input_mean", mean, vec![channels as i64]),
            initializer("input_var", var, vec![channels as i64]),
        ],
        vec![node(
            vec!["X", "scale", "B", "input_mean", "input_var"],
            vec!["Y"],
            "BatchNormalization",
            vec![attribute("epsilon", 0.1)],
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
            // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
            // For X=0, Y = (0 - 100) / sqrt(10 + 0.1) * 1 + 1000 = 968,53
            968.5342, 968.8488, 969.16345, 969.47815, 969.7928, 970.1074, 970.4221, 970.73676,
            971.05145, -931.4045, -930.77515, -930.1458, -929.51654, -928.8872, -928.2579,
            -927.62854, -926.99927, -926.36993,
        ],
    );
}
