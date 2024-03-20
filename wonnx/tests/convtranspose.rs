use std::collections::HashMap;
use wonnx::utils::{attribute, graph, initializer, model, node, tensor, OutputTensor};
mod common;

#[test]
fn convtranspose_default() {
    let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_shape = vec![1, 1, 3, 3];

    let data_w = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let kernel_shape = vec![1, 2, 3, 3];

    let output_shape = vec![1, 2, 5, 5];

    let input_data = HashMap::from([("X".to_string(), data.as_slice().into())]);

    let convtranpose_model = model(graph(
        vec![tensor("X", &input_shape)],
        vec![tensor("Y", &output_shape)],
        vec![],
        vec![initializer("W", data_w, kernel_shape)],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "convtranspose",
            "ConvTranspose",
            vec![attribute("kernel_shape", vec![3, 3])],
        )],
    ));

    let session = pollster::block_on(wonnx::Session::from_model(convtranpose_model))
        .expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    assert_eq!(
        result["Y"],
        OutputTensor::F32(vec![
            0.0, 1.0, 3.0, 3.0, 2.0, 3.0, 8.0, 15.0, 12.0, 7.0, 9.0, 21.0, 36.0, 27.0, 15.0, 9.0,
            20.0, 33.0, 24.0, 13.0, 6.0, 13.0, 21.0, 15.0, 8.0, 0.0, 1.0, 3.0, 3.0, 2.0, 3.0, 8.0,
            15.0, 12.0, 7.0, 9.0, 21.0, 36.0, 27.0, 15.0, 9.0, 20.0, 33.0, 24.0, 13.0, 6.0, 13.0,
            21.0, 15.0, 8.0,
        ])
    );
}

#[test]
fn convtranspose_strides() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // (1, 1, 3, 3)
    let input_shape = vec![1, 1, 3, 3];

    let data_w = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let kernel_shape = vec![1, 2, 3, 3];

    let output_data = vec![
        0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0,
        3.0, 2.0, 2.0, 3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 3.0,
        3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 6.0, 6.0, 13.0, 7.0,
        15.0, 8.0, 8.0, 6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0,
        0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 3.0, 3.0, 7.0, 4.0,
        9.0, 5.0, 5.0, 3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 6.0,
        6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 6.0, 6.0, 13.0, 7.0,
        15.0, 8.0, 8.0,
    ];
    let output_shape = vec![1, 2, 9, 7];

    let convtranpose_model = model(graph(
        vec![tensor("X", &input_shape)],
        vec![tensor("Y", &output_shape)],
        vec![],
        vec![initializer("W", data_w, kernel_shape)],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "convtranspose",
            "ConvTranspose",
            vec![
                attribute("kernel_shape", vec![3, 3]),
                attribute("strides", vec![3, 2]),
            ],
        )],
    ));

    let input_data = HashMap::from([("X".to_string(), data.as_slice().into())]);
    let session = pollster::block_on(wonnx::Session::from_model(convtranpose_model))
        .expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], OutputTensor::F32(output_data));
}

#[test]
fn convtranspose_pads() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_shape = vec![1, 1, 3, 3];

    let data_w = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let kernel_shape = vec![1, 2, 3, 3];

    let output_data = vec![
        1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 7.0, 4.0, 9.0, 7.0, 4.0, 9.0, 7.0, 4.0, 9.0, 13.0, 7.0, 15.0,
        13.0, 7.0, 15.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 7.0, 4.0, 9.0, 7.0, 4.0, 9.0, 7.0, 4.0, 9.0,
        13.0, 7.0, 15.0, 13.0, 7.0, 15.0,
    ];
    let output_shape = vec![1, 2, 7, 3];

    let convtranpose_model = model(graph(
        vec![tensor("X", &input_shape)],
        vec![tensor("Y", &output_shape)],
        vec![],
        vec![initializer("W", data_w, kernel_shape)],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "convtranspose",
            "ConvTranspose",
            vec![
                attribute("strides", vec![3, 2]),
                attribute("pads", vec![1, 2, 1, 2]),
            ],
        )],
    ));

    let input_data = HashMap::from([("X".to_string(), data.as_slice().into())]);
    let session = pollster::block_on(wonnx::Session::from_model(convtranpose_model))
        .expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(result["Y"], OutputTensor::F32(output_data));
}
