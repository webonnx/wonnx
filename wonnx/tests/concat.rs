use std::{collections::HashMap, convert::TryInto};
use wonnx::utils::{graph, model, node, tensor};
mod common;

#[test]
fn test_concat() {
    let n: usize = 16;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 2) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![tensor("X", &input_dims), tensor("Y", &input_dims)],
        vec![tensor("Z", &output_dims)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "a", "Concat", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat_long() {
    let n: usize = 100000;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 2) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![tensor("X", &input_dims), tensor("Y", &input_dims)],
        vec![tensor("Z", &output_dims)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y"], vec!["Z"], "a", "Concat", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);

    common::assert_eq_vector((&result["Z"]).try_into().unwrap(), &expected_result);
}

#[test]
fn test_concat4() {
    let n: usize = 13;

    let xdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut ydata: Vec<f32> = (n..2 * n).map(|x| x as f32).collect();
    let mut zdata: Vec<f32> = (n * 2..3 * n).map(|x| x as f32).collect();
    let mut wdata: Vec<f32> = (n * 3..4 * n).map(|x| x as f32).collect();
    let input_dims = vec![n as i64];
    let output_dims = vec![(n * 4) as i64];

    let input_data = HashMap::from([
        ("X".into(), xdata.as_slice().into()),
        ("Y".into(), ydata.as_slice().into()),
        ("Z".into(), zdata.as_slice().into()),
        ("W".into(), wdata.as_slice().into()),
    ]);

    let model = model(graph(
        vec![
            tensor("X", &input_dims),
            tensor("Y", &input_dims),
            tensor("Z", &input_dims),
            tensor("W", &input_dims),
        ],
        vec![tensor("O", &output_dims)],
        vec![],
        vec![],
        vec![node(vec!["X", "Y", "Z", "W"], vec!["O"], "a", "Concat", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    let mut expected_result = xdata.clone();
    expected_result.append(&mut ydata);
    expected_result.append(&mut zdata);
    expected_result.append(&mut wdata);

    common::assert_eq_vector((&result["O"]).try_into().unwrap(), &expected_result);
}
