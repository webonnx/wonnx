use approx::assert_relative_eq;
use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value
use std::time::Instant;

use ndarray;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

async fn run() {
    let _ = execute_gpu().await.unwrap();

    let _ = single_dimension_matrix_multiplication().await.unwrap();
    for i in 0..10 {
        // println!("steps: {:?}", &steps[n * i..n * (i + 1)]);
    }
    //assert_eq!(steps[0..5], [16.0, 16.0, 16.0, 16.0, 16.0]);
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data_a = ndarray::Array2::eye(n);
    let dims_a = vec![n as i64, n as i64];
    let mut data_b = ndarray::Array2::<f32>::zeros((n, n));
    data_b[[0, 0]] = 0.0;
    data_b[[0, 1]] = 0.5;
    data_b[[0, 2]] = -0.5;
    data_b[[1, 0]] = 1.0;
    data_b[[1, 1]] = 1.5;
    data_b[[1, 2]] = -1.5;
    data_b[[2, 0]] = 2.0;
    data_b[[2, 1]] = 2.5;
    data_b[[2, 2]] = -2.5;
    let dims_b = vec![n as i64, n as i64];

    let sum = data_a.dot(&data_b);

    input_data.insert(
        "A".to_string(),
        (data_a.as_slice().unwrap(), dims_a.as_slice()),
    );
    input_data.insert(
        "B".to_string(),
        (data_b.as_slice().unwrap(), dims_b.as_slice()),
    );

    // ONNX INPUTS

    let mut shape_tensor_proto_dim = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim.set_dim_value(n as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![
        shape_tensor_proto_dim.clone(),
        shape_tensor_proto_dim,
    ]));

    let mut type_proto_tensor = crate::onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(1);
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = crate::onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut input_a = crate::onnx::ValueInfoProto::new();
    input_a.set_name("A".to_string());
    input_a.set_field_type(type_proto.clone());

    let mut input_b = crate::onnx::ValueInfoProto::new();
    input_b.set_name("B".to_string());
    input_b.set_field_type(type_proto.clone());

    let mut output = crate::onnx::ValueInfoProto::new();
    output.set_name("C".to_string());
    output.set_field_type(type_proto.clone());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("MatMul".to_string());
    node.set_name("node".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "A".to_string(),
        "B".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["C".to_string()]));

    let mut graph = wonnx::onnx::GraphProto::new();

    graph.set_node(protobuf::RepeatedField::from(vec![node.clone()]));
    graph.set_input(protobuf::RepeatedField::from(vec![input_a, input_b]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));

    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC
    let time_session = Instant::now();
    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    let time_finished_creation = Instant::now();
    println!(
        "time: finished_creation_session: {:#?}",
        time_finished_creation - time_session
    );
    let result = wonnx::run(&mut session, input_data).await;
    let time_finished_computation = Instant::now();
    println!(
        "time: finished_computation: {:#?}",
        time_finished_computation - time_finished_creation
    );

    assert_eq!(result.clone().unwrap().as_slice(), sum.as_slice().unwrap());
    result
}

// Hardware management
async fn single_dimension_matrix_multiplication() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 128;
    let mut input_data = HashMap::new();

    let mut data_a = ndarray::Array::random((1, n), Uniform::new(0., 10.));

    let dims_a = vec![1 as i64, n as i64];
    let data_b = ndarray::Array::random((n, n), Uniform::new(0., 10.));
    let dims_b = vec![n as i64, n as i64];

    let data_w = ndarray::Array::random((1, n), Uniform::new(0., 10.));
    let dims_w = vec![1 as i64, n as i64];
    let sum = data_a.dot(&data_b) + data_w.clone();

    input_data.insert(
        "A".to_string(),
        (data_a.as_slice().unwrap(), dims_a.as_slice()),
    );
    input_data.insert(
        "B".to_string(),
        (data_b.as_slice().unwrap(), dims_b.as_slice()),
    );
    input_data.insert(
        "W".to_string(),
        (data_w.as_slice().unwrap(), dims_w.as_slice()),
    );

    // ONNX INPUTS

    let mut shape_tensor_proto_dim_unit = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim_unit.set_dim_value(1 as i64);

    let mut shape_tensor_proto_dim = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim.set_dim_value(n as i64);

    let mut shape_tensor_proto_a = onnx::TensorShapeProto::new();
    shape_tensor_proto_a.set_dim(protobuf::RepeatedField::from(vec![
        shape_tensor_proto_dim_unit.clone(),
        shape_tensor_proto_dim.clone(),
    ]));

    let mut shape_tensor_proto_b = onnx::TensorShapeProto::new();
    shape_tensor_proto_b.set_dim(protobuf::RepeatedField::from(vec![
        shape_tensor_proto_dim.clone(),
        shape_tensor_proto_dim,
    ]));

    let mut type_proto_tensor_a = crate::onnx::TypeProto_Tensor::new();
    type_proto_tensor_a.set_elem_type(1);
    type_proto_tensor_a.set_shape(shape_tensor_proto_a);
    let mut type_proto_tensor_b = crate::onnx::TypeProto_Tensor::new();
    type_proto_tensor_b.set_elem_type(1);
    type_proto_tensor_b.set_shape(shape_tensor_proto_b);
    let mut type_proto_a = crate::onnx::TypeProto::new();
    type_proto_a.set_tensor_type(type_proto_tensor_a);

    let mut type_proto_b = crate::onnx::TypeProto::new();
    type_proto_b.set_tensor_type(type_proto_tensor_b);

    let mut input_a = crate::onnx::ValueInfoProto::new();
    input_a.set_name("A".to_string());
    input_a.set_field_type(type_proto_a.clone());

    let mut input_b = crate::onnx::ValueInfoProto::new();
    input_b.set_name("B".to_string());
    input_b.set_field_type(type_proto_b.clone());

    let mut input_w = crate::onnx::ValueInfoProto::new();
    input_w.set_name("W".to_string());
    input_w.set_field_type(type_proto_a.clone());

    let mut output = crate::onnx::ValueInfoProto::new();
    output.set_name("C".to_string());
    output.set_field_type(type_proto_a.clone());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("MatMul".to_string());
    node.set_name("node".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "A".to_string(),
        "B".to_string(),
        "W".to_string(),
    ]));

    node.set_output(protobuf::RepeatedField::from(vec!["C".to_string()]));

    let mut graph = wonnx::onnx::GraphProto::new();

    graph.set_node(protobuf::RepeatedField::from(vec![node.clone()]));
    graph.set_input(protobuf::RepeatedField::from(vec![input_a, input_b]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));

    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC
    let time_session = Instant::now();
    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    let time_finished_creation = Instant::now();
    println!(
        "time: finished_creation_session: {:#?}",
        time_finished_creation - time_session
    );
    let result = wonnx::run(&mut session, input_data).await;
    let time_finished_computation = Instant::now();
    println!(
        "time: finished_computation: {:#?}",
        time_finished_computation - time_finished_creation
    );
    for (a, b) in result.clone().unwrap().iter().zip(sum.as_slice().unwrap()) {
        assert_relative_eq!(a, b, epsilon = 0.01)
    }
    result
}
#[test]
fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
