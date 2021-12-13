// use approx::assert_relative_eq;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value
use std::time::Instant;

#[test]
fn execute_gpu() {
    // USER INPUT

    let n: usize = 16;
    let mut input_data = HashMap::new();

    let data_a = ndarray::Array2::eye(n);
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

    let sum = data_a.dot(&data_b);

    input_data.insert("A".to_string(), data_a.as_slice().unwrap());
    input_data.insert("B".to_string(), data_b.as_slice().unwrap());

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
    let mut session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = &pollster::block_on(wonnx::run(&mut session, input_data)).unwrap()["Y"];
    let time_finished_creation = Instant::now();
    println!(
        "time: finished_creation_session: {:#?}",
        time_finished_creation - time_session
    );
    let time_finished_computation = Instant::now();
    println!(
        "time: finished_computation: {:#?}",
        time_finished_computation - time_finished_creation
    );

    assert_eq!(result.as_slice(), sum.as_slice().unwrap());
}

fn _single_dimension_matrix_multiplication() {
    // USER INPUT

    let n = 128.;
    let input_data = HashMap::new();

    //let mut data_a = ndarray::ArrayBase::range(0., n, 1.);
    //data_a.reshape((1, n as _));
    //let mut data_b = ndarray::ArrayBase::range(0., n * n, 1.);
    //data_b.reshape((n as _, n as _));
    //let mut data_w = ndarray::ArrayBase::range(0., n, 1.);
    //data_w.reshape((1, n as _));

    // let sum: ndarray::Array<f64, ndarray::Dim(1)> = data_a.dot(&data_b) + data_w.clone();

    // let sum = sum.as_slice().unwrap();

    //input_data.insert("A".to_string(), data_a.as_slice().unwrap());
    //input_data.insert("B".to_string(), data_b.as_slice().unwrap());
    //input_data.insert("W".to_string(), data_w.as_slice().unwrap());

    // ONNX INPUTS

    let mut shape_tensor_proto_dim_unit = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim_unit.set_dim_value(1i64);

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
    let mut session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let _result = &pollster::block_on(wonnx::run(&mut session, input_data)).unwrap()["Y"];

    let time_finished_creation = Instant::now();
    println!(
        "time: finished_creation_session: {:#?}",
        time_finished_creation - time_session
    );
    let time_finished_computation = Instant::now();
    println!(
        "time: finished_computation: {:#?}",
        time_finished_computation - time_finished_creation
    );
    //for (a, b) in result.iter().zip(sum) {
    // assert_relative_eq!(a, b.into(), epsilon = 0.01)
    //}
}
