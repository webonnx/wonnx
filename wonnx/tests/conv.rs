use std::collections::HashMap;
use std::convert::TryInto;
use wonnx::utils::{attribute, graph, initializer, model, node, tensor, OutputTensor};
use wonnx::*;
mod common;

#[test]
fn conv_pad() {
    let n = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..50).map(|x| x as f32).collect();
    let shape = vec![2, c, n, n];
    input_data.insert("X".to_string(), data.as_slice().into());

    let data_w: Vec<f32> = (0..2 * c * 3 * 3).map(|_| 1.0f32).collect();

    let conv_model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &[2, 2, n, n])],
        vec![],
        vec![initializer("W", data_w, vec![2, c, 3, 3])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "Conv",
            vec![
                attribute("kernel_shape", vec![3, 3]),
                attribute("auto_pad", "SAME_UPPER"),
            ],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(conv_model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    assert_eq!(
        result["Y"],
        OutputTensor::F32(vec![
            12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0,
            81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 12.0, 21.0,
            27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0, 81.0, 93.0,
            144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 112.0, 171.0, 177.0,
            183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0, 231.0,
            243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0, 112.0, 171.0,
            177.0, 183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0,
            231.0, 243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0
        ])
    );
}

#[test]
fn conv_without_pad() {
    let n = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    let shape = vec![1, c, n as i64, n as i64];
    input_data.insert("X".to_string(), data.as_slice().into());

    let kernel_n = 3;
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();
    let conv_model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &[1, 1, 3, 3])],
        vec![],
        vec![initializer("W", data_w, vec![m, c, kernel_n, kernel_n])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "Conv",
            vec![attribute("kernel_shape", vec![3, 3])],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(conv_model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector(
        (&result["Y"]).try_into().unwrap(),
        &[54., 63., 72., 99., 108., 117., 144., 153., 162.],
    );
}

#[test]
fn conv_group_simple() {
    let mut input_data = HashMap::new();

    let shape = vec![1, 2, 1, 1];
    input_data.insert("X".to_string(), [1.0, 2.0][..].into());

    let conv_model = model(graph(
        vec![tensor("X", &shape)],
        vec![tensor("Y", &shape)],
        vec![],
        vec![initializer("W", vec![0.5, 2.0], vec![2, 1, 1, 1])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "Conv",
            vec![attribute("kernel_shape", vec![1, 1]), attribute("group", 2)],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(conv_model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector((&result["Y"]).try_into().unwrap(), &[0.5, 4.0]);
}

#[test]
fn conv_stride() {
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..35).map(|x| x as f32).collect();
    input_data.insert("X".to_string(), data.as_slice().into());

    // ONNX INPUTS

    let kernel_n = 3;
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();

    //    let mut auto_pad = crate::onnx::AttributeProto::new();
    //    auto_pad.set_name("auto_pad".to_string());
    //    auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

    let model = model(graph(
        vec![tensor("X", &[1, c, 7, 5])],
        vec![tensor("Y", &[1, 1, 4, 3])],
        vec![],
        vec![initializer("W", data_w, vec![m, c, kernel_n, kernel_n])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "Conv",
            vec![
                attribute("strides", vec![2, 2]),
                attribute("pads", vec![1, 1, 1, 1]),
                attribute("kernel_shape", vec![kernel_n, kernel_n]),
            ],
        )],
    ));

    // LOGIC

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector(
        (&result["Y"]).try_into().unwrap(),
        &[
            12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.,
        ],
    )
}

#[test]
fn conv_asymetric_stride() {
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..35).map(|x| x as f32).collect();
    input_data.insert("X".to_string(), data.as_slice().into());

    let kernel_n = 3;
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();

    let model = model(graph(
        vec![tensor("X", &[1, c, 7, 5])],
        vec![tensor("Y", &[1, 1, 4, 2])],
        vec![],
        vec![initializer("W", data_w, vec![m, c, kernel_n, kernel_n])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "Conv",
            vec![
                attribute("strides", vec![2, 2]),
                attribute("pads", vec![1, 0, 1, 0]),
                attribute("kernel_shape", vec![kernel_n, kernel_n]),
            ],
        )],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();
    common::assert_eq_vector(
        (&result["Y"]).try_into().unwrap(),
        &[21., 33., 99., 117., 189., 207., 171., 183.],
    );
}

fn _conv_kernel_3() {
    let n: usize = 4;
    let c = 1;

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(2i64);

    let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
    dim_channel.set_dim_value(c as i64);

    let mut dim_n = onnx::TensorShapeProto_Dimension::new();
    dim_n.set_dim_value(n as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![
        dim_batch,
        dim_channel,
        dim_n.clone(),
        dim_n,
    ]));

    let mut type_proto_tensor = crate::onnx::TypeProto_Tensor::new();
    type_proto_tensor.set_elem_type(1);
    type_proto_tensor.set_shape(shape_tensor_proto);

    let mut type_proto = crate::onnx::TypeProto::new();
    type_proto.set_tensor_type(type_proto_tensor);

    let mut input = crate::onnx::ValueInfoProto::new();
    input.set_name("X".to_string());
    input.set_field_type(type_proto.clone());

    let mut output = crate::onnx::ValueInfoProto::new();
    output.set_name("Y".to_string());
    output.set_field_type(type_proto.clone());

    let kernel_n = 3;
    let m = 2;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1f32).collect();
    let mut initializer_w = crate::onnx::TensorProto::new();
    initializer_w.set_name("W".to_string());
    initializer_w.set_float_data(data_w);
    initializer_w.set_data_type(1);
    initializer_w.set_dims(vec![m as i64, c as i64, kernel_n as i64, kernel_n as i64]);

    let mut kernel_shape = crate::onnx::AttributeProto::new();
    kernel_shape.set_name("kernel_shape".to_string());
    kernel_shape.set_ints(vec![kernel_n as i64, kernel_n as i64]);

    let mut pads = crate::onnx::AttributeProto::new();
    pads.set_name("pads".to_string());
    pads.set_ints(vec![1, 1, 1, 1]);

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Conv".to_string());
    node.set_name("conv".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "X".to_string(),
        "W".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![kernel_shape, pads]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    graph.set_initializer(protobuf::RepeatedField::from(vec![initializer_w]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);
}
