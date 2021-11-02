use log::info;
use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value

async fn run() {
    let conv_pad = conv_pad().await.unwrap();

    assert_eq!(
        conv_pad,
        [
            12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0,
            81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 12.0, 21.0,
            27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0, 117.0, 81.0, 93.0,
            144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0, 112.0, 171.0, 177.0,
            183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0, 231.0,
            243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0, 112.0, 171.0,
            177.0, 183.0, 124.0, 183.0, 279.0, 288.0, 297.0, 201.0, 213.0, 324.0, 333.0, 342.0,
            231.0, 243.0, 369.0, 378.0, 387.0, 261.0, 172.0, 261.0, 267.0, 273.0, 184.0
        ]
    );

    let conv_without_pad = conv_without_pad().await.unwrap();

    assert_eq!(
        conv_without_pad,
        [54., 63., 72., 99., 108., 117., 144., 153., 162., 0.0, 0.0, 0.0]
    );

    let conv_stride = conv_stride().await.unwrap();

    assert_eq!(
        conv_stride,
        [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.]
    );

    let conv_asymetric_stride = conv_asymetric_stride().await.unwrap();

    assert_eq!(
        conv_asymetric_stride,
        [21., 33., 99., 117., 189., 207., 171., 183.]
    );

    //    let conv_asymetric_stride = conv_kernel_3().await.unwrap();
    //
    //    assert_eq!(
    //        conv_asymetric_stride,
    //        [
    //            10., 18., 24., 30., 36.,
    //            37., 14., 18., 22., 26.,
    //        ]
    //    );

    let steps = conv_simple().await.unwrap();
    let conv = 1;
    let n = 2;
    for i in 0..conv {
        println!("");
        for j in 0..n {
            info!(
                "steps: {:?}",
                &steps[i * n * n + n * j..i * n * n + n * (j + 1)]
            );
        }
        println!("steps: ]");
    }
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
}

async fn conv_simple() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 3;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        .iter()
        .map(|x| *x as f32)
        .collect();
    let dims = vec![1, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(2 as i64);

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

    let kernel_n = 2;
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
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
    pads.set_ints(vec![0, 0, 0, 0]);

    // let mut auto_pad = crate::onnx::AttributeProto::new();
    // auto_pad.set_name("auto_pad".to_string());
    // auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

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

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
}

async fn conv_pad() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..50).map(|x| x as f32).collect();
    let dims = vec![2, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(2 as i64);

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
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
    let mut initializer_w = crate::onnx::TensorProto::new();
    initializer_w.set_name("W".to_string());
    initializer_w.set_float_data(data_w);
    initializer_w.set_data_type(1);
    initializer_w.set_dims(vec![m as i64, c as i64, kernel_n as i64, kernel_n as i64]);

    let mut kernel_shape = crate::onnx::AttributeProto::new();
    kernel_shape.set_name("kernel_shape".to_string());
    kernel_shape.set_ints(vec![kernel_n as i64, kernel_n as i64]);

    // let mut pads = crate::onnx::AttributeProto::new();
    // pads.set_name("pads".to_string());
    // pads.set_ints(vec![1, 1, 1, 1]);

    let mut auto_pad = crate::onnx::AttributeProto::new();
    auto_pad.set_name("auto_pad".to_string());
    auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Conv".to_string());
    node.set_name("conv".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "X".to_string(),
        "W".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![kernel_shape, auto_pad]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    graph.set_initializer(protobuf::RepeatedField::from(vec![initializer_w]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
}

async fn conv_without_pad() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    let dims = vec![1, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(1 as i64);

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
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
    let mut initializer_w = crate::onnx::TensorProto::new();
    initializer_w.set_name("W".to_string());
    initializer_w.set_float_data(data_w);
    initializer_w.set_data_type(1);
    initializer_w.set_dims(vec![m as i64, c as i64, kernel_n as i64, kernel_n as i64]);

    let mut kernel_shape = crate::onnx::AttributeProto::new();
    kernel_shape.set_name("kernel_shape".to_string());
    kernel_shape.set_ints(vec![kernel_n as i64, kernel_n as i64]);

    // let mut pads = crate::onnx::AttributeProto::new();
    // pads.set_name("pads".to_string());
    // pads.set_ints(vec![1, 1, 1, 1]);

    //    let mut auto_pad = crate::onnx::AttributeProto::new();
    //    auto_pad.set_name("auto_pad".to_string());
    //    auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Conv".to_string());
    node.set_name("conv".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "X".to_string(),
        "W".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![kernel_shape]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    graph.set_initializer(protobuf::RepeatedField::from(vec![initializer_w]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
}

async fn conv_stride() -> Option<Vec<f32>> {
    // USER INPUT

    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..35).map(|x| x as f32).collect();
    let dims = vec![1, c as i64, 7 as i64, 5 as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(1 as i64);

    let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
    dim_channel.set_dim_value(c as i64);

    let mut dim_m = onnx::TensorShapeProto_Dimension::new();
    dim_m.set_dim_value(7 as i64);

    let mut dim_n = onnx::TensorShapeProto_Dimension::new();
    dim_n.set_dim_value(5 as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![
        dim_batch,
        dim_channel,
        dim_m.clone(),
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
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
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

    let mut strides = crate::onnx::AttributeProto::new();
    strides.set_name("strides".to_string());
    strides.set_ints(vec![2, 2]);
    //    let mut auto_pad = crate::onnx::AttributeProto::new();
    //    auto_pad.set_name("auto_pad".to_string());
    //    auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Conv".to_string());
    node.set_name("conv".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "X".to_string(),
        "W".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![
        kernel_shape,
        pads,
        strides,
    ]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    graph.set_initializer(protobuf::RepeatedField::from(vec![initializer_w]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
}
async fn conv_asymetric_stride() -> Option<Vec<f32>> {
    // USER INPUT

    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..35).map(|x| x as f32).collect();
    let dims = vec![1, c as i64, 7 as i64, 5 as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(1 as i64);

    let mut dim_channel = onnx::TensorShapeProto_Dimension::new();
    dim_channel.set_dim_value(c as i64);

    let mut dim_m = onnx::TensorShapeProto_Dimension::new();
    dim_m.set_dim_value(7 as i64);

    let mut dim_n = onnx::TensorShapeProto_Dimension::new();
    dim_n.set_dim_value(5 as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![
        dim_batch,
        dim_channel,
        dim_m.clone(),
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
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
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
    pads.set_ints(vec![1, 0, 1, 0]);

    let mut strides = crate::onnx::AttributeProto::new();
    strides.set_name("strides".to_string());
    strides.set_ints(vec![2, 2]);
    //    let mut auto_pad = crate::onnx::AttributeProto::new();
    //    auto_pad.set_name("auto_pad".to_string());
    //    auto_pad.set_s("SAME_UPPER".to_string().into_bytes());

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("Conv".to_string());
    node.set_name("conv".to_string());
    node.set_input(protobuf::RepeatedField::from(vec![
        "X".to_string(),
        "W".to_string(),
    ]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![
        kernel_shape,
        pads,
        strides,
    ]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    graph.set_initializer(protobuf::RepeatedField::from(vec![initializer_w]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
}

async fn conv_kernel_3() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 4;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let dims = vec![2, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    // ONNX INPUTS

    let mut dim_batch = onnx::TensorShapeProto_Dimension::new();
    dim_batch.set_dim_value(2 as i64);

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
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1 as f32).collect();
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

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
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
