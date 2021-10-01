use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value

async fn run() {
    let steps = conv_pad().await.unwrap();

    let conv = 2;
    let n = 5;

    assert_eq!(
        steps,
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
    for i in 0..conv * 2 {
        println!("");
        for j in 0..n {
            println!(
                "steps: {:?}",
                &steps[i * n * n + n * j..i * n * n + n * (j + 1)]
            );
        }
        println!("steps: ]");
    }
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
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

    let mut shape_tensor_proto_dim = onnx::TensorShapeProto_Dimension::new();
    shape_tensor_proto_dim.set_dim_value(n as i64);

    let mut shape_tensor_proto = onnx::TensorShapeProto::new();
    shape_tensor_proto.set_dim(protobuf::RepeatedField::from(vec![shape_tensor_proto_dim]));

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

    session.run(input_data).await
}

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
