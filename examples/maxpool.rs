use protobuf;
use std::collections::HashMap;
// use wasm_bindgen_test::*;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value

async fn run() {
    let steps = conv_pad().await.unwrap();

    let n = 5;

    println!("steps: [");
    for j in 0..n {
        println!("steps: {:?}", &steps[n * j..n * (j + 1)]);
    }
    println!("steps: ]");
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
}

async fn conv_pad() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    let dims = vec![1, c as i64, n as i64, n as i64];
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

    let mut kernel_shape = crate::onnx::AttributeProto::new();
    kernel_shape.set_name("kernel_shape".to_string());
    kernel_shape.set_ints(vec![kernel_n as i64, kernel_n as i64]);

    let mut pads = crate::onnx::AttributeProto::new();
    pads.set_name("pads".to_string());
    pads.set_ints(vec![1, 1, 1, 1]);

    let mut node = crate::onnx::NodeProto::new();
    node.set_op_type("MaxPool".to_string());
    node.set_name("maxpool".to_string());
    node.set_input(protobuf::RepeatedField::from(vec!["X".to_string()]));
    node.set_output(protobuf::RepeatedField::from(vec!["Y".to_string()]));
    node.set_attribute(protobuf::RepeatedField::from(vec![kernel_shape, pads]));

    let mut graph = wonnx::onnx::GraphProto::new();
    graph.set_node(protobuf::RepeatedField::from(vec![node]));
    graph.set_input(protobuf::RepeatedField::from(vec![input]));
    graph.set_output(protobuf::RepeatedField::from(vec![output]));
    let mut model = crate::onnx::ModelProto::new();
    model.set_graph(graph);

    // LOGIC

    let session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    session.run(input_data).await
}
// Hardware management
async fn conv_no_pad() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    let dims = vec![1, c as i64, n as i64, n as i64];
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

    let session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    session.run(input_data).await
}

// #[wasm_bindgen_test]
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
