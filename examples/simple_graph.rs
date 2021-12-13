use std::collections::HashMap;
// use wasm_bindgen_test::*;
// Indicates a f32 overflow in an intermediate Collatz value
use std::error;
use wonnx::utils::{attribute, graph, initializer, model, node, tensor};
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

async fn run() {
    let result = &execute_gpu().await.unwrap();
    let (_, result) = result.iter().next().unwrap();

    assert_eq!(result, &[54., 63., 72., 99., 108., 117., 144., 153., 162.]);
}

// Hardware management
async fn execute_gpu() -> Result<HashMap<String, Vec<f32>>> {
    // USER INPUT
    let n = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    input_data.insert("X".to_string(), data.as_slice());

    // ONNX INPUTS
    let dims = vec![1, c as i64, n as i64, n as i64];
    let kernel_n = 3;
    let m = 1;
    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();
    let model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Y", &[1, m, 3, 3])],
        vec![tensor("W", &[m, c, 3, 3])],
        vec![initializer("W", data_w)],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "conv",
            "Conv",
            vec![attribute("kernel_shape", vec![3, 3])],
        )],
    ));

    // LOGIC

    let mut session = wonnx::Session::from_model(model)
        .await
        .expect("Session did not create");

    wonnx::run(&mut session, input_data).await
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
