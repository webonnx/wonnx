use std::collections::HashMap;
// use wasm_bindgen_test::*;
// Indicates a f32 overflow in an intermediate Collatz value
use std::error;
use wonnx::utils::{attribute, graph, initializer, model, node, tensor};
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

async fn run() {
    let steps = execute_gpu().await.unwrap();

    assert_eq!(steps[0..5], [1.0, 1.0, 1.0, 1.0, 1.0]);
    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    log::info!("steps[0..5]: {:#?}", &steps[0..5]);
    assert_eq!(steps[0..5], [1.0, 1.0, 1.0, 1.0, 1.0]);
}

// Hardware management
async fn execute_gpu() -> Result<Vec<f32>> {
    // USER INPUT

    let n = 5;
    let c = 1;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..50).map(|x| x as f32).collect();
    let dims = vec![2, c as i64, n as i64, n as i64];
    input_data.insert("X".to_string(), data.as_slice());

    // ONNX INPUTS

    let data_w: Vec<f32> = (0..2 * c * 3 * 3).map(|_| 1.0f32).collect();

    let model = model(graph(
        vec![tensor("X", &dims)],
        vec![tensor("Y", &[2, 2, n, n])],
        vec![initializer("W", data_w, &[2, c, 3, 3])],
        vec![node(
            vec!["X", "W"],
            vec!["Y"],
            "conv",
            "Conv",
            vec![
                attribute("kernel_shape", vec![3, 3]),
                attribute("auto_pad", "SAME_UPPER"),
            ],
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
