use log::debug;
use std::collections::HashMap;
use std::time::Instant;
// use wasm_bindgen_test::*;
// Indicates a f32 overflow in an intermediate Collatz value

// Args Management
async fn run() {
    let steps = execute_gpu().await.unwrap();

    assert_eq!(steps[0..5], [0.0, 1.0, 2.0, 3.0, 4.0]);
    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    // log::info!("steps[0..5]: {:#?}", &steps[0..5]);
    assert_eq!(steps[0..5], [0.0, 0.0, 0.0, 0.0, 0.0]);
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    // USER INPUT

    let n: i64 = 1024;
    let mut input_data = HashMap::new();
    let data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
    let dims = vec![n, n];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));
    // LOGIC

    // Add Input to Graph
    // let mut initializer = wonnx::onnx::TensorProto::new();
    // initializer.set_name("X".to_string());
    // initializer.set_float_data(input_data);

    // let initializers = graph.mut_initializer();
    // initializers.insert(initializers.len(), initializer);
    let time_start = Instant::now();
    let session = wonnx::Session::from_path("tests/two_transposes.onnx")
        .await
        .unwrap();
    debug!("session.model: {:#?}", session.model.get_graph());
    let time_init = Instant::now();
    debug!("time: init: {:#?}", time_init - time_start);

    let res = session.run(input_data).await;

    let time_run = Instant::now();
    debug!("time: run: {:#?}", time_run - time_init);

    res
}

#[test]
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
