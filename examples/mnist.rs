use std::collections::HashMap;
// Indicates a f32 overflow in an intermediate Collatz value
// use wasm_bindgen_test::*;

use std::time::Instant;
// Args Management
async fn run() {
    let steps = execute_gpu().await.unwrap();
    println!("steps: {:#?}", steps);
    // println!("steps[1..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    // log::info!("steps[0..5]: {:#?}", &steps[0..5]);
    assert_eq!(steps[0..5], [0.0, 0.0, 0.0, 0.0, 0.0]);
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    let n: usize = 28;
    let mut input_data = HashMap::new();

    let data: Vec<f32> = (0..n * n).map(|x| x as f32).collect();
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    input_data.insert("Input3".to_string(), (data.as_slice(), dims.as_slice()));

    let mut session = wonnx::Session::from_path("tests/mnist-8.onnx")
        .await
        .unwrap();
    let time_pre_compute = Instant::now();
    let a = (wonnx::run(&mut session, input_data)).await;
    let time_post_compute = Instant::now();
    println!(
        "time: post_compute: {:#?}",
        time_post_compute - time_pre_compute
    );
    a
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let time_pre_compute = Instant::now();
        pollster::block_on(run());
        println!("time: post_compute: {:#?}", time_pre_compute.elapsed());
    }
    #[cfg(target_arch = "wasm32")]
    {
        // std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        //  console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
