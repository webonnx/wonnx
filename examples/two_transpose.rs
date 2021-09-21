use log::debug;
use std::collections::HashMap;
use std::time::Instant;
// Indicates a f32 overflow in an intermediate Collatz value

// Args Management
async fn run() {
    let steps = execute_gpu().await.unwrap();

    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    log::info!("Steps: [{}]", disp_steps.join(", "));
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    // USER INPUT

    let n: i32 = 1024;
    let mut input_data = HashMap::new();
    let data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
    let dims = vec![n, n];
    input_data.insert("X", (data.as_slice(), dims.as_slice()));
    // LOGIC

    // Add Input to Graph
    // let mut initializer = wonnx::onnx::TensorProto::new();
    // initializer.set_name("X".to_string());
    // initializer.set_float_data(input_data);

    // let initializers = graph.mut_initializer();
    // initializers.insert(initializers.len(), initializer);
    let time_start = Instant::now();
    let session = wonnx::Session::new("tests/two_transposes.onnx")
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
