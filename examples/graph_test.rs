use log::info;
use std::time::Instant;
use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value
const LEN: u32 = 512 * 512 * 128;
// Args Management
async fn run() {
    let steps = execute_gpu().await.unwrap();

    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    log::info!("Steps: [{}]", disp_steps.join(", "));
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    let (device, queue) = ressource::request_device_queue().await;

    let numbers: &[f32] = &vec![1.0; LEN as usize];

    let storage_buffer = ressource::create_buffer_init(&device, numbers);

    let time_preprocess = Instant::now();

    for _ in 0..20 {
        crate::compute::wrapper(
            &device,
            &queue,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
            &[crate::compute::InnerType::ArrayMatrix],
            &crate::op::matrix_map(&"cos(cos(cos(cos(cos(b_0.data[global_id.x][index_mat])))))"),
            LEN / 16,
            1,
            1,
        )
        .unwrap();
    }
    let buffer_slice = storage_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to f32
        let result = bytemuck::cast_slice(&data).to_vec();

        let time_postprocess = Instant::now();
        info!(
            "time_postprocess - time_preprocess: {:#?}",
            time_postprocess - time_preprocess
        );
        drop(data);

        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
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
