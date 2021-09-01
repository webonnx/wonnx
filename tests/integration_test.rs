use wonnx::*;
// Indicates a f32 overflow in an intermediate Collatz value
const LEN: u32 = 10_000_000;
// Args Management
async fn run() {
    let steps = execute_gpu().await.unwrap();

    println!("steps[0..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    log::info!("Steps: [{}]", disp_steps.join(", "));
}

// Hardware management
async fn execute_gpu() -> Option<Vec<f32>> {
    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = ressource::request_device_queue().await;

    let numbers: &[f32] = &vec![1.0; LEN as usize];

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = ressource::create_buffer_init(&device, numbers);
    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let binding_group_entry = wgpu::BindGroupEntry {
        binding: 0,
        resource: storage_buffer.as_entire_binding(),
    };

    let (compute_pipeline, bind_group) = crate::compute::wrapper(
        &device,
        &[binding_group_entry],
        &[0],
        &crate::op::map(&"cos"),
    );
    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(LEN, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    // encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = storage_buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to f32
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);

        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
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
