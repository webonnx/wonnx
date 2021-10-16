use std::collections::HashMap;
// Indicates a f32 overflow in an intermediate Collatz value
// use wasm_bindgen_test::*;

use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use std::path::Path;
use std::time::Instant;

// Args Management
async fn run() {
    execute_gpu().await;
    // println!("steps[1..5]: {:#?}", &steps[0..5]);
    #[cfg(target_arch = "wasm32")]
    // log::info!("steps[0..5]: {:#?}", &steps[0..5]);
    assert_eq!(steps[0..5], [0.0, 0.0, 0.0, 0.0, 0.0]);
}

// Hardware management
async fn execute_gpu() {
    let n: usize = 28;

    let mut session = wonnx::Session::from_path("examples/data/models/opt-mnist.onnx")
        .await
        .unwrap();

    let image = load_image("0.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let probabilities = (wonnx::run(&mut session, input_data)).await.unwrap();

    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 0);

    let mut session = wonnx::Session::from_path("examples/data/models/opt-mnist.onnx")
        .await
        .unwrap();
    let image = load_image("3.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let probabilities = (wonnx::run(&mut session, input_data)).await.unwrap();
    println!("probabilities: {:#?}", probabilities);
    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 3);

    let mut session = wonnx::Session::from_path("examples/data/models/opt-mnist.onnx")
        .await
        .unwrap();
    let image = load_image("5.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let probabilities = (wonnx::run(&mut session, input_data)).await.unwrap();

    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 5);

    let mut session = wonnx::Session::from_path("examples/data/models/opt-mnist.onnx")
        .await
        .unwrap();
    let image = load_image("7.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let probabilities = (wonnx::run(&mut session, input_data)).await.unwrap();

    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 7);
}

#[test]
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

pub fn load_image(
    number: &str,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/data/images")
            .join(number),
    )
    .unwrap()
    .resize_exact(28 as u32, 28 as u32, FilterType::Nearest)
    .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    let array = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Batch of 1
    array
}
