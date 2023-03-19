use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use ndarray::s;
use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;
use wonnx::utils::InputTensor;
mod common;

#[test]
fn test_relu() {
    let mut input_data = HashMap::new();
    let data = vec![-1.0f32, 1.0];
    input_data.insert("x".to_string(), data.as_slice().into());

    let session = pollster::block_on(wonnx::Session::from_path("../data/models/single_relu.onnx"))
        .expect("session did not create");
    let result = pollster::block_on(session.run(&input_data)).unwrap();

    common::assert_eq_vector((&result["y"]).try_into().unwrap(), &[0.0, 1.0]);
}

fn infer_mnist(image: InputTensor) -> (usize, f32) {
    let session = pollster::block_on(wonnx::Session::from_path("../data/models/opt-mnist.onnx"))
        .expect("Session did not create");

    let mut input_data = HashMap::new();
    input_data.insert("Input3".to_string(), image);
    let output = pollster::block_on(session.run(&input_data)).unwrap();
    let output_tensor: &[f32] = (&output["Plus214_Output_0"]).try_into().unwrap();
    output_tensor
        .iter()
        .enumerate()
        .fold((0, 0.), |(idx_max, val_max), (idx, val)| {
            if &val_max > val {
                (idx_max, val_max)
            } else {
                (idx, *val)
            }
        })
}

#[test]
fn test_mnist() {
    let _ = env_logger::builder().is_test(true).try_init();
    let image = load_image("0.jpg");
    let result = infer_mnist(image.as_slice().unwrap().into());
    assert_eq!(result.0, 0);

    let image = load_image("3.jpg");
    let result = infer_mnist(image.as_slice().unwrap().into());
    assert_eq!(result.0, 3);

    let image = load_image("5.jpg");
    let result = infer_mnist(image.as_slice().unwrap().into());
    assert_eq!(result.0, 5);

    let image = load_image("7.jpg");
    let result = infer_mnist(image.as_slice().unwrap().into());
    assert_eq!(result.0, 7);
}

// Ignore on Windows now because of: https://github.com/gfx-rs/wgpu/issues/2285
// Ignore on Linux because for some reason this does not seem to work properly using Lavapipe
#[cfg(target_os = "macos")]
#[test]
fn test_squeeze() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut input_data = HashMap::new();
    let image = load_squeezenet_image();
    input_data.insert("data".to_string(), image.as_slice().unwrap().into());

    let session = pollster::block_on(wonnx::Session::from_path("../data/models/opt-squeeze.onnx"))
        .expect("session did not create");
    let result =
        &pollster::block_on(session.run(&input_data)).unwrap()["squeezenet0_flatten0_reshape0"];
    let output_tensor: &[f32] = result.try_into().unwrap();
    let mut probabilities: Vec<(usize, f32)> = output_tensor.iter().cloned().enumerate().collect();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    log::info!("{probabilities:?}");
    assert_eq!(probabilities[0].0, 22);
}

pub fn load_image(
    image_path: &str,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/images")
            .join(image_path),
    )
    .unwrap()
    .resize_exact(28, 28, FilterType::Nearest)
    .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    })
}

pub fn load_squeezenet_image(
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/images")
            .join("bald_eagle.jpeg"),
    )
    .unwrap()
    .resize_exact(224, 224, FilterType::Nearest)
    .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    // Batch of 1
    array
}
