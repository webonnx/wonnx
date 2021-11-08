use std::collections::HashMap;
// Indicates a f32 overflow in an intermediate Collatz value
// use wasm_bindgen_test::*;

use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use std::path::Path;

use ndarray::s;
#[test]
fn test_relu() {
    let n: usize = 2;
    let mut input_data = HashMap::new();
    let data = vec![-1.0f32, 1.0];
    let dims = vec![1, n as i64];
    input_data.insert("x".to_string(), (data.as_slice(), dims.as_slice()));

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/single_relu.onnx",
    ))
    .expect("session did not create");
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    assert_eq!(result, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
#[ignore] // TODO: Implement transpose
fn test_two_transposes() {
    // USER INPUT

    let mut input_data = HashMap::new();
    let data = (0..2 * 3 * 4).map(|x| x as f32).collect::<Vec<f32>>();
    let dims = vec![2, 3, 4];
    input_data.insert("X".to_string(), (data.as_slice(), dims.as_slice()));

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/two_transposes.onnx",
    ))
    .expect("session did not create");
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    assert_eq!(result[0..5], [0., 1., 2., 3., 4., 5.]);
}

#[test]
fn test_mnist() {
    let n: usize = 28;

    let image = load_image("0.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-mnist.onnx",
    ))
    .expect("Session did not create");

    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 0);

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-mnist.onnx",
    ))
    .expect("session did not create");
    let image = load_image("3.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();
    println!("probabilities: {:#?}", probabilities);

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 3);

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-mnist.onnx",
    ))
    .expect("session did not create");
    let image = load_image("5.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 5);

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-mnist.onnx",
    ))
    .expect("session did not create");
    let image = load_image("7.jpg");
    let dims = vec![1, 1 as i64, n as i64, n as i64];
    let mut input_data = HashMap::new();
    input_data.insert(
        "Input3".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();

    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(probabilities[0].0, 7);
}

#[test]
fn test_squeeze() {
    let n: usize = 224;
    let mut input_data = HashMap::new();
    let image = load_squeezenet_image();
    let dims = vec![1, 3 as i64, n as i64, n as i64];
    input_data.insert(
        "data".to_string(),
        (image.as_slice().unwrap(), dims.as_slice()),
    );

    let mut session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-squeeze.onnx",
    ))
    .expect("session did not create");
    let result = pollster::block_on(wonnx::run(&mut session, input_data)).unwrap();
    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    assert_eq!(probabilities[0].0, 22);
}

pub fn load_image(
    image_path: &str,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/data/images")
            .join(image_path),
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

pub fn load_squeezenet_image(
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/data/images")
            .join("bald_eagle.jpeg"),
    )
    .unwrap()
    .resize_exact(224 as u32, 224 as u32, FilterType::Nearest)
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
