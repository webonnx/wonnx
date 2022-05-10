use std::path::Path;

use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use ndarray::s;

// Loads an image as (1,1,w,h) with pixels ranging 0...1 for 0..255 pixel values
pub fn load_bw_image(
    image_path: &Path,
    width: usize,
    height: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
        .unwrap()
        .resize_exact(width as u32, height as u32, FilterType::Nearest)
        .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    ndarray::Array::from_shape_fn((1, 1, width, height), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    })
}

// Loads an image as (1, 3, w, h)
pub fn load_rgb_image(
    image_path: &Path,
    width: usize,
    height: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    log::info!("load_rgb_image {:?} {}x{}", image_path, width, height);
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
        .unwrap()
        .resize_to_fill(width as u32, height as u32, FilterType::Nearest)
        .to_rgb8();

    // Python:
    // # image[y, x, RGB]
    // # x==0 --> left
    // # y==0 --> top

    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // WARNING: Note order of declaration of arguments: (_,c,j,i)
    let mut array = ndarray::Array::from_shape_fn((1, 3, height, width), |(_, c, j, i)| {
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
