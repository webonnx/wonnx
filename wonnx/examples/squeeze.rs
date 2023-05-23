use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use log::info;
use ndarray::s;
use std::collections::HashMap;
use std::convert::TryInto;
use std::time::Instant;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use wonnx::utils::TensorData;
use wonnx::WonnxError;

// Args Management
async fn run() {
    let probabilities = execute_gpu().await.unwrap();
    let probabilities = probabilities.into_iter().next().unwrap().1;
    let probabilities: Vec<f32> = probabilities.try_into().unwrap();
    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();
    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let class_labels = get_imagenet_labels();

    for i in 0..10 {
        println!(
            "Infered result: {} of class: {}",
            class_labels[probabilities[i].0], probabilities[i].0
        );
    }
}

// Hardware management
async fn execute_gpu<'a>() -> Result<HashMap<String, TensorData<'a>>, WonnxError> {
    let mut input_data = HashMap::new();
    let image = load_image();
    input_data.insert("data".to_string(), image.as_slice().unwrap().into());

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/models")
        .join("opt-squeeze.onnx");
    let session = wonnx::Session::from_path(model_path).await?;
    let time_pre_compute = Instant::now();
    info!("Start Compute");
    let result = session.run(&input_data).await?;
    let time_post_compute = Instant::now();
    println!(
        "time: first_prediction: {:#?}",
        time_post_compute - time_pre_compute
    );
    Ok(result)
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let time_pre_compute = Instant::now();

        pollster::block_on(run());
        let time_post_compute = Instant::now();
        println!("time: main: {:#?}", time_post_compute - time_pre_compute);
    }
    #[cfg(target_arch = "wasm32")]
    {
        // std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        //  console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}

pub fn load_image() -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let args: Vec<String> = std::env::args().collect();
    let image_path = if args.len() == 2 {
        Path::new(&args[1]).to_path_buf()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/images")
            .join("pelican.jpeg")
    };

    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
        .unwrap()
        .resize_to_fill(224, 224, FilterType::Nearest)
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

fn get_imagenet_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/models")
        .join("squeeze-labels.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}
