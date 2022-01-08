use std::collections::HashMap;
// Indicates a f32 overflow in an intermediate Collatz value
// use wasm_bindgen_test::*;
use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use log::info;
use ndarray::s;
use std::time::Instant;
use std::{
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Duration,
};
use wonnx::WonnxError;

// Args Management
async fn run() {
    let probabilities = &execute_gpu().await.unwrap();

    let (_, probabilities) = probabilities.iter().next().unwrap();

    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let class_labels = get_imagenet_labels();

    for i in 0..10 {
        //      println!(
        //          "probabilities[i*9..(i+1)*9]: {:#?}",
        //          &probabilities[i * 12..(i + 1) * 12]
        //      );
        println!(
            "Infered result: {} of class: {}",
            class_labels[probabilities[i].0], probabilities[i].0
        );
    }
}

// Hardware management
async fn execute_gpu() -> Result<HashMap<String, Vec<f32>>, WonnxError> {
    let mut input_data = HashMap::new();
    let image = load_image();
    input_data.insert("data".to_string(), image.as_slice().unwrap());

    let session = wonnx::Session::from_path("examples/data/models/opt-squeeze.onnx").await?;
    let time_pre_compute = Instant::now();
    info!("Start Compute");
    let result = session.run(input_data.clone()).await?;
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
            .join("examples/data/images")
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
        .join("examples/data/models")
        .join("synset.txt");
    if !labels_path.exists() {
        let url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt";
        println!("Downloading {:?} to {:?}...", url, labels_path);
        let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()
            .map_err(Box::new)
            .unwrap();

        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();
        println!("Downloading {} bytes...", len);

        let mut reader = resp.into_reader();

        let f = fs::File::create(&labels_path).unwrap();
        let mut writer = io::BufWriter::new(f);

        let bytes_io_count = io::copy(&mut reader, &mut writer).unwrap();

        assert_eq!(bytes_io_count, len as u64);
    }
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}
