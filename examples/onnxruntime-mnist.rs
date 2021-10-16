use onnxruntime::{
    environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel,
};
use std::time::Instant;
type Error = Box<dyn std::error::Error>;
use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    // Setup the example's log level.
    // NOTE: ONNX Runtime's log level is controlled separately when building the environment.

    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        //   .use_cuda(0)?
        .with_optimization_level(GraphOptimizationLevel::DisableAll)?
        .with_number_threads(1)?
        // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8),
        //       _not_ SqueezeNet 1.1 as downloaded by '.with_model_downloaded(ImageClassification::SqueezeNet)'
        //       Obtain it with:
        //          curl -LO "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
        .with_model_from_file("examples/data/models/opt-mnist.onnx")?;

    let output0_shape: Vec<usize> = session.outputs[0]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();
    // initialize input data with values in [0.0, 1.0]
    let array = load_image();
    let input_tensor_values = vec![array];

    let time_pre_compute = Instant::now();
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

    let time_post_compute = Instant::now();
    println!(
        "time: post_compute: {:#?}",
        time_post_compute - time_pre_compute
    );
    assert_eq!(outputs[0].shape(), output0_shape.as_slice());
    for i in 0..10 {
        println!("Score for class [{}] =  {}", i, outputs[0][[0, i]]);
    }

    Ok(())
}

pub fn load_image() -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("5.jpg"),
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
