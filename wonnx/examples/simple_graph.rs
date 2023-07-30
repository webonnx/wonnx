use std::collections::HashMap;
use wonnx::builder::*;

fn main() {
    env_logger::init();
    pollster::block_on(run()).unwrap();
}

async fn run() -> Result<(), WonnxError> {
    let result = execute_gpu().await?;
    let result = result.into_iter().next().unwrap().1;

    println!("{:#?}", result);

    assert_eq!(
        result,
        TensorData::F32(vec![54., 63., 72., 99., 108., 117., 144., 153., 162.].into())
    );
    Ok(())
}

async fn execute_gpu() -> Result<HashMap<String, TensorData<'static>>, SessionError> {
    // Hyperparameters
    let n = 5;
    let c = 1;
    let kernel_n = 3;
    let m = 1;

    let data_w: Vec<f32> = (0..m * c * kernel_n * kernel_n).map(|_| 1.0f32).collect();

    let session = {
        let input_x = input("X", ScalarType::F32, &[1, c, n, n]);
        let weights = tensor("W", &[m, c, 3, 3], data_w.into());
        let conv = input_x.conv(&weights, &[3, 3], &[m, c, 3, 3]);
        session_for_outputs(&["Y"], &[conv], 13).await?
    };

    let mut input_data = HashMap::new();
    let data: Vec<f32> = (0..25).map(|x| x as f32).collect();
    input_data.insert("X".to_string(), data.as_slice().into());
    Ok(session.run(&input_data).await?.to_owned())
}
