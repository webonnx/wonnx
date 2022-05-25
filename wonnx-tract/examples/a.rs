use futures::executor::block_on;
use tract_core::prelude::{DatumType, Tensor};
use tract_data::tvec;
use tract_tensorflow::prelude::*;
use wonnx_tract::GpuAccel;

fn main() {
    let gpu = block_on(GpuAccel::default()).unwrap();

    let x = 2;
    let y = 2;
    let z = 2;
    let w = 2;
    let mut data = Vec::new();
    for i in 1..(x * y * z * w + 1) {
        data.push(i as f32);
    }

    let inp = gpu.import_tensor(
        "inp".to_string(),
        &Tensor::from_shape(&tvec![x, y, z, w], &data).unwrap(),
    );
    let a = gpu.create_storage_tensor("a".to_string(), DatumType::F32, tvec![x, y, z, w]);
    let out = gpu.create_out_tensor("out".to_string(), DatumType::F32, tvec![x, y, z, w]);

    gpu.tanh(&inp, &a);
    gpu.sigmoid(&a, &out);

    println!("{:#?}", block_on(gpu.tensor_move_out(out)).dump(true));

    let model = tract_tensorflow::tensorflow()
        .model_for_path("mobilenet_v2_1.4_224_frozen.pb")
        .unwrap()
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)),
        )
        .unwrap()
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap();
    // GPU state
    println!("{:#?}", SimplePlan::new(model.clone()));
}
