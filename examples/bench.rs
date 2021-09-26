use ndarray::*;
use rayon::prelude::*;
use std::time::Instant;

fn matmul() -> Result<f32, ShapeError> {
    let n = 1024 * 4;
    let n2 = n * n;
    let a: Vec<f32> = (0..n2).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..n2).map(|_| 4.0 as f32).collect();
    let a = Array::from_shape_vec((n, n), a)?;
    let b = Array::from_shape_vec((n, n), b)?;

    let mut c = a.dot(&b);
    for _ in 0..20 {
        c = c + a.dot(&b);
    }
    Ok(c[[0, 0]])
}
fn main() {
    const N: i32 = i32::pow(2, 17);
    let time_start = Instant::now();
    let v = vec![1.0f32; N as _];
    let sum: f32 = v.par_iter().sum();
    println!("{}", sum);
    let time_sum = Instant::now();
    println!("time: sum -> start: {:#?}", time_sum - time_start);
    let mut v = vec![1.0f32; 512 * 512 * 128];
    for _ in 0..1 {
        v = v.iter().map(|x| f32::cos(*x)).collect();
    }
    let time_cos = Instant::now();
    println!("time: sum -> cos : {:#?}", time_cos - time_sum);
    let mut v = vec![1.0f32; 512 * 512 * 128];
    for _ in 0..1 {
        v = v.par_iter().map(|x| f32::max(*x, 0.0)).collect();
    }
    let time_cos_parallel = Instant::now();
    println!(
        "time: cos_parallel -> cos : {:#?}",
        time_cos_parallel - time_cos
    );
    let now = Instant::now();
    let result = matmul().unwrap();
    println!("result: {:#?}", result);
    println!("MatMul: {:#?}", Instant::now() - now);
}
