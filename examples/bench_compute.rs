use ndarray::*;
use rayon::prelude::*;
use std::time::Instant;

fn matmul() -> Result<f32, ShapeError> {
    let n = 1024;
    let n2 = n * n;
    let a: Vec<f32> = (0..n2).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..n2).map(|x| 4.0 as f32).collect();
    let a = Array::from_shape_vec((n, n), a)?;
    let b = Array::from_shape_vec((n, n), b)?;

    let mut c = a.dot(&b);
    for i in 0..10 {
        c = c + a.dot(&b);
    }
    Ok(c[[0, 0]])
}
fn main() {
    const N: i32 = i32::pow(2, 17);
    let now = Instant::now();
    let v = vec![1.0f32; N as _];
    let sum: f32 = v.par_iter().sum();
    println!("sum: {:#?}", sum);
    println!("Instant::now() - now: {:#?}", Instant::now() - now);
    let now = Instant::now();
    let result = matmul().unwrap();
    println!("result: {:#?}", result);
    println!("Instant::now() - now: {:#?}", Instant::now() - now);
}
