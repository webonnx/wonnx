use rayon::prelude::*;

fn main() {
    const n: i32 = i32::pow(2, 21);
    let v = vec![1.0f32; n as _];
    let sum: f32 = v.par_iter().sum();
    println!("sum: {:#?}", sum);
}
