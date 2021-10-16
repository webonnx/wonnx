pub fn len(dims: &[i64]) -> i64 {
    dims.iter().product::<i64>()
}
