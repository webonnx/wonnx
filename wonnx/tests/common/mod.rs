use approx::assert_ulps_eq;

/// Assert two vectors are equal up to a specific number of units in last place (ULPS)
pub fn assert_eq_vector(xs: &[f32], ys: &[f32]) {
    assert_eq!(xs.len(), ys.len());
    for i in 0..xs.len() {
        assert_ulps_eq!(xs[i], ys[i], max_ulps = 2);
    }
}
