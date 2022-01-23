cd py-wonnx
rustup override set nightly-2022-01-01
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1,+popcnt'
maturin publish \
  --skip-existing \
  --username __token__