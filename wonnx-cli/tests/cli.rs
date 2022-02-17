use assert_cmd::prelude::*;
use std::{path::Path, process::Command};

#[test]
fn simple_inference() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("nnx")?;

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/models/single_relu.onnx");

    cmd.arg("infer")
        .arg(model_path)
        .arg("-r")
        .arg("x=99,-99")
        .assert()
        .success();
    Ok(())
}
