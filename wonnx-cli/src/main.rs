use crate::info::info_table;
use info::print_graph;
use prettytable::{cell, row, Table};
use protobuf::{self, Message};
use std::collections::HashMap;
use structopt::StructOpt;
use wonnx::onnx::ModelProto;
use wonnx::utils::{OutputTensor, Shape};
use wonnx_preprocessing::text::{get_lines, EncodedText};
use wonnx_preprocessing::Tensor;

mod gpu;
mod info;
mod types;
mod utils;

use crate::types::*;

async fn run() -> Result<(), NNXError> {
    env_logger::init();
    let opt = Opt::from_args();

    match opt.cmd {
        Command::Devices => {
            let instance = wgpu::Instance::new(wgpu::Backends::all());
            let adapters = instance.enumerate_adapters(wgpu::Backends::all());
            let mut adapters_table = Table::new();
            adapters_table.add_row(row![b->"Adapter", b->"Vendor", b->"Backend"]);
            for adapter in adapters {
                let info = adapter.get_info();
                adapters_table.add_row(row![info.name, info.vendor, format!("{:?}", info.backend)]);
            }
            adapters_table.printstd();
            Ok(())
        }

        Command::Info(info_opt) => {
            // Load the model
            let model_path = info_opt
                .model
                .into_os_string()
                .into_string()
                .expect("invalid path");
            let model = ModelProto::parse_from_bytes(
                &std::fs::read(model_path).expect("ONNX Model path not found."),
            )
            .expect("Could not deserialize the model");
            let table = info_table(&model)?;
            table.printstd();
            Ok(())
        }

        Command::Graph(info_opt) => {
            // Load the model
            let model_path = info_opt
                .model
                .into_os_string()
                .into_string()
                .expect("invalid path");
            let model = ModelProto::parse_from_bytes(
                &std::fs::read(model_path).expect("ONNX Model path not found."),
            )
            .expect("Could not deserialize the model");
            print_graph(&model);
            Ok(())
        }

        Command::Infer(infer_opt) => infer_command(infer_opt).await,
    }
}

fn print_qa_output(
    infer_opt: &InferOptions,
    qa_encoding: &EncodedText,
    mut outputs: HashMap<String, OutputTensor>,
) -> Result<(), NNXError> {
    let start_output: Vec<f32> = outputs
        .remove(&infer_opt.qa_answer_start)
        .ok_or_else(|| NNXError::OutputNotFound(infer_opt.qa_answer_start.to_string()))?
        .try_into()?;

    let end_output: Vec<f32> = outputs
        .remove(&infer_opt.qa_answer_end)
        .ok_or_else(|| NNXError::OutputNotFound(infer_opt.qa_answer_start.to_string()))?
        .try_into()?;

    println!(
        "{}",
        qa_encoding
            .get_answer(
                &start_output,
                &end_output,
                infer_opt.context.as_ref().unwrap()
            )
            .text
    );
    Ok(())
}

fn print_output(
    infer_opt: &InferOptions,
    output_name: &str,
    output: OutputTensor,
    print_output_names: bool,
    print_newlines: bool,
) {
    // Look up label
    match &infer_opt.labels {
        Some(labels_path) => {
            if print_output_names {
                println!("{}: ", output_name);
            }
            let labels = get_lines(labels_path);

            let output_slice: Vec<f32> = output.try_into().unwrap();
            let mut probabilities = output_slice.iter().enumerate().collect::<Vec<_>>();
            probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

            let top = infer_opt.top.unwrap_or(10);
            for i in 0..top.min(labels.len()) {
                if infer_opt.probabilities {
                    println!("{}: {}", labels[probabilities[i].0], probabilities[i].1);
                } else {
                    println!("{}", labels[probabilities[i].0]);
                }
            }
        }
        None => {
            if print_output_names {
                print!("{}: ", output_name);
            }

            // Just print the output tensor values, one a line
            match output {
                wonnx::utils::OutputTensor::F32(fs) => {
                    for i in fs {
                        if print_newlines {
                            println!("{:.3}", i);
                        } else {
                            print!("{:.3} ", i);
                        }
                    }
                }
                wonnx::utils::OutputTensor::I32(ints) => {
                    for i in ints {
                        if print_newlines {
                            println!("{}", i);
                        } else {
                            print!("{}", i);
                        }
                    }
                }
                wonnx::utils::OutputTensor::I64(ints) => {
                    for i in ints {
                        if print_newlines {
                            println!("{}", i);
                        } else {
                            print!("{}", i);
                        }
                    }
                }
            }
        }
    }

    if !print_newlines {
        println!();
    }
}

async fn infer_command(infer_opt: InferOptions) -> Result<(), NNXError> {
    // Load the model
    let model_path = infer_opt
        .model
        .clone()
        .into_os_string()
        .into_string()
        .expect("invalid path");
    let model = ModelProto::parse_from_bytes(
        &std::fs::read(&model_path).expect("ONNX Model path not found."),
    )
    .expect("Could not deserialize the model");

    let inference_input = InferenceInput::new(&infer_opt, &model)?;

    // Determine which outputs we will be reading
    let mut output_names = infer_opt.output_name.clone();
    if output_names.is_empty() {
        for output in model.get_graph().get_output() {
            output_names.push(output.get_name().to_string());
        }
        log::info!("no outputs given; using {:?}", output_names);
    }

    #[cfg(feature = "cpu")]
    if infer_opt.compare {
        return infer_compare(&model_path, inference_input, infer_opt, output_names, model).await;
    }

    let first_result = async {
        let compile_start = std::time::Instant::now();
        let backend = infer_opt
            .backend
            .inferer_for_model(
                &model_path,
                &inference_input.input_shapes,
                Some(output_names.clone()),
            )
            .await?;
        log::info!(
            "compile phase took {}ms",
            compile_start.elapsed().as_millis()
        );

        if infer_opt.benchmark {
            let benchmark_start = std::time::Instant::now();
            for _ in 0..100 {
                let _ = backend
                    .infer(&output_names, &inference_input.inputs, &model)
                    .await?;
            }
            let benchmark_time = benchmark_start.elapsed();
            println!(
                "time for 100 inferences: {}ms ({}/s)",
                benchmark_time.as_millis(),
                1000 / (benchmark_time.as_millis() / 100)
            );
        }

        let infer_start = std::time::Instant::now();
        let res = backend
            .infer(&output_names, &inference_input.inputs, &model)
            .await;
        log::info!("infer phase took {}ms", infer_start.elapsed().as_millis());
        res
    };

    let mut output_tensors = match first_result.await {
        Ok(x) => x,
        Err(e) => {
            #[cfg(feature = "cpu")]
            if infer_opt.fallback {
                match infer_opt.backend.fallback() {
                    Some(fallback_backend) => {
                        log::warn!(
                            "inference with {:?} backend failed: {}",
                            infer_opt.backend,
                            e,
                        );
                        log::warn!("trying {:?} backend instead", fallback_backend);
                        let fallback_inferer = fallback_backend
                            .inferer_for_model(
                                &model_path,
                                &inference_input.input_shapes,
                                Some(output_names.clone()),
                            )
                            .await?;
                        fallback_inferer
                            .infer(&output_names, &inference_input.inputs, &model)
                            .await?
                    }
                    None => return Err(e),
                }
            } else {
                return Err(e);
            }

            #[cfg(not(feature = "cpu"))]
            return Err(e);
        }
    };

    if infer_opt.qa_answer {
        // Print outputs as QA answer
        print_qa_output(
            &infer_opt,
            &inference_input.qa_encoding.unwrap(),
            output_tensors,
        )?;
    } else {
        // Print outputs individually
        let print_output_names = output_names.len() > 1;
        let print_newlines = !print_output_names;

        for output_name in &output_names {
            let output = output_tensors.remove(output_name).unwrap();
            print_output(
                &infer_opt,
                output_name,
                output,
                print_output_names,
                print_newlines,
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cpu")]
mod cpu;

impl Backend {
    #[cfg(feature = "cpu")]
    fn fallback(&self) -> Option<Backend> {
        match self {
            #[cfg(feature = "cpu")]
            Backend::Cpu => None,

            Backend::Gpu => {
                #[cfg(feature = "cpu")]
                return Some(Backend::Cpu);

                #[cfg(not(feature = "cpu"))]
                return None;
            }
        }
    }

    async fn inferer_for_model(
        &self,
        model_path: &str,
        #[allow(unused_variables)] input_shapes: &HashMap<String, Shape>,
        outputs: Option<Vec<String>>,
    ) -> Result<Box<dyn Inferer>, NNXError> {
        Ok(match self {
            Backend::Gpu => Box::new(gpu::GPUInferer::new(model_path, outputs).await?),
            #[cfg(feature = "cpu")]
            Backend::Cpu => Box::new(cpu::CPUInferer::new(model_path, input_shapes).await?),
        })
    }
}

#[cfg(feature = "cpu")]
async fn infer_compare(
    model_path: &str,
    inference_input: InferenceInput,
    infer_opt: InferOptions,
    output_names: Vec<String>,
    model: ModelProto,
) -> Result<(), NNXError> {
    let gpu_backend = Backend::Gpu
        .inferer_for_model(
            model_path,
            &inference_input.input_shapes,
            Some(output_names.clone()),
        )
        .await?;
    let gpu_start = std::time::Instant::now();
    if infer_opt.benchmark {
        for _ in 0..100 {
            let _ = gpu_backend
                .infer(&output_names, &inference_input.inputs, &model)
                .await?;
        }
    }
    let gpu_output_tensors = gpu_backend
        .infer(&output_names, &inference_input.inputs, &model)
        .await?;
    let gpu_time = gpu_start.elapsed();
    log::info!("gpu time: {}ms", gpu_time.as_millis());
    drop(gpu_backend);

    let cpu_backend = Backend::Cpu
        .inferer_for_model(
            model_path,
            &inference_input.input_shapes,
            Some(output_names.clone()),
        )
        .await?;
    let cpu_start = std::time::Instant::now();
    if infer_opt.benchmark {
        for _ in 0..100 {
            let _ = cpu_backend
                .infer(&output_names, &inference_input.inputs, &model)
                .await?;
        }
    }
    let cpu_output_tensors = cpu_backend
        .infer(&output_names, &inference_input.inputs, &model)
        .await?;
    let cpu_time = cpu_start.elapsed();
    log::info!(
        "cpu time: {}ms ({:.2}x gpu time)",
        cpu_time.as_millis(),
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
    if gpu_output_tensors.len() != cpu_output_tensors.len() {
        return Err(NNXError::Comparison(format!(
            "number of outputs in GPU result ({}) mismatches CPU result ({})",
            gpu_output_tensors.len(),
            cpu_output_tensors.len()
        )));
    }

    for output_name in &output_names {
        let cpu_output: Vec<f32> = cpu_output_tensors[output_name].clone().try_into()?;
        let gpu_output: Vec<f32> = gpu_output_tensors[output_name].clone().try_into()?;
        log::info!(
            "comparing output {} (gpu_len={}, cpu_len={})",
            output_name,
            gpu_output.len(),
            cpu_output.len()
        );

        for i in 0..gpu_output.len() {
            let diff = (gpu_output[i] - cpu_output[i]).abs();
            println!(
                "{};{};{}",
                gpu_output[i],
                cpu_output[i],
                (gpu_output[i] - cpu_output[i])
            );
            if diff > 0.001 {
                return Err(NNXError::Comparison(format!(
							"output {}: element {} differs too much: GPU says {} vs CPU says {} (difference is {})",
							output_name, i, gpu_output[i], cpu_output[i], diff
						)));
            }
        }
    }

    if infer_opt.benchmark {
        println!(
            "OK (gpu={}ms, cpu={}ms, {:.2}x)",
            gpu_time.as_millis(),
            cpu_time.as_millis(),
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );
    } else {
        println!("OK")
    }
    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    std::process::exit(match pollster::block_on(run()) {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Error: {}", err);
            1
        }
    });
}
