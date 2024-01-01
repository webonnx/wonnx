//! Compiles individual ONNX ops to a WebGPU shader using WGSL templates
use std::sync::OnceLock;

use crate::utils::{
    ceil, AttributeNotFoundError, DataTypeError, MultiType, NodeAttributes, ScalarType, Shape,
};
use num::integer::{gcd, Roots};
use tera::{Context, Tera};
use thiserror::Error;

/// The maximum number of threads that can be spawned in each dimension, according to the WebGPU specification. See
/// <https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsperdimension>
pub const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// The maximum workgroup size per dimension (see <https://www.w3.org/TR/webgpu/#dom-supported-limits-maxcomputeworkgroupsizex>)
pub const MAX_WORKGROUP_SIZE_X: u32 = 256;
pub const MAX_WORKGROUP_SIZE_Y: u32 = 256;
// pub const MAX_WORKGROUP_SIZE_Z: u32 = 64;

static TEMPLATES: OnceLock<Tera> = OnceLock::new();

fn get_templates() -> &'static Tera {
    TEMPLATES.get_or_init(|| {
        let mut tera = Tera::default();
        tera.add_raw_template(
            "endomorphism/activation.wgsl",
            include_str!("../templates/endomorphism/activation.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/arithmetic.wgsl",
            include_str!("../templates/endomorphism/arithmetic.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/batchnormalization.wgsl",
            include_str!("../templates/endomorphism/batchnormalization.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/softmax.wgsl",
            include_str!("../templates/endomorphism/softmax.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/map.wgsl",
            include_str!("../templates/endomorphism/map.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/cast.wgsl",
            include_str!("../templates/endomorphism/cast.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/concat.wgsl",
            include_str!("../templates/matrix/concat.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm_1.wgsl",
            include_str!("../templates/matrix/gemm_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/gemm.wgsl",
            include_str!("../templates/matrix/gemm.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/pad.wgsl",
            include_str!("../templates/matrix/pad.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/resize.wgsl",
            include_str!("../templates/matrix/resize.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/split.wgsl",
            include_str!("../templates/matrix/split.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "matrix/transpose.wgsl",
            include_str!("../templates/matrix/transpose.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/aggregate.wgsl",
            include_str!("../templates/pool/aggregate.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_1.wgsl",
            include_str!("../templates/pool/conv_kernel_1.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv_kernel_3.wgsl",
            include_str!("../templates/pool/conv_kernel_3.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/conv.wgsl",
            include_str!("../templates/pool/conv.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "pool/reduce.wgsl",
            include_str!("../templates/pool/reduce.wgsl"),
        )
        .unwrap();
        tera.add_raw_template("structs.wgsl", include_str!("../templates/structs.wgsl"))
            .unwrap();
        tera.add_raw_template(
            "snippets/activation_vec.wgsl",
            include_str!("../templates/snippets/activation_vec.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "snippets/activation_scalar.wgsl",
            include_str!("../templates/snippets/activation_scalar.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/gather.wgsl",
            include_str!("../templates/endomorphism/gather.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/onehot.wgsl",
            include_str!("../templates/endomorphism/onehot.wgsl"),
        )
        .unwrap();
        tera.add_raw_template(
            "endomorphism/broadcast.wgsl",
            include_str!("../templates/endomorphism/broadcast.wgsl"),
        )
        .unwrap();
        tera
    })
}

pub struct CompiledNode {
    pub shader: String,
    pub threads: (u32, u32, u32),
}

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("dimensions information missing for input/output '{0}' of node '{1}'. You may want to run onnx-simplifier on the model first.")]
    DimensionsMissing(String, String),

    #[error("attribute not found: {0}")]
    AttributeNotFound(#[from] AttributeNotFoundError),

    #[error("operation not recognized: {0}")]
    InvalidOperation(String),

    #[error("op {0} is not implemented yet! Check the README if you want to implement it")]
    UnimplementedOp(String),

    #[error("'{variant}' is not yet implemented for op {op}")]
    UnimplementedVariant { variant: String, op: String },

    #[error("the opset version {0} is not supported")]
    UnsupportedOpsetVersion(i64),

    #[error("the value '{value}' is invalid for attribute '{attribute}' (opset version {opset_version})")]
    InvalidAttributeValue {
        attribute: String,
        value: String,
        opset_version: i64,
    },

    #[error("input {input_index} has invalid shape {input_shape}")]
    InvalidInputShape {
        input_index: usize,
        input_shape: Shape,
    },

    #[error("the model exceeds the limit for {0}: {1} > {2}")]
    ComputeLimitExceeded(String, u32, u32),

    #[error("cannot determine data type to use: {0} or {1}")]
    TypesDisagree(ScalarType, ScalarType),

    #[error("cannot infer data type to use")]
    TypeUnderspecified,

    #[error("invalid type encountered: {0}")]
    InvalidType(#[from] DataTypeError),

    #[error("expected {expected} inputs, but there are only {actual}")]
    InvalidInputCount { expected: usize, actual: usize },

    #[error("cannot broadcast inputs to specified output dimensions")]
    InvalidBroadcast {
        input_shapes: Vec<Shape>,
        output_shape: Shape,
    },
}

struct NodeTemplate {
    scalar_type: ScalarType,
    template: &'static str,
    threads: (u32, u32, u32),
}

/// Returns the data type of the input and output shapes, but error if these types differ or when no input/output was specified
fn agreed_type(
    input_shapes: &[&Shape],
    output_shapes: &[&Shape],
) -> Result<ScalarType, CompileError> {
    let mut data_type: Option<ScalarType> = None;

    for input_shape in input_shapes {
        let input_type = input_shape.data_type;
        match data_type {
            Some(dt) => {
                if dt != input_type {
                    return Err(CompileError::TypesDisagree(dt, input_type));
                }
            }
            None => data_type = Some(input_type),
        }
    }

    for output_shape in output_shapes {
        let output_type = output_shape.data_type;
        match data_type {
            Some(dt) => {
                if dt != output_type {
                    return Err(CompileError::TypesDisagree(dt, output_type));
                }
            }
            None => data_type = Some(output_type),
        }
    }

    if let Some(ScalarType::I64) = data_type {
        data_type = Some(ScalarType::I32);
    }

    data_type.ok_or(CompileError::TypeUnderspecified)
}

pub fn compile(
    node: &crate::onnx::NodeProto,
    input_shapes: &[&Shape],
    output_shapes: &[&Shape],
    opset_version: i64,
) -> Result<CompiledNode, CompileError> {
    let input_lengths = input_shapes
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    let output_lengths = output_shapes
        .iter()
        .map(|shape| shape.element_count())
        .collect::<Vec<u64>>();

    let input_chunks: Vec<Vec<u64>> = input_shapes.iter().map(|d| d.chunks()).collect();
    let output_chunks: Vec<Vec<u64>> = output_shapes.iter().map(|d| d.chunks()).collect();
    let i_dims: Vec<&Vec<u64>> = input_shapes.iter().map(|s| &s.dims).collect();
    let o_dims: Vec<&Vec<u64>> = output_shapes.iter().map(|s| &s.dims).collect();

    let mut context = Context::new();
    context.insert("i_lens", &input_lengths);
    context.insert("o_lens", &output_lengths);
    context.insert("i_shape", &i_dims);
    context.insert("o_shape", &o_dims);
    context.insert("i_chunks", &input_chunks);
    context.insert("o_chunks", &output_chunks);
    context.insert("op_type", &node.get_op_type());
    context.insert("opset_version", &opset_version);

    let node_template: NodeTemplate = match node.get_op_type() {
        op @ ("Reshape" | "Dropout" | "Identity" | "Flatten" | "Squeeze" | "Unsqueeze") => {
            // These ops should all be optimized away earlier
            return Err(CompileError::InvalidOperation(op.to_string()));
        }

        // Map simple function
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" | "Reciprocal" | "Acosh"
        | "Asinh" | "Atanh" | "Neg" => {
            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 4),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);
            NodeTemplate {
                scalar_type: agreed_type(input_shapes, output_shapes)?,
                template: "endomorphism/map.wgsl",
                threads: (x_threads, 1, 1),
            }
        }

        op @ ("ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd"
        | "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp"
        | "ReduceSumSquare") => {
            let all_axes: Vec<i64> = (0..(i_dims[0].len() as i64)).collect();
            let axes: Vec<i64> = node
                .get_attribute_value("axes", Some(all_axes))?
                .into_iter()
                .map(|idx| {
                    if idx < 0 {
                        (i_dims[0].len() as i64) + idx
                    } else {
                        idx
                    }
                })
                .collect();
            let scalar_type = agreed_type(&[input_shapes[0]], output_shapes)?;

            let dims_removed: Vec<i64> = input_shapes[0]
                .dims
                .iter()
                .enumerate()
                .map(|(idx, dim)| {
                    if axes.contains(&(idx as i64)) {
                        1
                    } else {
                        *dim as i64
                    }
                })
                .collect();
            let chunks_with_dims_preserved = Shape::from(scalar_type, &dims_removed).chunks();

            log::debug!(
                "reduce Op={} axes={:?} output_shape={:?} chunks_with_dims_preserved={:?} output_length={}",
                op,
                axes,
                o_dims[0],
                chunks_with_dims_preserved,
                output_lengths[0]
            );

            // The reduce shader will be invoked once for each scalar in the output (which represents one reduce operation)
            let (x_threads, workgroup_size_x) = workgroup_size(
                output_lengths[0],
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;

            context.insert("workgroup_size_x", &workgroup_size_x);
            context.insert("chunks_with_dims_preserved", &chunks_with_dims_preserved);
            context.insert("axes", &axes);

            NodeTemplate {
                scalar_type,
                template: "pool/reduce.wgsl",
                threads: (x_threads, 1, 1),
            }
        }

        "OneHot" => {
            // Currently only OneHot on the last axis is supported
            let axis = node.get_attribute_value("axis", Some(-1))?;
            if axis != -1 {
                return Err(CompileError::UnimplementedVariant {
                    variant: format!("axis={}", axis),
                    op: String::from("OneHot"),
                });
            }

            // Depth tensor must have exactly one element
            if input_shapes[1].element_count() != 1 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 1,
                    input_shape: input_shapes[1].clone(),
                });
            }

            // Values tensor must have exactly two elements
            if input_shapes[2].element_count() != 2 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 2,
                    input_shape: input_shapes[2].clone(),
                });
            }

            // OneHot will invoke once for each index
            let (x_threads, workgroup_size_x) = workgroup_size(
                input_lengths[0],
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);

            NodeTemplate {
                scalar_type: output_shapes[0].data_type,
                template: "endomorphism/onehot.wgsl",

                threads: (x_threads, 1, 1),
            }
        }

        "Gather" => {
            // Input 0 is data, input 1 is indices
            // Which axis to gather on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
            // Default is 0. See https://github.com/onnx/onnx/blob/main/docs/Operators.md#attributes-25
            let axis = node.get_attribute_value("axis", Some(0))?;
            if axis != 0 {
                return Err(CompileError::UnimplementedVariant {
                    variant: format!("axis={}", axis),
                    op: String::from("Gather"),
                });
            }

            let elements_per_index = input_chunks[0][0];
            let scalar_type = agreed_type(&input_shapes[0..1], output_shapes)?;
            let chunk_type = MultiType::for_size(elements_per_index as usize, scalar_type);
            let chunk_size = chunk_type.elements();

            // The X dimension represents the indexes
            let (x_threads, workgroup_size_x) = workgroup_size(
                input_lengths[1],
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;

            // The Y dimension represents the elements to copy for each index
            let (y_threads, workgroup_size_y) = workgroup_size(
                ceil(elements_per_index, chunk_size as u64),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_Y,
            )?;

            context.insert("chunk_type", &chunk_type.wgsl_type_name());
            context.insert("chunk_size", &chunk_size);
            context.insert("workgroup_size_x", &workgroup_size_x);
            context.insert("workgroup_size_y", &workgroup_size_y);

            NodeTemplate {
                scalar_type,
                template: "endomorphism/gather.wgsl",
                threads: (x_threads, y_threads, 1),
            }
        }

        "Cast" => {
            let cast_to_type =
                ScalarType::from_i32(node.get_attribute_value::<i64>("to", None)? as i32)?;

            if !cast_to_type.wgsl_supported() {
                return Err(CompileError::UnimplementedVariant {
                    variant: format!("with data type {} (WGSL limitation)", cast_to_type),
                    op: "Cast".to_string(),
                });
            }

            context.insert("cast_to_type", cast_to_type.wgsl_type_name());

            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 4),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);
            NodeTemplate {
                scalar_type: agreed_type(input_shapes, &[])?,
                template: "endomorphism/cast.wgsl",
                threads: (x_threads, 1, 1),
            }
        }

        "Softmax" => {
            let default_axis = match opset_version {
                1..=10 => 1,   // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-1
                11..=12 => 1, // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-11
                13..=15 => -1, // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#softmax-13
                _ => return Err(CompileError::UnsupportedOpsetVersion(opset_version)),
            };

            /* Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely
            describes the batch_size. From version 13 onwards, counting backwards is also allowed. */
            let mut axis = node.get_attribute_value("axis", Some(default_axis))?;
            if axis < 0 {
                if opset_version >= 13 {
                    axis += input_shapes[0].rank() as i64;
                } else {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "axis".to_string(),
                        value: format!("{}", axis),
                        opset_version,
                    });
                }
            }

            if axis >= (input_shapes[0].rank() as i64) || axis < 0 {
                return Err(CompileError::InvalidAttributeValue {
                    attribute: "axis".to_string(),
                    value: format!("{}", axis),
                    opset_version,
                });
            }

            let left_of_axis = input_shapes[0].dims[0..(axis as usize)]
                .iter()
                .product::<u64>();
            let axis_chunk = input_shapes[0].dims[(axis as usize)..]
                .iter()
                .product::<u64>();
            let right_of_axis_chunk = input_shapes[0].dims[((axis + 1) as usize)..]
                .iter()
                .product::<u64>();

            context.insert("axis_chunk", &axis_chunk);

            let (x_threads, workgroup_size_x) = workgroup_size(
                left_of_axis,
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;
            context.insert("workgroup_size_x", &workgroup_size_x);

            // For opset version < 11, Softmax simply aggregates all values in an axis layer. For later opsets, Softmax
            // calculates it per-layer.
            if opset_version < 11 {
                context.insert("workgroup_size_y", &1);
                NodeTemplate {
                    scalar_type: agreed_type(input_shapes, output_shapes)?,
                    template: "endomorphism/softmax.wgsl",
                    threads: (x_threads, 1, 1),
                }
            } else {
                context.insert("right_of_axis_chunk", &right_of_axis_chunk);
                context.insert("axis_dims", &input_shapes[0].dims[axis as usize]);
                let (y_threads, workgroup_size_y) = workgroup_size(
                    right_of_axis_chunk,
                    MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    MAX_WORKGROUP_SIZE_Y,
                )?;
                context.insert("workgroup_size_y", &workgroup_size_y);
                NodeTemplate {
                    scalar_type: agreed_type(input_shapes, output_shapes)?,
                    template: "endomorphism/softmax.wgsl",
                    threads: (x_threads, y_threads, 1),
                }
            }
        }

        // Arithmetic operation
        op @ ("Add" | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less"
        | "LessOrEqual" | "Mod" | "Mul" | "Or" | "Sub" | "Pow" | "PRelu") => {
            let broadcast = node.get_attribute_value("broadcast", Some(0))?;
            if broadcast != 0 {
                return Err(CompileError::UnimplementedVariant {
                    op: op.to_string(),
                    variant: "broadcast".to_string(),
                });
            }

            // Determine the operator to use in the shader for this op
            context.insert(
                "op_type",
                match op {
                    "Add" => "+",
                    "And" => "&",
                    "Div" => "/",
                    "Equal" => "==",
                    "Greater" => ">",
                    "GreaterOrEqual" => ">=",
                    "Less" => "<",
                    "LessOrEqual" => "<=",
                    "Mod" => "%",
                    "Mul" => "*",
                    "Or" => "|",
                    "Sub" => "-",
                    "Pow" => "Pow",
                    "PRelu" => "PRelu",
                    _ => {
                        return Err(CompileError::UnimplementedOp(
                            node.get_op_type().to_string(),
                        ))
                    }
                },
            );

            if input_shapes.len() == 2
                && (input_shapes[0] != output_shapes[0] || input_shapes[1] != output_shapes[0])
            {
                // We are likely broadcasting; check if the broadcast is valid. Compute the possible broadcast output shape
                let out_shape =
                    Shape::multi_broadcast(&[input_shapes[0].clone(), input_shapes[1].clone()])
                        .ok_or(CompileError::InvalidBroadcast {
                            input_shapes: input_shapes
                                .iter()
                                .map(|x| (*x).clone())
                                .collect::<Vec<Shape>>(),
                            output_shape: output_shapes[0].clone(),
                        })?;

                if &out_shape != output_shapes[0] {
                    return Err(CompileError::InvalidBroadcast {
                        input_shapes: input_shapes
                            .iter()
                            .map(|x| (*x).clone())
                            .collect::<Vec<Shape>>(),
                        output_shape: output_shapes[0].clone(),
                    });
                }

                let lhs_padded_shape = input_shapes[0].left_padded_to(1, out_shape.rank());
                let rhs_padded_shape = input_shapes[1].left_padded_to(1, out_shape.rank());
                context.insert("lhs_padded_shape", &lhs_padded_shape.dims);
                context.insert("rhs_padded_shape", &rhs_padded_shape.dims);
                context.insert("lhs_padded_chunks", &lhs_padded_shape.chunks());
                context.insert("rhs_padded_chunks", &rhs_padded_shape.chunks());

                log::debug!(
                    "padded shapes for broadcast: {:?}, {:?} => {:?}",
                    lhs_padded_shape,
                    rhs_padded_shape,
                    out_shape.dims
                );

                let (x_threads, workgroup_size_x) = workgroup_size(
                    output_lengths[0],
                    MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    MAX_WORKGROUP_SIZE_X,
                )?;
                context.insert("workgroup_size_x", &workgroup_size_x);

                NodeTemplate {
                    scalar_type: agreed_type(input_shapes, output_shapes)?,
                    template: "endomorphism/broadcast.wgsl",
                    threads: (x_threads, 1, 1),
                }
            } else if input_shapes[0] != output_shapes[0] {
                // If we are not broadcasting, the input shape needs to be equal to the output shape
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_shapes[0].clone(),
                });
            } else {
                // Not broadcasting
                let coefficient = node.get_attribute_value("coefficient", Some(1.0))?;
                context.insert("coefficient", &coefficient);

                let (x_threads, workgroup_size_x) = workgroup_size(
                    ceil(output_lengths[0], 4) as _,
                    MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    MAX_WORKGROUP_SIZE_X,
                )?;
                context.insert("workgroup_size_x", &workgroup_size_x);

                NodeTemplate {
                    scalar_type: agreed_type(input_shapes, output_shapes)?,
                    template: "endomorphism/arithmetic.wgsl",
                    threads: (x_threads, 1, 1),
                }
            }
        }
        // Not taking into account attributes
        "BatchNormalization" => {
            /* Prior to version 9, BatchNormalization supported a 'spatial' mode where input mean/variance are of shape
            [C,W,H] instead of just [C]. See https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-7.
            This mode is not supported. */
            if let Ok(spatial_value) = node.get_attribute_value::<i64>("spatial", None) {
                if opset_version < 9 {
                    return Err(CompileError::UnimplementedVariant {
                        op: "BatchNormalization".to_string(),
                        variant: "spatial".to_string(),
                    });
                } else {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "spatial".to_string(),
                        opset_version,
                        value: spatial_value.to_string(),
                    });
                }
            }

            // [N,C,w,h] => [N,C,w,h] where [w,h] is normalized using stats for each [N,C]
            // N and C are optional and assumed to be one for lower-rank inputs
            if input_shapes[0].rank() <= 2 || input_shapes[0].rank() > 4 {
                return Err(CompileError::UnimplementedVariant {
                    op: "BatchNormalization".to_string(),
                    variant: format!("with input {}", input_shapes[0]),
                });
            }

            let (input_batches, input_channels, input_w, input_h) = match input_shapes[0].rank() {
                2 => (1, 1, input_shapes[0].dim(0), input_shapes[0].dim(1)), // WxH, C=1, N=1
                3 => (
                    1,
                    input_shapes[0].dim(0),
                    input_shapes[0].dim(1),
                    input_shapes[0].dim(2),
                ), // CxWxH, single batch N=1
                4 => (
                    input_shapes[0].dim(0),
                    input_shapes[0].dim(1),
                    input_shapes[0].dim(2),
                    input_shapes[0].dim(3),
                ), // NxCxWxH
                _ => unreachable!(),
            };

            if input_batches == 0 || input_channels == 0 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_shapes[0].clone(),
                });
            }

            // If w*h is a multiple of 4, we can use vec4 in our shader
            let elem_type = MultiType::for_size((input_w * input_h) as usize, ScalarType::F32);

            context.insert("elem_type", &elem_type.wgsl_type_name());
            context.insert("elem_stride", &elem_type.stride());

            // The default for epsilon is 1e05, see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#attributes-252
            let epsilon = node.get_attribute_value("epsilon", Some(1e-05))?;
            context.insert("epsilon", &epsilon);
            context.insert(
                "batch_size",
                &ceil(
                    input_channels * input_w * input_h,
                    elem_type.elements() as u64,
                ),
            );
            context.insert(
                "channel_size",
                &ceil(input_w * input_h, elem_type.elements() as u64),
            );

            NodeTemplate {
                scalar_type: agreed_type(&input_shapes[0..1], &output_shapes[0..1])?,
                template: "endomorphism/batchnormalization.wgsl",
                threads: (
                    ceil(input_w * input_h, elem_type.elements() as u64) as _,
                    input_channels as _,
                    input_batches as _,
                ),
            }
        }
        op @ ("Relu" | "Sigmoid" | "Softsign" | "Softplus" | "Clip" | "Celu" | "Elu"
        | "LeakyRelu" | "HardSigmoid") => {
            let alpha = match op {
                "LeakyRelu" => node.get_attribute_value("alpha", Some(0.01))?,
                "HardSigmoid" => node.get_attribute_value("alpha", Some(0.2))?,
                _ => node.get_attribute_value("alpha", Some(1.0))?,
            };

            let beta = if op == "HardSigmoid" {
                node.get_attribute_value("beta", Some(0.5))?
            } else {
                node.get_attribute_value("beta", Some(1.0))?
            };

            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            if op == "Clip" {
                let min: Vec<f32> =
                    node.get_attribute_value("min", Some(vec![f32::NEG_INFINITY]))?;
                let max: Vec<f32> = node.get_attribute_value("max", Some(vec![f32::INFINITY]))?;
                if min.len() != 1 {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "min".into(),
                        value: format!("{min:?}"),
                        opset_version,
                    });
                }
                if max.len() != 1 {
                    return Err(CompileError::InvalidAttributeValue {
                        attribute: "max".into(),
                        value: format!("{max:?}"),
                        opset_version,
                    });
                }
                context.insert("min", &format!("{:.1}", min[0]));
                context.insert("max", &format!("{:.1}", max[0]));
            }

            let (x_threads, workgroup_size_x) = workgroup_size(
                ceil(output_lengths[0], 4),
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_X,
            )?;

            context.insert("workgroup_size_x", &workgroup_size_x);

            NodeTemplate {
                scalar_type: agreed_type(input_shapes, output_shapes)?,
                template: "endomorphism/activation.wgsl",
                threads: (x_threads, 1, 1),
            }
        }
        "Concat" => {
            let mut input_cumulative_len = vec![];
            let mut sum = 0;
            for len in input_lengths.iter() {
                sum += len;
                input_cumulative_len.push(sum);
            }
            context.insert("cum_len", &input_cumulative_len);

            let root = output_lengths[0].sqrt() + 1;
            let per_dim = ceil(root, 16) + 1;

            NodeTemplate {
                scalar_type: agreed_type(input_shapes, output_shapes)?,
                template: "matrix/concat.wgsl",
                threads: (per_dim as u32, per_dim as u32, 1),
            }
        }
        op @ ("MaxPool" | "AveragePool" | "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish"
        | "GlobalAveragePool") => {
            // TODO: Conv only support NxCxHxW for the moment.
            if input_shapes[0].rank() != 4 {
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_shapes[0].clone(),
                });
            }

            // GlobalAveragePool is equivalent to AveragePool, with the kernel shape set to the size of the input tensor
            // See https://github.com/onnx/onnx/blob/main/docs/Operators.md#globalaveragepool
            // Other attributes are not supported and also not relevant, and are simply ignored
            let is_global_average_pool = op == "GlobalAveragePool";
            if is_global_average_pool {
                // Generate shader code as if this were a regular AveragePool
                context.insert("op_type", "AveragePool");
            }

            let auto_pad = node.get_attribute_value("auto_pad", Some("NOTSET".to_string()))?;
            let dilations = node.get_attribute_value("dilations", Some(vec![1, 1]))?;
            let kernel_shape = if is_global_average_pool {
                vec![input_shapes[0].dim(2) as i64, input_shapes[0].dim(3) as i64]
            } else {
                node.get_attribute_value::<Vec<i64>>("kernel_shape", None)?
            };
            let strides = node.get_attribute_value("strides", Some(vec![1, 1]))?;
            let pads = node.get_attribute_value("pads", Some(vec![0, 0, 0, 0]))?;
            let count_include_pad = node.get_attribute_value("count_include_pad", Some(0))?;
            let group = node.get_attribute_value("group", Some(1))? as u64;

            let pads = match auto_pad.as_str() {
                "NOTSET" => pads.to_vec(),
                "SAME_UPPER" => {
                    let pad_0 = (output_shapes[0].dim(2) as i64 - 1) * strides[0] + kernel_shape[0]
                        - input_shapes[0].dim(3) as i64;
                    let pad_1 = (output_shapes[0].dim(2) as i64 - 1) * strides[1] + kernel_shape[1]
                        - input_shapes[0].dim(3) as i64;
                    vec![pad_0 / 2, pad_1 / 2]
                }
                "SAME_LOWER" => {
                    let pad_0 = (output_shapes[0].dim(2) as i64 - 1) * strides[0] + kernel_shape[0]
                        - input_shapes[0].dim(3) as i64;
                    let pad_1 = (output_shapes[0].dim(2) as i64 - 1) * strides[1] + kernel_shape[1]
                        - input_shapes[0].dim(3) as i64;
                    vec![pad_0 - pad_0 / 2, pad_1 - pad_1 / 2]
                }
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: op.to_string(),
                        variant: format!("auto_pad={}", auto_pad),
                    })
                }
            };

            let input_shape = &input_shapes[0];
            let output_shape = &output_shapes[0];
            assert!(kernel_shape.len() >= 2);
            assert!(kernel_shape[0] >= 0 && kernel_shape[1] >= 0);

            let channels_per_group = input_shape.dim(1) / group;
            if channels_per_group * group != input_shape.dim(1) {
                // Input channel count must be divisible by the group count.
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: (*input_shape).clone(),
                });
            }

            if output_shape.dim(0) != input_shape.dim(0) {
                // Output batch size != Input batch size.
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: (*input_shape).clone(),
                });
            }

            if input_shapes.len() >= 2 && output_shape.dim(1) != input_shapes[1].dim(0) {
                // Output feature map count != Filter count.
                return Err(CompileError::InvalidInputShape {
                    input_index: 1,
                    input_shape: (*input_shapes[1]).clone(),
                });
            }

            if input_shapes.len() >= 3 && input_shapes[2].dim(0) != input_shapes[1].dim(0) {
                // Bias count != Filter count.
                return Err(CompileError::InvalidInputShape {
                    input_index: 2,
                    input_shape: (*input_shapes[2]).clone(),
                });
            }

            context.insert("original_width", &input_shape.dim(3));
            context.insert("width", &output_shape.dim(3));
            context.insert("original_height", &input_shape.dim(2));
            context.insert("channel", &input_shape.dim(1));
            context.insert("groups", &group);
            context.insert("channels_per_group", &channels_per_group);
            context.insert("stride", &strides);
            context.insert("kernel_shape", &kernel_shape);
            context.insert("kernel_length", &(kernel_shape[0] * kernel_shape[1]));
            context.insert(
                "kernel_channel_len",
                &((kernel_shape[0] as u64) * (kernel_shape[1] as u64) * channels_per_group),
            );
            context.insert("pad", &pads);
            context.insert("count_include_pad", &count_include_pad);
            context.insert("dilation", &dilations);

            // GLSL shader for convolution computation
            match op {
                "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
                    if channels_per_group % 4 == 0 {
                        NodeTemplate {
                            scalar_type: agreed_type(input_shapes, &output_shapes[0..1])?,
                            template: "pool/aggregate.wgsl",
                            threads: (ceil(output_lengths[0], 1024) as _, 1, 1),
                        }
                    } else {
                        NodeTemplate {
                            scalar_type: agreed_type(input_shapes, &output_shapes[0..1])?,
                            template: "pool/aggregate.wgsl",
                            threads: (ceil(output_lengths[0], 256) as _, 1, 1),
                        }
                    }
                }
                "Conv" | "ConvRelu" | "ConvLeakyRelu" | "ConvMish" => {
                    // Alpha is the Leaky Relu attribute
                    let alpha = node.get_attribute_value("alpha", Some(0.01))?;
                    context.insert("alpha", &alpha);

                    let scalar_type = agreed_type(input_shapes, output_shapes)?;

                    // WGSL shader for convolution computation
                    // Matrixes in WGSL are only supported for floating point types, so we can only use these (faster) shader
                    // implementations when scalar_type is float
                    if (strides == [1, 1])
                        && (kernel_shape == [1, 1])
                        && (dilations == [1, 1] && (pads == [0, 0, 0, 0]))
                        && (channels_per_group % 16 == 0)
                        && (output_shape.dim(1) % 4 == 0)
                        && scalar_type.is_float()
                        && group == 1
                    {
                        NodeTemplate {
                            scalar_type: agreed_type(input_shapes, output_shapes)?,
                            template: "pool/conv_kernel_1.wgsl",
                            threads: (ceil(output_lengths[0], 1024) as _, 1, 1),
                        }
                    } else if (strides == [1, 1])
                        && (kernel_shape == [3, 3])
                        && (dilations == [1, 1])
                        && (output_shape.dim(1) % 4 == 0)
                        && scalar_type.is_float()
                        && group == 1
                    {
                        NodeTemplate {
                            scalar_type,
                            template: "pool/conv_kernel_3.wgsl",
                            threads: (ceil(output_lengths[0], 1024) as _, 1, 1),
                        }
                    } else {
                        NodeTemplate {
                            scalar_type: agreed_type(input_shapes, output_shapes)?,
                            template: "pool/conv.wgsl",
                            threads: (ceil(output_lengths[0], 256) as _, 1, 1),
                        }
                    }
                }
                _ => return Err(CompileError::InvalidOperation(op.to_string())),
            }
        }
        op @ ("Gemm" | "MatMul") => {
            // Generic matrix multiplication; outputs an M*N matrix from inputs A (size M*K) and B (size K*N)

            // MatMul behaves "like numpy.matmul" (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html)
            // If both arguments are 2-D they are multiplied like conventional matrices. If they are not, special rules are
            // applied, that may (among others) lead to 'stacking' (basically multiple matrix multiplications are performed
            // in parallel). Below, these rules are applied. These rewrite the shapes for input and output and in some cases
            // set a 'stack count' and 'stack stride'.
            let mut stack_count = 1;
            let mut input_left_shape = input_shapes[0].clone();
            let mut input_right_shape = input_shapes[1].clone();
            let mut output_shape = output_shapes[0].clone();
            let mut stack_left_stride: u64 = 0;
            let mut stack_right_stride: u64 = 0;
            let mut stack_output_stride: u64 = 0;

            if op == "MatMul" {
                // - If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes
                //   and broadcast accordingly.
                if input_left_shape.rank() > 2 || input_right_shape.rank() > 2 {
                    if input_left_shape.rank() != input_right_shape.rank()
                        || output_shape.rank() != input_left_shape.rank()
                        || input_left_shape.dims[0..(input_left_shape.dims.len() - 2)]
                            != input_right_shape.dims[0..(input_right_shape.dims.len() - 2)]
                        || input_left_shape.dims[0..(input_left_shape.dims.len() - 2)]
                            != output_shape.dims[0..(output_shape.dims.len() - 2)]
                    {
                        return Err(CompileError::UnimplementedVariant {
                            variant: format!("broadcasting for two stacks of matrixes (left side has shape {}, right side has shape {})", input_left_shape, input_right_shape),
                            op: op.to_string(),
                        });
                    }

                    let stack_dims = input_left_shape.dims.len() - 2;
                    stack_count = input_left_shape.dims[0..stack_dims].iter().product();
                    input_left_shape.dims.drain(0..stack_dims);
                    input_right_shape.dims.drain(0..stack_dims);
                    output_shape.dims.drain(0..stack_dims);
                    stack_left_stride = input_left_shape.dims.iter().product();
                    stack_right_stride = input_right_shape.dims.iter().product();
                    stack_output_stride = output_shape.dims.iter().product();

                    log::debug!(
                        "MatMul stacking: left {} right {} stack_dims={} stack_count={} stack_left_stride={} stack_right_stride={} stack_output_stride={}",
                        input_left_shape,
                        input_right_shape,
                        stack_dims,
                        stack_count,
                        stack_left_stride,
                        stack_right_stride,
                        stack_output_stride
                    );
                }

                // If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After
                // matrix multiplication the prepended 1 is removed.
                if input_left_shape.rank() == 1 {
                    input_left_shape.dims.insert(0, 1);
                }

                // If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After
                // matrix multiplication the appended 1 is removed.
                if input_right_shape.rank() == 1 {
                    input_left_shape.dims.push(1);
                }
            }

            context.insert("stack_left_stride", &stack_left_stride);
            context.insert("stack_right_stride", &stack_right_stride);
            context.insert("stack_output_stride", &stack_output_stride);
            context.insert("left_shape", &input_left_shape.dims);
            context.insert("right_shape", &input_right_shape.dims);
            context.insert("output_shape", &output_shape.dims);

            // Check dimensions
            let dim_m = output_shape.dim(0);
            let dim_n = output_shape.dim(1);
            let dim_k = input_left_shape.dim(1);
            if dim_m != input_left_shape.dim(0) {
                return Err(CompileError::InvalidInputShape {
                    input_index: 0,
                    input_shape: input_left_shape.clone(),
                });
            }

            if dim_n != input_right_shape.dim(1) || dim_k != input_right_shape.dim(0) {
                return Err(CompileError::InvalidInputShape {
                    input_index: 1,
                    input_shape: input_right_shape.clone(),
                });
            }

            if op == "Gemm" {
                // Check if A resp. B should be transposed, or C should be broadcast (default: 0 = false)
                let transpose_a = node.get_attribute_value("transA", Some(0))?;
                let transpose_b = node.get_attribute_value("transB", Some(0))?;
                let broadcast = node.get_attribute_value("broadcast", Some(0))?;

                if transpose_a != 0 || transpose_b != 0 || broadcast != 0 {
                    return Err(CompileError::UnimplementedVariant {
                        variant: "with transA/transB/broadcast not equal to zero".to_string(),
                        op: op.to_string(),
                    });
                }

                // If there is a bias input, it should be "unidirectionally broadcastable to M*N". Currently we only support a bias of M*N though
                if input_shapes.len() > 2 {
                    let mut bias_shape = input_shapes[2].clone();

                    // A shape of higher rank than 2 can never be broadcasted
                    if bias_shape.rank() > 2 || bias_shape.rank() == 0 {
                        return Err(CompileError::InvalidInputShape {
                            input_index: 2,
                            input_shape: bias_shape.clone(),
                        });
                    }

                    // For bias shape of rank 1, prepend a 1
                    if bias_shape.rank() == 1 {
                        bias_shape.dims.insert(0, 1);
                    }

                    // First dimension of bias must be either 1 or M
                    if bias_shape.dim(0) != dim_m && bias_shape.dim(0) != 1 {
                        return Err(CompileError::InvalidInputShape {
                            input_index: 2,
                            input_shape: bias_shape.clone(),
                        });
                    }

                    // Second dimension of bias must be either 1 or N
                    if bias_shape.dim(1) != dim_n && bias_shape.dim(1) != 1 {
                        return Err(CompileError::InvalidInputShape {
                            input_index: 2,
                            input_shape: bias_shape.clone(),
                        });
                    }

                    context.insert("bias_shape", &bias_shape.dims);
                    context.insert("bias_broadcast_rows", &(bias_shape.dim(0) == 1));
                    context.insert("bias_broadcast_columns", &(bias_shape.dim(1) == 1));
                }
            }

            // Due to a limitation in WGSL, we currently only support floating-point matrix multiplication
            // See https://github.com/gfx-rs/naga/issues/1896
            let scalar_type = agreed_type(input_shapes, output_shapes)?;
            match scalar_type {
                ScalarType::I32 | ScalarType::I64 | ScalarType::U8 => {
                    return Err(CompileError::UnimplementedVariant {
                        variant: "with integers".to_string(),
                        op: op.to_string(),
                    })
                }
                ScalarType::F32 => (),
            }

            // Obtain alpha and beta coefficients
            let alpha = node.get_attribute_value("alpha", Some(1.0))?;
            let beta = node.get_attribute_value("beta", Some(1.0))?;
            context.insert("alpha", &alpha);
            context.insert("beta", &beta);

            // Determine and set thread count/workgroup size when stacking (shader y dimension)
            let (y_threads, workgroup_size_y) = workgroup_size(
                stack_count,
                MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                MAX_WORKGROUP_SIZE_Y,
            )?;
            context.insert("workgroup_size_y", &workgroup_size_y);

            if dim_m == 1 {
                let n_elements = output_shapes[0].dim(1);
                let (x_threads, workgroup_size_x) = workgroup_size(
                    n_elements,
                    MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    MAX_WORKGROUP_SIZE_X,
                )?;

                context.insert("workgroup_size_x", &workgroup_size_x);
                NodeTemplate {
                    scalar_type: agreed_type(input_shapes, output_shapes)?,
                    template: "matrix/gemm_1.wgsl",
                    threads: (x_threads as _, y_threads, 1),
                }
            } else {
                // Matrix multiplication is performed (by the gemm.wgsl shader) in blocks of 4x4, except when the output matrix
                // or any of the inputs has a dimension smaller than 4, in which case we can do 3x3 or 2x2
                let kernel_size = gcd(dim_m, gcd(dim_k, dim_n)).clamp(1, 4);
                if kernel_size == 1 {
                    return Err(CompileError::UnimplementedVariant {
                        variant: String::from(
                            "usage with input matrixes whose dimensions are not divisible by 2",
                        ),
                        op: op.to_string(),
                    });
                }

                let n_blocks = ceil(dim_m * dim_n, kernel_size * kernel_size);
                let (x_threads, workgroup_size_x) = workgroup_size(
                    n_blocks,
                    MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    MAX_WORKGROUP_SIZE_X,
                )?;

                context.insert("m_chunks", &(dim_m / kernel_size).max(1));
                context.insert("n_chunks", &(dim_n / kernel_size).max(1));
                context.insert("k_chunks", &(dim_k / kernel_size).max(1));
                context.insert("kernel_size", &kernel_size);
                context.insert("workgroup_size_x", &workgroup_size_x);
                NodeTemplate {
                    scalar_type,
                    template: "matrix/gemm.wgsl",
                    threads: (x_threads as _, y_threads, 1),
                }
            }
        }
        "Resize" => {
            let coordinate_transformation_mode = node.get_attribute_value(
                "coordinate_transformation_mode",
                Some("half_pixel".to_string()),
            )?;
            context.insert(
                "coordinate_transformation_mode",
                &coordinate_transformation_mode,
            );

            match coordinate_transformation_mode.as_str() {
                "half_pixel" => {}
                "pytorch_half_pixel" => {}
                "align_corners" => {}
                "asymmetric" => {}
                "tf_crop_and_resize" => {
                    let roi = node.get_attribute_value::<Vec<i64>>("roi", None)?;
                    let extrapolation_value =
                        node.get_attribute_value("extrapolation_value", Some(0.0))?;
                    context.insert("roi", &roi);
                    context.insert("extrapolation_value", &extrapolation_value);
                }
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: "Resize".to_string(),
                        variant: format!(
                            "coordinate_transformation_mode={}",
                            coordinate_transformation_mode
                        ),
                    })
                }
            }

            let scales = node.get_attribute_value::<Vec<f32>>("scales", Some(vec![]))?;
            let scale_prints = if scales.is_empty() {
                let sizes = node.get_attribute_value::<Vec<i64>>("sizes", Some(vec![]))?;
                sizes
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let tmp = *x as f32 / input_shapes[0].dim(i) as f32;
                        format!("{:.2}", tmp)
                    })
                    .collect::<Vec<_>>()
            } else {
                scales.iter().map(|x| format!("{:.2}", x)).collect()
            };

            let mode = node.get_attribute_value("mode", Some("nearest".to_string()))?;
            context.insert("mode", &mode);
            context.insert("scales", &scale_prints);

            match mode.as_str() {
                "nearest" => {
                    let nearest_mode = node.get_attribute_value(
                        "nearest_mode",
                        Some("round_prefer_floor".to_string()),
                    )?;
                    match nearest_mode.as_str() {
                        "floor" => {}
                        _ => {
                            return Err(CompileError::UnimplementedVariant {
                                op: "Resize".to_string(),
                                variant: format!("nearest_mode={}", nearest_mode),
                            })
                        }
                    }
                }
                "cubic" => {
                    let cubic_coeff_a = node.get_attribute_value("cubic_coeff_a", Some(-0.75))?;
                    context.insert("cubic_coeff_a", &cubic_coeff_a);
                    return Err(CompileError::UnimplementedVariant {
                        op: String::from("Resize"),
                        variant: format!("mode={}", mode),
                    });
                }
                /* "linear" | */
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: String::from("Resize"),
                        variant: format!("mode={}", mode),
                    });
                }
            };

            let exclude_outside = node.get_attribute_value("exclude_outside", Some(0))?;
            context.insert("exclude_outside", &exclude_outside);

            NodeTemplate {
                scalar_type: agreed_type(&input_shapes[0..1], &output_shapes[0..1])?,
                template: "matrix/resize.wgsl",
                threads: (ceil(output_lengths[0], 256) as u32, 1, 1),
            }
        }
        "Sum" => return Err(CompileError::UnimplementedOp(String::from("Sum"))),
        "Split" => {
            let mut axis = node.get_attribute_value("axis", Some(0))?;
            if axis < 0 {
                axis += input_shapes[0].rank() as i64
            }
            context.insert("axis", &axis);

            let split_chunk = input_shapes[0].dim(axis as usize) as usize / output_shapes.len();
            let default_split = (1..=output_shapes.len())
                .map(|x| (x * split_chunk) as _)
                .collect();

            let split = node.get_attribute_value::<Vec<i64>>("split", Some(default_split))?;
            context.insert("split", &split);

            NodeTemplate {
                scalar_type: agreed_type(&input_shapes[0..1], output_shapes)?,
                template: "matrix/split.wgsl",
                threads: (ceil(input_lengths[0], 256) as u32, 1, 1),
            }
        }
        "Pad" => {
            let mode = node.get_attribute_value("mode", Some("constant".to_string()))?;
            match mode.as_str() {
                "constant" => {}
                _ => {
                    return Err(CompileError::UnimplementedVariant {
                        op: String::from("Pad"),
                        variant: format!("mode={}", mode),
                    })
                }
            }

            let pads: Vec<i64> = node.get_attribute_value("pads", None)?;
            if pads.len() != input_shapes[0].rank() * 2 {
                return Err(CompileError::InvalidAttributeValue {
                    attribute: "pads".into(),
                    value: format!("{:?}", pads),
                    opset_version,
                });
            }
            let constant_value = node.get_attribute_value("constant_value", Some(0.0))?;
            context.insert("constant_value", &constant_value);

            #[derive(serde::Serialize)]
            struct PadInfo {
                copy_start: u64,
                end_pad_start: u64,
            }
            let mut pad_info = Vec::with_capacity(input_shapes[0].rank());
            for axis in 0..input_shapes[0].rank() {
                let begin = pads[axis];
                let end = pads[input_shapes[0].rank() + axis];

                pad_info.push(PadInfo {
                    copy_start: begin as _,
                    end_pad_start: input_shapes[0].dim(axis) + begin as u64 - end as u64,
                });
            }
            context.insert("pad_info", &pad_info);

            NodeTemplate {
                scalar_type: agreed_type(&input_shapes[0..1], &output_shapes[0..1])?,
                template: "matrix/pad.wgsl",
                threads: (ceil(output_lengths[0], 256) as u32, 1, 1),
            }
        }
        "Transpose" => {
            let n_dims: i64 = input_shapes[0].rank() as i64;
            let default = (0..n_dims).rev().collect::<Vec<i64>>();
            let perms: Vec<i64> = node.get_attribute_value("perm", Some(default))?;

            // The number of elements in the permutations list must be equal to the output shape rank
            if perms.len() != output_shapes[0].rank() {
                return Err(CompileError::InvalidAttributeValue {
                    attribute: "perm".to_string(),
                    value: format!("{:?}", perms),
                    opset_version,
                });
            }

            let chunks = perms
                .iter()
                .map(|p| {
                    input_shapes[0].dims[((*p as usize) + 1)..]
                        .iter()
                        .product::<u64>()
                })
                .collect::<Vec<_>>();

            context.insert("permuted_chunks", &chunks);

            NodeTemplate {
                scalar_type: agreed_type(input_shapes, output_shapes)?,
                template: "matrix/transpose.wgsl",
                threads: (ceil(output_lengths[0], 256) as _, 1, 1),
            }
        }
        op => return Err(CompileError::UnimplementedOp(op.to_string())),
    };

    // Check if we remain within the limits of the thread count allowed by WebGPU
    if node_template.threads.0 > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("X threads"),
            node_template.threads.0 as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }
    if node_template.threads.1 > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("Y threads"),
            node_template.threads.1 as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }
    if node_template.threads.2 > MAX_COMPUTE_WORKGROUPS_PER_DIMENSION {
        return Err(CompileError::ComputeLimitExceeded(
            String::from("Z threads"),
            node_template.threads.2 as _,
            MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        ));
    }

    // Determine (default) scalar data type to use
    context.insert("scalar_type", node_template.scalar_type.wgsl_type_name());
    context.insert(
        "scalar_type_is_float",
        &node_template.scalar_type.is_float(),
    );
    context.insert("scalar_stride", &node_template.scalar_type.stride());
    context.insert(
        "vec4_stride",
        &(MultiType::Vec(node_template.scalar_type, 4).stride()),
    );
    context.insert(
        "mat4x4_stride",
        &(MultiType::Mat(node_template.scalar_type, 4, 4).stride()),
    );
    context.insert("mat3x3_stride", &(48));

    // Render template
    let shader = get_templates()
        .render(node_template.template, &context)
        .expect("failed to render shader");

    Ok(CompiledNode {
        shader,
        threads: node_template.threads,
    })
}

/// Determines the appropriate number of threads and workgroup size given a number of times the entry point of the shader should be run
fn workgroup_size(
    x: u64,
    max_threads: u32,
    max_workgroup_size: u32,
) -> Result<(u32, u32), CompileError> {
    let max_x = max_threads as u64;

    Ok(if x > max_x {
        let workgroup_size = ceil(x, max_x) as _;
        let threads = ceil(x, workgroup_size as u64) as _;
        log::debug!(
            "number of items ({}) exceeds maximum number of threads ({}); adjusting workgroup size={} and threads={} (this will compute {} items)",
            x,
            max_x,
            workgroup_size,
            threads,
            workgroup_size * threads
        );

        if threads > max_threads {
            return Err(CompileError::ComputeLimitExceeded(
                String::from("threads"),
                threads,
                max_threads,
            ));
        }

        if workgroup_size > max_workgroup_size {
            return Err(CompileError::ComputeLimitExceeded(
                String::from("workgroup size"),
                workgroup_size,
                max_workgroup_size,
            ));
        }

        (threads, workgroup_size)
    } else {
        (x as u32, 1)
    })
}
