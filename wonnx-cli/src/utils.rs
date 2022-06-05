use ndarray::{Array, ArrayBase};
use std::collections::HashMap;
use std::path::Path;
use wonnx::onnx::{ModelProto, TensorShapeProto, ValueInfoProto};
use wonnx::utils::{DataTypeError, ScalarType, Shape};
use wonnx::WonnxError;
use wonnx_preprocessing::image::{load_bw_image, load_rgb_image};
use wonnx_preprocessing::text::{EncodedText, TextTokenizer};
use wonnx_preprocessing::Tensor;

use crate::types::{InferOptions, InferenceInput, NNXError};
pub trait ValueInfoProtoUtil {
    fn dimensions(&self) -> Vec<usize>;
    fn data_type(&self) -> Result<ScalarType, DataTypeError>;
}

pub trait TensorShapeProtoUtil {
    fn shape_dimensions(&self) -> Vec<usize>;
}

impl ValueInfoProtoUtil for ValueInfoProto {
    fn dimensions(&self) -> Vec<usize> {
        match &self.get_field_type().value {
            Some(x) => match x {
                wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => {
                    t.get_shape().shape_dimensions()
                }
                wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => vec![],
        }
    }

    fn data_type(&self) -> Result<ScalarType, DataTypeError> {
        Ok(match &self.get_field_type().value {
            Some(x) => match x {
                wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => {
                    ScalarType::from_i32(t.get_elem_type())?
                }
                wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
                wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
            },
            None => return Err(DataTypeError::Undefined),
        })
    }
}

impl TensorShapeProtoUtil for TensorShapeProto {
    fn shape_dimensions(&self) -> Vec<usize> {
        self.get_dim()
            .iter()
            .map(|d| match d.value {
                Some(wonnx::onnx::TensorShapeProto_Dimension_oneof_value::dim_value(i)) => {
                    i as usize
                }
                _ => 0,
            })
            .collect()
    }
}

pub trait ModelProtoUtil {
    fn get_input_shape(&self, input_name: &str) -> Result<Option<Shape>, WonnxError>;
}

impl ModelProtoUtil for ModelProto {
    fn get_input_shape(&self, input_name: &str) -> Result<Option<Shape>, WonnxError> {
        let value_info = self
            .get_graph()
            .get_input()
            .iter()
            .find(|x| x.get_name() == input_name);
        match value_info {
            Some(vi) => Ok(Some(vi.get_shape()?)),
            None => Ok(None),
        }
    }
}

pub fn load_image_input(
    input_image: &Path,
    input_shape: &Shape,
) -> Result<ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>, NNXError> {
    if input_shape.rank() == 3 {
        let mut w = input_shape.dim(1) as usize;
        let mut h = input_shape.dim(2) as usize;
        if w == 0 {
            w = 224;
        }
        if h == 0 {
            h = 224;
        }

        if input_shape.dim(0) == 3 {
            log::info!("input is (3,?,?), loading as RGB image");
            Ok(load_rgb_image(input_image, w, h).into_dyn())
        } else if input_shape.dim(0) == 1 {
            log::info!("input is (1,?,?), loading as BW image");
            Ok(load_bw_image(input_image, w, h).into_dyn())
        } else {
            Err(NNXError::InvalidInputShape)
        }
    } else if input_shape.rank() == 4 {
        let mut w = input_shape.dim(2) as usize;
        let mut h = input_shape.dim(3) as usize;
        if w == 0 {
            w = 224;
        }
        if h == 0 {
            h = 224;
        }

        if input_shape.dim(1) == 3 {
            log::info!("input is (?,3,?,?), loading as RGB image");
            Ok(load_rgb_image(input_image, w, h).into_dyn())
        } else if input_shape.dim(1) == 1 {
            log::info!("input is (?,1,?,?), loading as BW image");
            Ok(load_bw_image(input_image, w, h).into_dyn())
        } else {
            Err(NNXError::InvalidInputShape)
        }
    } else {
        Err(NNXError::InvalidInputShape)
    }
}

impl InferenceInput {
    pub fn new(infer_opt: &InferOptions, model: &ModelProto) -> Result<InferenceInput, NNXError> {
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        let mut input_shapes = HashMap::with_capacity(inputs.len());

        // Do we have question and context?
        let mut qa_encoding: Option<EncodedText> = None;
        if let (Some(question), Some(context)) = (&infer_opt.question, &infer_opt.context) {
            let tokens_input_shape = model
                .get_input_shape(&infer_opt.qa_tokens_input)?
                .ok_or_else(|| NNXError::InputNotFound(infer_opt.qa_tokens_input.clone()))?;
            let mask_input_shape = model
                .get_input_shape(&infer_opt.qa_mask_input)?
                .ok_or_else(|| NNXError::InputNotFound(infer_opt.qa_mask_input.clone()))?;
            let segment_input_shape = model
                .get_input_shape(&infer_opt.qa_segment_input)?
                .ok_or_else(|| NNXError::InputNotFound(infer_opt.qa_segment_input.clone()))?;

            let segment_length = tokens_input_shape.element_count() as usize;

            if segment_length != mask_input_shape.element_count() as usize {
                return Err(NNXError::InvalidInputShape);
            }
            if segment_length != segment_input_shape.element_count() as usize {
                return Err(NNXError::InvalidInputShape);
            }

            log::info!(
                "QA: writing question '{}', context '{}' to {}/{}/{} (segment length: {})",
                question,
                context,
                infer_opt.qa_tokens_input,
                infer_opt.qa_mask_input,
                infer_opt.qa_segment_input,
                segment_length
            );

            let bert_tokenizer = TextTokenizer::from_config(&infer_opt.tokenizer)?;
            let mut encoding = bert_tokenizer.tokenize_question_answer(question, context)?;

            let first_encoding = encoding.remove(0);

            let mut tokens_input = first_encoding.get_tokens();
            let mut mask_input = first_encoding.get_mask();
            let mut segment_input = first_encoding.get_segments();
            log::debug!(
                "tokens={:?} mask={:?} segments={:?}",
                tokens_input,
                mask_input,
                segment_input
            );

            tokens_input.resize(segment_length, 0);
            mask_input.resize(segment_length, 0);
            segment_input.resize(segment_length, 0);
            let tokens_input_data =
                ndarray::Array::from_iter(tokens_input.iter().map(|x| (*x) as i64)).into_dyn();
            let mask_input_data =
                ndarray::Array::from_iter(mask_input.iter().map(|x| (*x) as i64)).into_dyn();
            let segment_input_data =
                ndarray::Array::from_iter(segment_input.iter().map(|x| (*x) as i64)).into_dyn();
            inputs.insert(
                infer_opt.qa_tokens_input.clone(),
                Tensor::I64(tokens_input_data),
            );
            input_shapes.insert(infer_opt.qa_tokens_input.clone(), tokens_input_shape);
            inputs.insert(
                infer_opt.qa_mask_input.clone(),
                Tensor::I64(mask_input_data),
            );
            input_shapes.insert(infer_opt.qa_mask_input.clone(), mask_input_shape);
            inputs.insert(
                infer_opt.qa_segment_input.clone(),
                Tensor::I64(segment_input_data),
            );
            input_shapes.insert(infer_opt.qa_segment_input.clone(), segment_input_shape);
            qa_encoding = Some(first_encoding);
        }

        // Process text inputs
        if !infer_opt.text.is_empty() || !infer_opt.text_mask.is_empty() {
            let bert_tokenizer = TextTokenizer::from_config(&infer_opt.tokenizer)?;

            // Tokenized text input
            for (text_input_name, text) in &infer_opt.text {
                let text_input_shape = model
                    .get_input_shape(text_input_name)?
                    .ok_or_else(|| NNXError::InputNotFound(text_input_name.clone()))?;
                let input = bert_tokenizer.get_input_for(text, &text_input_shape)?;
                inputs.insert(text_input_name.clone(), input);
                input_shapes.insert(text_input_name.clone(), text_input_shape);
            }

            // Tokenized text input: mask
            for (text_input_name, text) in &infer_opt.text_mask {
                let text_input_shape = model
                    .get_input_shape(text_input_name)?
                    .ok_or_else(|| NNXError::InputNotFound(text_input_name.clone()))?;
                let input = bert_tokenizer.get_mask_input_for(text, &text_input_shape)?;
                inputs.insert(text_input_name.clone(), input);
                input_shapes.insert(text_input_name.clone(), text_input_shape);
            }
        }

        // Process raw inputs
        for (raw_input_name, text) in &infer_opt.raw {
            let raw_input_shape = model
                .get_input_shape(raw_input_name)?
                .ok_or_else(|| NNXError::InputNotFound(raw_input_name.clone()))?;

            let values: Result<Vec<f32>, _> = text.split(',').map(|v| v.parse::<f32>()).collect();
            let mut values = values.map_err(NNXError::InvalidNumber)?;
            values.resize(raw_input_shape.element_count() as usize, 0.0);
            inputs.insert(
                raw_input_name.clone(),
                Tensor::F32(Array::from_vec(values).into_dyn()),
            );
            input_shapes.insert(raw_input_name.clone(), raw_input_shape);
        }

        // Load input image if it was supplied
        for (input_name, image_path) in &infer_opt.input_images {
            let mut input_shape = model
                .get_input_shape(input_name)?
                .ok_or_else(|| NNXError::InputNotFound(input_name.clone()))?;

            let data = load_image_input(image_path, &input_shape)?;

            // Some models allow us to set the number of items we are throwing at them.
            if input_shape.dim(0) == 0 {
                input_shape.dims[0] = 1;
                log::info!(
                    "changing first dimension for input {} to {:?}",
                    input_name,
                    input_shape
                );
            }

            inputs.insert(input_name.clone(), Tensor::F32(data));
            input_shapes.insert(input_name.clone(), input_shape.clone());
        }

        Ok(InferenceInput {
            input_shapes,
            inputs,
            qa_encoding,
        })
    }
}
