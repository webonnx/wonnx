use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;
use tokenizers::{EncodeInput, Encoding, InputSequence, Tokenizer};
use wonnx::utils::Shape;

use crate::Tensor;

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("text tokenization error: {0}")]
    TextTokenizationError(#[from] Box<dyn std::error::Error + Sync + Send>),
}

pub struct TextTokenizer {
    pub tokenizer: Tokenizer,
}

#[derive(Debug)]
pub struct EncodedText {
    pub encoding: Encoding,
}

impl TextTokenizer {
    pub fn new(tokenizer: Tokenizer) -> TextTokenizer {
        TextTokenizer { tokenizer }
    }

    pub fn from_config<P: AsRef<Path>>(path: P) -> Result<TextTokenizer, std::io::Error> {
        let tokenizer_config_file = File::open(path)?;
        let tokenizer_config_reader = BufReader::new(tokenizer_config_file);
        let tokenizer = serde_json::from_reader(tokenizer_config_reader)?;
        Ok(TextTokenizer::new(tokenizer))
    }

    pub fn tokenize_question_answer(
        &self,
        question: &str,
        context: &str,
    ) -> Result<Vec<EncodedText>, PreprocessingError> {
        let mut encoding = self
            .tokenizer
            .encode(
                EncodeInput::Dual(
                    InputSequence::Raw(Cow::from(question)),
                    InputSequence::Raw(Cow::from(context)),
                ),
                true,
            )
            .map_err(PreprocessingError::TextTokenizationError)?;

        let mut overflowing = encoding.take_overflowing();
        overflowing.insert(0, encoding);

        Ok(overflowing
            .into_iter()
            .map(|x| EncodedText { encoding: x })
            .collect())
    }

    fn tokenize(&self, text: &str) -> Result<EncodedText, PreprocessingError> {
        let encoding = self
            .tokenizer
            .encode(
                EncodeInput::Single(InputSequence::Raw(Cow::from(text))),
                true,
            )
            .map_err(PreprocessingError::TextTokenizationError)?;
        Ok(EncodedText { encoding })
    }

    pub fn decode(&self, encoding: &EncodedText) -> Result<String, PreprocessingError> {
        let ids: Vec<u32> = encoding.get_tokens().iter().map(|x| *x as u32).collect();
        self.tokenizer
            .decode(&ids, true)
            .map_err(PreprocessingError::TextTokenizationError)
    }

    pub fn get_mask_input_for(
        &self,
        text: &str,
        shape: &Shape,
    ) -> Result<Tensor, PreprocessingError> {
        let segment_length = shape.dim(shape.rank() - 1) as usize;
        let tokenized = self.tokenize(text)?;
        let mut tokens = tokenized.get_mask();
        tokens.resize(segment_length, 0);
        let data = ndarray::Array::from_iter(tokens.iter().map(|x| (*x) as f32)).into_dyn();
        Ok(Tensor::F32(data))
    }

    pub fn get_input_for(&self, text: &str, shape: &Shape) -> Result<Tensor, PreprocessingError> {
        let segment_length = shape.dim(shape.rank() - 1) as usize;
        let tokenized = self.tokenize(text)?;
        let mut tokens = tokenized.get_tokens();
        tokens.resize(segment_length, 0);
        let data = ndarray::Array::from_iter(tokens.iter().map(|x| (*x) as f32)).into_dyn();
        Ok(Tensor::F32(data))
    }
}

#[derive(Debug)]
pub struct Answer {
    pub text: String,
    pub tokens: Vec<String>,
    pub score: f32,
}

impl EncodedText {
    pub fn get_mask(&self) -> Vec<i64> {
        self.encoding
            .get_attention_mask()
            .iter()
            .map(|x| *x as i64)
            .collect()
    }

    pub fn get_tokens(&self) -> Vec<i64> {
        self.encoding.get_ids().iter().map(|x| *x as i64).collect()
    }

    pub fn get_segments(&self) -> Vec<i64> {
        self.encoding
            .get_type_ids()
            .iter()
            .map(|x| *x as i64)
            .collect()
    }

    pub fn get_answer(&self, start_output: &[f32], end_output: &[f32], context: &str) -> Answer {
        let mut best_start_logit = f32::MIN;
        let mut best_start_idx: usize = 0;

        let input_tokens = self.encoding.get_tokens();
        let special_tokens_mask = self.encoding.get_special_tokens_mask();

        for (start_idx, start_logit) in start_output.iter().enumerate() {
            if start_idx > input_tokens.len() - 1 {
                break;
            }

            // Skip special tokens such as [CLS], [SEP], [PAD]
            if special_tokens_mask[start_idx] == 1 {
                continue;
            }

            if *start_logit > best_start_logit {
                best_start_logit = *start_logit;
                best_start_idx = start_idx;
            }
        }

        // Find matching end
        let mut best_end_logit = f32::MIN;
        let mut best_end_idx = best_start_idx;
        for (end_idx, end_logit) in end_output[best_start_idx..].iter().enumerate() {
            if (end_idx + best_start_idx) > input_tokens.len() - 1 {
                break;
            }

            // Skip special tokens such as [CLS], [SEP], [PAD]
            if special_tokens_mask[end_idx + best_start_idx] == 1 {
                continue;
            }

            if *end_logit > best_end_logit {
                best_end_logit = *end_logit;
                best_end_idx = end_idx + best_start_idx;
            }
        }

        log::debug!("start index: {} ({})", best_start_idx, best_start_logit);
        log::debug!("end index: {} ({})", best_end_idx, best_end_logit);

        let chars: Vec<char> = context.chars().collect();
        let offsets = self.encoding.get_offsets();
        log::debug!("offsets: {:?}", &offsets[best_start_idx..=best_end_idx]);

        let answer_tokens: Vec<String> =
            self.encoding.get_tokens()[best_start_idx..best_end_idx].to_vec();

        let min_offset = offsets[best_start_idx..=best_end_idx]
            .iter()
            .map(|o| o.0)
            .min()
            .unwrap();
        let max_offset = offsets[best_start_idx..=best_end_idx]
            .iter()
            .map(|o| o.1)
            .max()
            .unwrap();
        assert!(min_offset <= max_offset);
        if max_offset > chars.len() - 1 {
            return Answer {
                text: "".to_string(),
                tokens: vec![],
                score: 0.0,
            };
        }

        let answer = chars[min_offset..max_offset].iter().collect::<String>();

        Answer {
            text: answer,
            tokens: answer_tokens,
            score: best_start_logit * best_end_logit,
        }
    }
}

pub fn get_lines(path: &Path) -> Vec<String> {
    let file = BufReader::new(File::open(path).unwrap());
    file.lines().map(|line| line.unwrap()).collect()
}
