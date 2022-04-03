use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;
use tokenizers::models::wordpiece::WordPieceBuilder;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::utils::padding::{PaddingDirection::Right, PaddingParams, PaddingStrategy::Fixed};
use tokenizers::utils::truncation::TruncationParams;
use tokenizers::utils::truncation::TruncationStrategy::LongestFirst;
use tokenizers::{AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer};
use wonnx::utils::Shape;

use crate::Tensor;

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("text tokenization error: {0}")]
    TextTokenizationError(#[from] Box<dyn std::error::Error + Sync + Send>),
}

pub struct BertTokenizer {
    pub tokenizer: Tokenizer,
}

pub struct BertEncodedText {
    pub encoding: Encoding,
}

impl BertTokenizer {
    pub fn new(vocab_path: &Path) -> BertTokenizer {
        let wp_builder = WordPieceBuilder::new()
            .files(vocab_path.as_os_str().to_string_lossy().to_string())
            .continuing_subword_prefix("##".into())
            .max_input_chars_per_word(100)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(wp_builder);
        tokenizer.with_padding(Some(PaddingParams {
            strategy: Fixed(60),
            direction: Right,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
            pad_to_multiple_of: None,
        }));
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: 60,
            strategy: LongestFirst,
            stride: 0,
            ..Default::default()
        }));
        tokenizer.with_pre_tokenizer(BertPreTokenizer);
        tokenizer.with_post_processor(BertProcessing::new(
            ("[SEP]".into(), 102),
            ("[CLS]".into(), 101),
        ));
        tokenizer.with_normalizer(BertNormalizer::new(true, true, Some(false), false));
        tokenizer.add_special_tokens(&[
            AddedToken {
                content: "[PAD]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false, //?
                ..Default::default()
            },
            AddedToken {
                content: "[CLS]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false, //?
                ..Default::default()
            },
            AddedToken {
                content: "[SEP]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false, //?
                ..Default::default()
            },
            AddedToken {
                content: "[MASK]".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false, //?
                ..Default::default()
            },
        ]);

        BertTokenizer { tokenizer }
    }

    pub fn tokenize_question_answer(
        &self,
        question: &str,
        context: &str,
    ) -> Result<BertEncodedText, PreprocessingError> {
        Ok(BertEncodedText {
            encoding: self
                .tokenizer
                .encode(
                    EncodeInput::Dual(
                        InputSequence::Raw(Cow::from(question)),
                        InputSequence::Raw(Cow::from(context)),
                    ),
                    true,
                )
                .map_err(PreprocessingError::TextTokenizationError)?,
        })
    }

    fn tokenize(&self, text: &str) -> Result<BertEncodedText, PreprocessingError> {
        let encoding = self
            .tokenizer
            .encode(
                EncodeInput::Single(InputSequence::Raw(Cow::from(text))),
                true,
            )
            .map_err(PreprocessingError::TextTokenizationError)?;
        Ok(BertEncodedText { encoding })
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

impl BertEncodedText {
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
        log::debug!("segment_ids={:?}", self.encoding.get_sequence_ids());
        self.encoding
            .get_sequence_ids()
            .iter()
            .map(|x| x.unwrap_or(0) as i64)
            .collect()
    }
}

pub fn get_lines(path: &Path) -> Vec<String> {
    let file = BufReader::new(File::open(path).unwrap());
    file.lines().map(|line| line.unwrap()).collect()
}
