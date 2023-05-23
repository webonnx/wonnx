//! Various utilities to deal with the ONNX format structure
use std::convert::Into;
use thiserror::Error;
use wonnx::onnx;

#[derive(Error, Debug)]
#[error("did not find attribute '{attribute}' for node '{node_name}'")]
pub struct AttributeNotFoundError {
    attribute: String,
    node_name: String,
}

pub trait NodeAttributes {
    fn has_attribute(&self, attribute_name: &str) -> bool;
    fn get_attribute_value<'a, T: std::convert::From<&'a onnx::AttributeProto>>(
        &'a self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError>;
}

impl NodeAttributes for onnx::NodeProto {
    fn has_attribute(&self, attribute_name: &str) -> bool {
        self.get_attribute()
            .iter()
            .any(|attr| attr.get_name() == attribute_name)
    }

    fn get_attribute_value<'a, T>(
        &'a self,
        attribute: &str,
        default: Option<T>,
    ) -> Result<T, AttributeNotFoundError>
    where
        T: From<&'a onnx::AttributeProto>,
    {
        match (
            self.get_attribute()
                .iter()
                .find(|attr| attr.get_name() == attribute),
            default,
        ) {
            (Some(attr), _) => Ok(attr.into()),
            (None, Some(default_attr)) => Ok(default_attr),
            (None, None) => Err(AttributeNotFoundError {
                attribute: attribute.to_string(),
                node_name: self.get_name().to_string(),
            }),
        }
    }
}
