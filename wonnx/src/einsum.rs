use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Display,
};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct Subscript(char);

#[derive(Debug, Clone)]
pub enum Subscripts {
    Indexes(Vec<Subscript>),
    Ellipsis {
        start: Vec<Subscript>,
        end: Vec<Subscript>,
    },
}

/// Represents an Einstein summation expression following the notation described [https://onnx.ai/onnx/operators/onnx__Einsum.html](here).
#[derive(Debug, Clone)]
pub struct Einsum {
    inputs: Vec<Subscripts>,
    output: Option<Subscripts>,
}

#[derive(Error, Debug)]
pub enum EinsumError {
    #[error("invalid character encountered: {0}")]
    InvalidCharacter(char),

    #[error("the formula has no inputs")]
    MissingInputs,
}

impl Subscript {
    pub fn from(c: char) -> Subscript {
        assert!(c.is_alphabetic());
        Subscript(c)
    }
}

fn count_indices(inputs: &[Subscripts]) -> BTreeMap<Subscript, u32> {
    let mut count = BTreeMap::new();
    for input in inputs {
        for c in input.subscripts() {
            count.entry(c).and_modify(|n| *n += 1).or_insert(1);
        }
    }
    count
}

impl Subscripts {
    fn push(&mut self, index: Subscript) {
        match self {
            Subscripts::Indexes(idxs) => idxs.push(index),
            Subscripts::Ellipsis { end, .. } => {
                end.push(index);
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Subscripts::Indexes(idx) => idx.is_empty(),
            Subscripts::Ellipsis { start, end } => start.is_empty() && end.is_empty(),
        }
    }

    fn subscripts(&self) -> Vec<Subscript> {
        match &self {
            Subscripts::Indexes(indices) => indices.clone(),
            Subscripts::Ellipsis { start, end } => {
                start.iter().chain(end.iter()).cloned().collect()
            }
        }
    }
}

impl Display for Subscript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Subscripts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subscripts::Indexes(idxs) => {
                for i in idxs {
                    write!(f, "{}", i)?;
                }
                Ok(())
            }
            Subscripts::Ellipsis { start, end } => {
                for i in start {
                    write!(f, "{}", i)?;
                }
                write!(f, "...")?;
                for i in end {
                    write!(f, "{}", i)?;
                }
                Ok(())
            }
        }
    }
}

impl Einsum {
    #[allow(dead_code)]
    pub fn from(str: &str) -> Result<Einsum, EinsumError> {
        let mut sum = Einsum {
            inputs: vec![],
            output: None,
        };

        // Parse up to arrow
        let mut chars = str.chars();
        let mut current_subscripts = Subscripts::Indexes(vec![]);
        let mut after_arrow = false;
        while let Some(character) = &chars.next() {
            match character {
                '-' if chars.next() == Some('>') => {
                    // Arrow: switch from inputs to outputs
                    if !current_subscripts.is_empty() {
                        sum.inputs.push(current_subscripts);
                        current_subscripts = Subscripts::Indexes(vec![]);
                    }
                    if sum.inputs.is_empty() {
                        return Err(EinsumError::MissingInputs);
                    }
                    after_arrow = true;
                }
                '.' if chars.next() == Some('.') && chars.next() == Some('.') => {
                    // Ellipsis
                    current_subscripts = match current_subscripts {
                        Subscripts::Indexes(idxs) => Subscripts::Ellipsis {
                            start: idxs,
                            end: vec![],
                        },
                        Subscripts::Ellipsis { .. } => {
                            return Err(EinsumError::InvalidCharacter('.'))
                        }
                    }
                }
                ' ' => {}
                ',' if !after_arrow => {
                    // Next input (cannot occur in output)
                    sum.inputs.push(current_subscripts);
                    current_subscripts = Subscripts::Indexes(vec![]);
                }
                c if c.is_alphabetic() => {
                    current_subscripts.push(Subscript::from(*c));
                }
                _ => return Err(EinsumError::InvalidCharacter(*character)),
            }
        }

        // If we still have subscripts, they are either the last input or the output
        if !current_subscripts.is_empty()
            || matches!(current_subscripts, Subscripts::Ellipsis { .. }) && after_arrow
        {
            if after_arrow {
                sum.output = Some(current_subscripts);
            } else {
                sum.inputs.push(current_subscripts);
            }
        }

        Ok(sum)
    }

    fn output_or_implicit_subscripts(&self) -> Vec<Subscript> {
        match &self.output {
            Some(o) => o.subscripts(),
            None => {
                // In implicit mode output indices are set to the alphabetically sorted sequence of indices
                // appearing exactly once in the equation.
                let counts = count_indices(&self.inputs);
                let mut keys: Vec<Subscript> = counts
                    .into_iter()
                    .filter_map(|(k, v)| if v == 1 { Some(k) } else { None })
                    .collect();
                keys.sort();
                keys
            }
        }
    }

    fn contraction_indices(&self) -> Vec<Subscript> {
        let count = count_indices(&self.inputs);
        let mut subscripts: BTreeSet<Subscript> = count
            .into_iter()
            .filter_map(|(key, value)| if value > 1 { Some(key) } else { None })
            .collect();
        for c in &self.output_or_implicit_subscripts() {
            subscripts.remove(c);
        }
        subscripts.into_iter().collect()
    }
}

impl Display for Einsum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.inputs
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )?;

        if let Some(output) = &self.output {
            write!(f, " -> {}", output)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{count_indices, Einsum, Subscript};

    pub fn compare_after_reserialize(formula: &str, expected: &str) {
        assert_eq!(Einsum::from(formula).unwrap().to_string(), expected);
    }

    pub fn expect_fail(formula: &str) {
        assert!(Einsum::from(formula).is_err())
    }

    #[test]
    pub fn test_parse_einsum() {
        compare_after_reserialize("ij,jk->ik", "ij,jk -> ik");
        compare_after_reserialize(" i j, j k -> i k", "ij,jk -> ik");
        compare_after_reserialize(" i j-> i k", "ij -> ik");

        compare_after_reserialize("a ...d,x... z->a ...z", "a...d,x...z -> a...z");
        compare_after_reserialize(" ...d,x... z->a ...", "...d,x...z -> a...");
        compare_after_reserialize("a...", "a...");
        compare_after_reserialize("a ...d,x... z->...", "a...d,x...z -> ...");

        expect_fail("ij- >ik");
        expect_fail("->ik");
        expect_fail("a ...d,x... z->a . ..z");
        expect_fail("a...b...c");
        expect_fail("a....b...c");
        expect_fail("a..b...c");
    }

    #[test]
    pub fn test_indices() {
        let es = Einsum::from("ij,jk->ik").unwrap();
        let out = count_indices(&es.inputs);
        assert_eq!(out.len(), 3);
        assert_eq!(out[&Subscript::from('i')], 1);
        assert_eq!(out[&Subscript::from('j')], 2);
        assert_eq!(out[&Subscript::from('k')], 1);

        let es = Einsum::from("i...k,k...m->i...m").unwrap();
        let out = count_indices(&es.inputs);
        println!("{:?}", out);
        assert_eq!(out.len(), 5);
        assert_eq!(out[&Subscript::from('i')], 1);
        assert_eq!(out[&Subscript::from('j')], 1);
        assert_eq!(out[&Subscript::from('k')], 2);
        assert_eq!(out[&Subscript::from('l')], 1);
        assert_eq!(out[&Subscript::from('m')], 1);
    }

    #[test]
    pub fn test_analysis() {
        let es = Einsum::from("ij,jk->ik").unwrap();
        assert_eq!(es.contraction_indices(), vec![Subscript::from('j')]);

        let es = Einsum::from("ij,jk").unwrap();
        assert_eq!(
            es.output_or_implicit_subscripts(),
            vec![Subscript::from('i'), Subscript::from('k')]
        );
        assert_eq!(es.contraction_indices(), vec![Subscript::from('j')]);

        let transpose = Einsum::from("ba").unwrap();
        assert_eq!(
            transpose.output_or_implicit_subscripts(),
            vec![Subscript::from('a'), Subscript::from('b')]
        );
    }
}
