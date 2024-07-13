use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_tch::PyTensor;
use std::convert::TryFrom;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

use crate::stream::*;

macro_rules! len {
    ($t: expr) => {{
        let shape = $t.size();
        shape[$t.dim() - 1]
    }};
}

pub trait Generate {
    fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor>;
}

pub trait PartialGenerate {
    fn partial_generate(
        model: &PyAny,
        input_ids: Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)>;
}

pub struct BeamSearchSampler;
pub struct MultinomialSampler;

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Beam,
    Random,
}

//PyTensor impls IntoPy needed by PyAny::call
pub fn forward(model: &PyAny, x: Tensor) -> Result<Tensor> {
    let tensor: PyTensor = model.call((PyTensor(x),), None)?.extract()?;
    Ok(tensor.0)
}

#[allow(dead_code)]
#[pyclass]
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub ctx_size: usize,
    pub temperature: f32,
    pub do_sample: bool,
    pub topk: Option<i64>,
    pub num_beams: Option<usize>,
    pub eos_token_id: i64,
    pub sampling_strategy: Option<SamplingStrategy>, //defaults to multinomial
}

impl GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            ctx_size: 384,
            temperature: 1.0,
            topk: None,
            do_sample: false,
            num_beams: Some(1),
            eos_token_id: 2,
            sampling_strategy: None,
        }
    }
}

impl PartialGenerate for MultinomialSampler {
    fn partial_generate(
        model: &PyAny,
        mut input_ids: Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)> {
        // let mut new_tokens = Tensor::empty([1, 1], (Kind::Int64, input_ids.device()));

        // truncate at model context window
        input_ids = input_ids.i((
            ..,
            i64::max(len!(input_ids) - gc.ctx_size as i64, 0)..len!(input_ids),
        ));

        let mut logits = forward(&model, input_ids.shallow_clone())?;
        logits = logits.i((.., -1, ..));

        logits = match gc.topk.as_ref() {
            Some(k) => {
                let (_, idx) = logits.topk(*k, -1, true, true);

                let mut mask = logits.zeros_like().to_dtype(Kind::Bool, true, false); // size bsz x n_embd

                // build bool mask with $index \in topk=1$
                for i in 0..len!(idx) {
                    let _ = mask.index_put_(
                        &[Some(Tensor::from(0)), Some(idx.i((0, i)))],
                        &Tensor::from(true),
                        false,
                    );
                }

                // approx -float('inf')
                logits = logits / Tensor::from(gc.temperature);
                logits.where_self(&mask, &Tensor::from(f64::MIN))
            }
            None => logits / Tensor::from(gc.temperature),
        };

        //sample from multinomial
        let tok = logits.softmax(-1, Kind::Float).multinomial(1, false);
        let eos_reached = tok == Tensor::from(gc.eos_token_id).view_(tok.size());

        // new_tokens = Tensor::concat(&[new_tokens, tok.copy()], -1);
        // new_tokens = new_tokens.i((.., 1..));

        input_ids = Tensor::concat(&[input_ids, tok], -1);

        Ok((eos_reached, input_ids))
    }
}

impl<T> Generate for T
where
    T: PartialGenerate,
{
    fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
        //let mut new_tokens = Tensor::empty([1, 1], (Kind::Int64, device));
        let mut input_ids = input_ids.shallow_clone().to(input_ids.device()); //&Tensor to Tensor
        let init_len = len!(input_ids);
        let mut done;

        while len!(input_ids) - init_len <= gc.max_new_tokens as i64 {
            (done, input_ids) = match T::partial_generate(model, input_ids, &gc) {
                Ok(res) => res,
                _ => panic!(),
            };

            if done {
                break;
            }
        }

        Ok(input_ids)
    }
}

// impl Generate for BeamSearchSampler {
//     fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
//         todo!();
//     }
// }

impl GenerationConfig {
    fn from_config(_kwargs: Option<&PyDict>) -> Self {
        Self {
            max_new_tokens: 32,
            ctx_size: 384,
            temperature: 1.0,
            do_sample: true,
            topk: Some(10),
            num_beams: Some(1),
            eos_token_id: 2,
            sampling_strategy: None,
        }
    }
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate(model: &PyAny, input_ids: PyTensor, kwargs: Option<&PyDict>) -> PyResult<PyTensor> {
    //TODO: build generation config from kwargs -> GenerationConfig::from(thing) thing
    //isinstance(PyDict)

    let mut gc = GenerationConfig::from_config(kwargs);

    let output_ids = match gc.sampling_strategy.take() {
        //Some(SamplingStrategy::Beam) => BeamSearchSampler::generate(model, &input_ids, gc),
        _ => MultinomialSampler::generate(model, &input_ids, gc),
    };

    Ok(PyTensor(output_ids.unwrap()))
}
