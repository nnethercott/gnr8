use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_tch::PyTensor;
use tch::{self, kind::Kind, Device, IndexOp, Tensor};

use crate::stream::*;

#[macro_export]
macro_rules! len {
    ($t: expr) => {{
        let shape = $t.size();
        shape[$t.dim() - 1]
    }};
}
pub use len;

pub trait PartialGenerate {
    fn partial_generate(logits: Tensor, gc: &GenerationConfig) -> Result<(bool, Tensor)>;
}

pub struct BeamSearchSampler;
pub struct MultinomialSampler;

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Beam(usize),
    Random,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    #[pyo3(get, set)]
    pub max_new_tokens: usize,
    #[pyo3(get, set)]
    pub ctx_size: usize,
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get, set)]
    pub do_sample: bool,
    #[pyo3(get, set)]
    pub topk: Option<i64>,
    #[pyo3(get, set)]
    pub num_beams: usize,
    #[pyo3(get, set)]
    pub eos_token_id: i64,
    pub sampling_strategy: SamplingStrategy, //don't expose
    #[pyo3(get, set)]
    pub stream: bool,
    #[pyo3(get, set)]
    pub tokenizer: Option<PyObject>,
}

#[pymethods]
impl GenerationConfig {
    #[new]
    pub fn new(
        max_new_tokens: usize,
        ctx_size: Option<usize>,
        temperature: Option<f32>,
        do_sample: Option<bool>,
        topk: Option<i64>,
        num_beams: Option<usize>,
        eos_token_id: Option<i64>,
        stream: Option<bool>,
        tokenizer: Option<PyObject>,
    ) -> Self {
        let do_sample = do_sample.unwrap_or(true);
        let num_beams = num_beams.unwrap_or(1);
        let sampling_strategy = if do_sample {
            SamplingStrategy::Random
        } else {
            SamplingStrategy::Beam(num_beams)
        };

        Self {
            max_new_tokens,
            ctx_size: ctx_size.unwrap_or(384),
            temperature: temperature.unwrap_or(1.0),
            topk,
            do_sample,
            num_beams,
            eos_token_id: eos_token_id.unwrap_or(-1), // no early stop
            sampling_strategy,
            stream: stream.unwrap_or(false),
            tokenizer,
        }
    }
}

impl PartialGenerate for MultinomialSampler {
    fn partial_generate(mut logits: Tensor, gc: &GenerationConfig) -> Result<(bool, Tensor)> {
        logits = logits.i((.., -1, ..));

        logits = match gc.topk.as_ref() {
            Some(k) => {
                let (_, idx) = logits.topk(*k, -1, true, true);

                // Create the mask tensor
                let mask = Tensor::zeros_like(&logits)
                    .to_dtype(Kind::Bool, true, false)
                    .to_device(logits.device())
                    .scatter_(
                        -1,
                        &idx,
                        &Tensor::ones_like(&idx).to_dtype(Kind::Bool, true, false),
                    );

                // approx -float('inf')
                logits = logits / Tensor::from(gc.temperature);
                logits.where_self(&mask, &Tensor::from(f64::MIN))
            }
            None => logits / Tensor::from(gc.temperature),
        };

        //sample from multinomial
        let mut tok = logits.softmax(-1, Kind::Float).multinomial(1, false);
        let eos_reached = tok
            == Tensor::from(gc.eos_token_id)
                .view_(tok.size())
                .to_device(tok.device());

        drop(logits);

        Ok((eos_reached, tok))
    }
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate_token(mut logits: PyTensor, gc: &GenerationConfig) -> PyResult<(bool, PyTensor)> {
    //bleh
    let (done, next_token) = match gc.sampling_strategy {
        SamplingStrategy::Beam(_) => {
            //BeamSearchSampler::generate(model, &input_ids, gc), //FIXME: add logic for n==1 and n>1
            todo!()
        }
        SamplingStrategy::Random => {
            // PyTensor -> Tensor
            let logits = logits.0;
            MultinomialSampler::partial_generate(logits, gc)
                .unwrap_or((true, Tensor::from(gc.eos_token_id)))
        }
    };
    // dummy
    Ok((done, PyTensor(next_token)))
}
