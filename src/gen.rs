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
    Beam(usize),
    Random,
}

//PyTensor impls IntoPy needed by PyAny::call
pub fn forward(model: &PyAny, x: Tensor) -> Result<Tensor> {
    tch::no_grad(|| {
        let tensor: PyTensor = model.call((PyTensor(x),), None)?.extract()?;
        Ok(tensor.0)
    })
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
            tokenizer,
        }
    }
}

impl PartialGenerate for MultinomialSampler {
    fn partial_generate(
        model: &PyAny,
        mut input_ids: Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)> {
        tch::no_grad(|| {
            // truncate at model context window
            input_ids = input_ids.i((
                ..,
                i64::max(len!(input_ids) - gc.ctx_size as i64, 0)..len!(input_ids),
            ));
            let device = input_ids.device();

            // store on cpu
            let mut new_tokens = input_ids.copy().to_device(Device::Cpu);
            let mut logits =
                forward(&model, input_ids).expect(&format!("failed forwarding logits"));
            logits = logits.i((.., -1, ..));

            logits = match gc.topk.as_ref() {
                Some(k) => {
                    let (_, idx) = logits.topk(*k, -1, true, true);

                    let mut mask = logits.zeros_like().to_dtype(Kind::Bool, true, false); // size bsz x n_embd

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

            // new_tokens = Tensor::concat(&[new_tokens, tok.copy()], -1);
            // new_tokens = new_tokens.i((.., 1..));

            new_tokens = Tensor::concat(&[new_tokens.to(device), tok], -1);

            // manually drop tensors
            drop(logits);

            Ok((eos_reached, new_tokens))
        })
    }
}

impl<T> Generate for T
where
    T: PartialGenerate,
{
    fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
        //let mut new_tokens = Tensor::empty([1, 1], (Kind::Int64, device));
        let mut input_ids = input_ids.copy().to(input_ids.device()); //&Tensor to Tensor
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

        Ok(input_ids.to_device(Device::Cpu))
    }
}

impl Generate for BeamSearchSampler {
    fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
        todo!();
    }
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate(model: &PyAny, input_ids: PyTensor, gc: GenerationConfig) -> PyResult<PyTensor> {
    // let mut gc = GenerationConfig::from_config(kwargs);

    let output_ids = match gc.sampling_strategy {
        SamplingStrategy::Beam(_) => BeamSearchSampler::generate(model, &input_ids, gc), //FIXME: add logic for n==1 and n>1
        _ => MultinomialSampler::stream_generate(model, &input_ids, gc),
    };

    Ok(PyTensor(output_ids.unwrap().to_device(Device::Cpu)))
}
