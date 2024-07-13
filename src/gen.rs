use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_tch::PyTensor;
use std::convert::TryFrom;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

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
    Beam,
    Random,
}

//PyTensor impls IntoPy needed by PyAny::call
pub fn forward(model: &PyAny, x: Tensor) -> Result<Tensor> {
    tch::no_grad(||{
        let tensor: PyTensor = model.call((PyTensor(x),), None)?.extract()?;
        Ok(tensor.0)
    })
}

#[allow(dead_code)]
pub struct GenerationConfig<'a> {
    pub max_new_tokens: usize,
    pub ctx_size: usize,
    pub temperature: f32,
    pub do_sample: bool,
    pub topk: Option<i64>,
    pub num_beams: Option<usize>,
    pub eos_token_id: i64,
    pub sampling_strategy: Option<SamplingStrategy>, //defaults to multinomial
    pub streamer: Option<Streamer<'a>>,
}

impl<'a> GenerationConfig<'a> {
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
            streamer: None,
        }
    }
}

impl PartialGenerate for MultinomialSampler {
    fn partial_generate(
        model: &PyAny,
        mut input_ids: Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)> {
        tch::no_grad(||{
            // truncate at model context window
            input_ids = input_ids.i((
                ..,
                i64::max(len!(input_ids) - gc.ctx_size as i64, 0)..len!(input_ids),
            ));
            let device = input_ids.device();

            // store on cpu 
            let mut new_tokens = input_ids.copy().to_device(Device::Cpu);
            let mut logits = forward(&model, input_ids).expect(&format!("failed fwding logits")); 
            logits = logits.i((.., -1, ..));

            logits = match gc.topk.as_ref() {
                Some(k) => {
                    let (_, idx) = logits.topk(*k, -1, true, true);

                    let mut mask = logits.zeros_like().to_dtype(Kind::Bool, true, false); // size bsz x n_embd

                    // Create the mask tensor
                    let mask = Tensor::zeros_like(&logits)
                        .to_dtype(Kind::Bool, true, false) 
                        .to_device(logits.device())
                        .scatter_(-1, &idx, &Tensor::ones_like(&idx).to_dtype(Kind::Bool, true, false)); 


                    // approx -float('inf')
                    logits = logits / Tensor::from(gc.temperature);
                    logits.where_self(&mask, &Tensor::from(f64::MIN))
                }
                None => logits / Tensor::from(gc.temperature),
            };

            //sample from multinomial
            let mut tok = logits.softmax(-1, Kind::Float).multinomial(1, false);
            let eos_reached = tok == Tensor::from(gc.eos_token_id).view_(tok.size());

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

// impl Generate for BeamSearchSampler {
//     fn generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
//         todo!();
//     }
// }

impl<'a> GenerationConfig<'a> {
    fn from_config(kwargs: Option<&'a PyDict>) -> Self {
        let streamer = kwargs.as_ref().map(|k| {
            let tokenizer = k.get_item("tokenizer").unwrap();
            Streamer { tokenizer }
        });

        Self {
            max_new_tokens: 384,
            ctx_size: 128,
            temperature: 0.8,
            do_sample: true,
            topk: Some(48),
            num_beams: Some(1),
            eos_token_id: 2,
            sampling_strategy: None,
            streamer,
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
        _ => MultinomialSampler::stream_generate(model, &input_ids, gc),
    };

    Ok(PyTensor(output_ids.unwrap().to_device(Device::Cpu)))
}
