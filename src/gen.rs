use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_tch::PyTensor;
use std::convert::TryFrom;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

macro_rules! len {
    ($t: expr) => {{
        let shape = $t.size();
        let len_last_dim = shape[$t.dim() - 1];
        len_last_dim
    }};
}

//############## NOTES #################
//  - design pattern: make function take an impl trait as arg
//  - TODO: add cuda support, define global device var `torch.cuda.is_available`
//  -

//note: PyTensor impls IntoPy needed by PyAny::call -> need this since model expects `torch.Tensor`
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
    pub temperature: Option<f32>,
    pub do_sample: bool,
    pub topk: Option<i64>,
    pub num_beams: Option<usize>,
    // do_sample: Option<bool>,
    // eos_token_id: Option<usize>,
}
impl GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            ctx_size: 384,
            temperature: Some(1.0),
            topk: None,
            do_sample: false,
            num_beams: Some(1),
        }
    }
}

// using a &Tensor since PyTensor impls Deref
pub fn tch_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
    let device = input_ids.device();

    let mut input_ids = input_ids.shallow_clone().to(device); //&Tensor to Tensor
    let mut new_tokens = Tensor::empty([1], (Kind::Int64, device));

    while len!(new_tokens) <= gc.max_new_tokens as i64 {
        // truncate at model context window
        input_ids = input_ids.i((
            ..,
            i64::max(len!(input_ids) - gc.ctx_size as i64, 0)..len!(input_ids),
        ));

        let mut logits = forward(&model, input_ids.shallow_clone())?;
        logits = logits.i((.., -1, ..));

        // apply sampling strategy
        let tok: Tensor;
        if gc.do_sample {
            tok = match gc.topk.as_ref() {
                Some(k) => {
                    let (_, i) = logits.topk(*k, -1, true, true);
                    //FIXME: add where_self and index_put_ to build mask and hard set logits (see
                    //chatgpt)

                    Tensor::new()
                }
                _ => todo!(),
            };
        } else {
            // beam search
            tok = logits.argmax(-1, false);
        }

        new_tokens = Tensor::concat(&[new_tokens, tok.copy()], -1);
        input_ids = Tensor::concat(&[input_ids, tok.unsqueeze(0)], -1);
    }

    Ok(new_tokens.i(1..)) // first token random from empty
                          //Ok(input_ids) // first token random from empty
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate(model: &PyAny, input_ids: PyTensor, kwargs: Option<&PyDict>) -> PyResult<PyTensor> {
    //println!("{}", Cuda::is_available());

    //TODO: impl from or into for PyDict -> GenerationConfig
    let gc = GenerationConfig::new(64);
    let output_ids = tch_generate(&model, &input_ids, gc).unwrap();
    Ok(PyTensor(output_ids))
}
