use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_tch::PyTensor;
use std::convert::TryFrom;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

macro_rules! len {
    ($t: expr) => {{
        let shape = $t.size();
        shape[$t.dim() - 1]
    }};
}

//############## NOTES #################
//  - design pattern: make function take an impl trait as arg

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
    pub temperature: f32,
    pub do_sample: bool,
    pub topk: Option<i64>,
    pub num_beams: Option<usize>,
    eos_token_id: Option<usize>,
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
            eos_token_id: None,
        }
    }
}

// using a &Tensor since PyTensor impls Deref
pub fn tch_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
    let device = input_ids.device();

    let mut input_ids = input_ids.shallow_clone().to(device); //&Tensor to Tensor
    let mut new_tokens = Tensor::empty([1, 1], (Kind::Int64, device));

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
                    logits = logits.divide(&Tensor::from(gc.temperature));
                    logits = logits.where_self(&mask, &Tensor::from(f64::MIN));

                    logits.softmax(-1, Kind::Float).multinomial(1, false)
                }
                _ => todo!(),
            };
        } else {
            // beam search
            tok = logits.argmax(-1, false);
        }

        //FIXME: add flag for only new tokens or not
        new_tokens = Tensor::concat(&[new_tokens, tok.copy()], -1);
        input_ids = Tensor::concat(&[input_ids, tok], -1);
    }
    new_tokens = new_tokens.i((.., 1..)); // first token random from empty
    Ok(input_ids) // first token random from empty
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate(model: &PyAny, input_ids: PyTensor, kwargs: Option<&PyDict>) -> PyResult<PyTensor> {
    //println!("{}", Cuda::is_available());

    //TODO: impl from or into for PyDict -> GenerationConfig
    //let gc = GenerationConfig::new(64);
    let gc = GenerationConfig {
        max_new_tokens: 32,
        ctx_size: 384,
        temperature: 1.0,
        do_sample: true,
        topk: Some(10),
        num_beams: Some(1),
        eos_token_id: Some(2),
    };
    let output_ids = tch_generate(&model, &input_ids, gc).unwrap();
    Ok(PyTensor(output_ids))
}
