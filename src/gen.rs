use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_tch::PyTensor;
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
    pub top_k: Option<usize>,
    pub device: Device,
    // do_sample: Option<bool>,
    // eos_token_id: Option<usize>,
}
impl GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            ctx_size: 384,
            temperature: Some(1.0),
            top_k: None,
            device: Device::Cpu,
        }
    }
}

// using a &Tensor since PyTensor impls Deref
pub fn tch_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
    let _device = gc.device;
    let mut input_ids = input_ids.shallow_clone(); //&Tensor to Tensor

    let mut new_tokens = Tensor::empty([1], (Kind::Int64, Device::Cpu));

    while len!(new_tokens) <= gc.max_new_tokens as i64 {
        input_ids = input_ids.i((
            ..,
            i64::max(len!(input_ids) - gc.ctx_size as i64, 0)..len!(input_ids),
        ));

        let mut logits = forward(&model, input_ids.shallow_clone())?;
        logits = logits.i((.., -1, ..));

        let tok = logits.argmax(-1, false);

        new_tokens = Tensor::concat(&[new_tokens, tok.copy()], -1);
        input_ids = Tensor::concat(&[input_ids, tok.unsqueeze(0)], -1);
    }

    //Ok(new_tokens.i(1..)) // first token random from empty
    Ok(input_ids) // first token random from empty
}

//pyfunction wrapper around tch_generate
#[pyfunction]
pub fn generate(model: &PyAny, input_ids: PyTensor) -> PyResult<PyTensor> {
    println!("{}", Cuda::is_available());

    let gc = GenerationConfig::new(10);
    let output_ids = tch_generate(&model, &input_ids, gc).unwrap();
    Ok(PyTensor(output_ids))
}
