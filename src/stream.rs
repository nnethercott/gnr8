use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_tch::PyTensor;
use std::io::{self, Write};
use tch::{self, kind::Kind, Device, IndexOp, Tensor};

use crate::gen::{len, GenerationConfig, PartialGenerate};

pub trait StreamGenerate {
    fn stream_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor>;
}

// impl<T> StreamGenerate for T
// where
//     T: PartialGenerate,
// {
//     /// spawn a printing thread since model (PyObject) isn't thread safe but tch::Tensors are
//     fn stream_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor> {
//         Python::with_gil(|py| {
//             let device = input_ids.device();
//             let mut input_ids = input_ids.shallow_clone().to(device); //&Tensor to Tensor
//             let init_len = len!(input_ids);
//             let mut done;
//
//             let tokenizer = match &gc.tokenizer {
//                 Some(t) => t.as_ref(py),
//                 _ => panic!("can't call `stream_generate` without providing a valid tokenizer"),
//             };
//
//             while len!(input_ids) - init_len <= gc.max_new_tokens as i64 {
//                 (done, input_ids) = match T::partial_generate(model, input_ids, &gc) {
//                     Ok(res) => res,
//                     Err(e) => {
//                         println!("{:?}", e);
//                         panic!();
//                     }
//                 };
//
//                 // inject kwargs to tokenizer.decode
//                 let kwargs = PyDict::new(py);
//                 let _ = kwargs.set_item("skip_special_tokens", true);
//
//                 // decode
//                 let s = tokenizer
//                     .call_method("decode", (PyTensor(input_ids.i(0).copy()),), Some(&kwargs))
//                     .unwrap();
//                 let decoded = s.str().unwrap().to_string_lossy().into_owned();
//
//                 print!("\x1B[2J\x1B[1;1H");
//                 print!("{:?}", decoded);
//                 io::stdout().flush().unwrap();
//
//                 if done {
//                     break;
//                 }
//             }
//             println!("");
//
//             Ok(input_ids)
//         })
//     }
// }
