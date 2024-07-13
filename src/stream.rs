use anyhow::Result;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

use crate::gen::{len, GenerationConfig, PartialGenerate};

pub struct Streamer<'a> {
    pub tokenizer: &'a PyAny,
}

pub trait StreamGenerate {
    fn stream_generate(model: &PyAny, input_ids: &Tensor, gc: GenerationConfig) -> Result<Tensor>;
}

impl<T> StreamGenerate for T
where
    T: PartialGenerate,
{
    /// spawn a printing thread since model (PyObject) isn't thread safe but tch::Tensors are
    fn stream_generate(
        model: &PyAny,
        input_ids: &Tensor,
        mut gc: GenerationConfig,
    ) -> Result<Tensor> {
        let device = input_ids.device();
        let mut input_ids = input_ids.shallow_clone().to(device); //&Tensor to Tensor
        let init_len = len!(input_ids);
        let mut done;

        // init channel
        let (tx, rx) = mpsc::channel::<Tensor>();

        let tokenizer: PyObject = gc.streamer.take().unwrap().tokenizer.into();
        let handle = thread::spawn(move || {
            let mut generated = Tensor::empty([1], (Kind::Int64, Device::Cpu));

            while let Ok(token) = rx.recv() {
                generated = Tensor::cat(&[generated, token.to_device(Device::Cpu)], -1);

                //decode
                let decoded = Python::with_gil(|py| {
                    // "borrow a GIL-bound reference to the contained object."
                    let tokenizer = tokenizer.as_ref(py);
                    let s = tokenizer
                        .call_method("decode", (PyTensor(generated.i(1..).copy()),), None)
                        .unwrap();
                    s.str().unwrap().to_string_lossy().into_owned();
                });

                print!("\r{:?}", decoded);
                io::stdout().flush().unwrap();
            }
            println!("");
        });

        while len!(input_ids) - init_len <= gc.max_new_tokens as i64 {
            (done, input_ids) = match T::partial_generate(model, input_ids, &gc) {
                Ok(res) => res,
                _ => panic!(),
            };

            let _ = tx.send(input_ids.i((.., -1)).shallow_clone());

            if done {
                break;
            }
        }
        drop(tx);

        let _ = handle.join().unwrap();

        Ok(input_ids)
    }
}
