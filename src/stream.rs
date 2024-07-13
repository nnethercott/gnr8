use anyhow::Result;
use pyo3::prelude::*;
use tch::{self, kind::Kind, Cuda, Device, IndexOp, Tensor};

use crate::gen::{GenerationConfig, PartialGenerate};

pub trait StreamGenerate {
    fn stream_generate(
        model: &PyAny,
        input_ids: &Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)>;
}

impl<T> StreamGenerate for T
where
    T: PartialGenerate,
{
    fn stream_generate(
        model: &PyAny,
        input_ids: &Tensor,
        gc: &GenerationConfig,
    ) -> Result<(bool, Tensor)> {
        todo!();
    }
}
