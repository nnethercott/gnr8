use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_tch::{wrap_tch_err, PyTensor, };

use tch::{self, IndexOp, Tensor, Device};

pub mod gen;
pub mod stream;

use crate::gen::*;

#[pymodule]
#[pyo3(name = "gnr8")]
fn gnr8(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // py.import("torch")?;
    m.add_function(wrap_pyfunction!(gen::generate, m)?)?;
    Ok(())
}
