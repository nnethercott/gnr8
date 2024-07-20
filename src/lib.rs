use pyo3::prelude::*;

pub mod gen;
pub mod stream;

use crate::gen::*;

#[pymodule]
#[pyo3(name = "gnr8")]
fn gnr8(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // py.import("torch")?;
    m.add_function(wrap_pyfunction!(gen::generate_token, m)?)?;
    m.add_class::<gen::GenerationConfig>()?;
    Ok(())
}
