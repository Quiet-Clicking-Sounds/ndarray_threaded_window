// #![feature(specialization)]

use ndarray::{ArrayView1, Ix1};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::pyfunction;
use pyo3::prelude::pymodule;
use pyo3::prelude::PyModule;
use pyo3::prelude::PyResult;
use pyo3::prelude::Python;
use pyo3::{pyclass, pymethods, wrap_pyfunction, wrap_pymodule, IntoPy, PyObject};

use threading::{thread_apply_over_window, WindowShape};

use crate::window_functions::WinFunc;

mod threading;
mod window_functions;

/// ndarray_threaded_window
#[pymodule]
fn ndarray_threaded_window(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__description__",
        "module for working with windowed sub-views of an array with builtin threading",
    )?;
    m.add_wrapped(wrap_pymodule!(base_functions))?;
    m.add_function(wrap_pyfunction!(apply_window, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(text_signature = "(py_img:numpy.ndarray,win_fn:int window_size:int, /)")]
fn apply_window<'py>(
    py: Python<'py>,
    array: PyReadonlyArray3<u8>,
    method: usize,
    win_size_x: usize,
    win_size_y: usize,
    win_size_z: usize,
) -> &'py PyArray3<u8> {
    let window_type = WindowShape::Triple(win_size_x, win_size_y, win_size_z);
    let m = match method {
        0 => window_functions::user_rms,
        1 => window_functions::fast_std,
        2 => window_functions::fast_std_clamp,
        3 => window_functions::stdev_ddof_0,
        4 => window_functions::stdev_ddof_1,
        _ => panic!("window method not found"),
    };
    let array_out = thread_apply_over_window(array.to_owned_array(), window_type, m);
    array_out.to_pyarray(py)
}

#[pymodule]
fn base_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add(
        "__description__",
        "non-threaded, non-windowed methods that can be called on arrays",
    )?;
    fn apply_func(arr: ArrayView1<u8>, func: WinFunc<u8, Ix1>) -> u8 {
        func(arr)
    }

    m.add_function(wrap_pyfunction!(stdev_ddof_0, m)?)?;
    #[pyfunction]
    fn stdev_ddof_0(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
        apply_func(arr.as_array(), window_functions::stdev_ddof_0).into_py(py)
    }

    m.add_function(wrap_pyfunction!(stdev_ddof_1, m)?)?;
    #[pyfunction]
    fn stdev_ddof_1(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
        apply_func(arr.as_array(), window_functions::stdev_ddof_1).into_py(py)
    }

    m.add_function(wrap_pyfunction!(fast_std, m)?)?;
    #[pyfunction]
    fn fast_std(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
        apply_func(arr.as_array(), window_functions::fast_std).into_py(py)
    }

    m.add_function(wrap_pyfunction!(fast_std_clamp, m)?)?;
    #[pyfunction]
    fn fast_std_clamp(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
        apply_func(arr.as_array(), window_functions::fast_std_clamp).into_py(py)
    }

    m.add_function(wrap_pyfunction!(user_rms, m)?)?;
    #[pyfunction]
    fn user_rms(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
        apply_func(arr.as_array(), window_functions::user_rms).into_py(py)
    }

    Ok(())
}
