use integer_conversion_traits::NumConv;
use window_functions::WinFunc;

use crate::array_shape_traits::{get_proc_count, set_thread_env_var, WinSh};

pub mod array_shape_traits;
pub mod array_threading;
pub mod integer_conversion_traits;
pub mod window_functions;

#[cfg(feature = "default")]
mod python_lib {
    use ndarray::{Array, ArrayView1, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn};
    use numpy::{
        PyArray, PyReadonlyArray1, PyReadonlyArrayDyn, ToPyArray,
    };
    use pyo3::{
        prelude::pyfunction, prelude::pymodule, prelude::PyModule, prelude::PyResult,
        prelude::Python, wrap_pyfunction, wrap_pymodule, IntoPy, PyObject,
    };

    use super::*;

    /// ndarray_threaded_window
    #[pymodule]
    fn ndarray_threaded_window(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add(
            "__description__",
            "module for working with windowed sub-views of an array with builtin threading",
        )?;
        m.add_wrapped(wrap_pymodule!(base_functions))?;
        m.add_wrapped(wrap_pymodule!(_nd_thread_window_subspace))?;

        // threading stuff
        m.add_function(wrap_pyfunction!(set_thread_count_envar, m)?)?;
        #[pyfunction]
        fn set_thread_count_envar(threads: usize) {
            set_thread_env_var(threads)
        }

        m.add_function(wrap_pyfunction!(get_thread_count_envar, m)?)?;
        #[pyfunction]
        fn get_thread_count_envar() -> usize {
            get_proc_count()
        }

        // functions
        m.add_function(wrap_pyfunction!(print_available_functions, m)?)?;

        let mut counter = 0usize;
        loop {
            match window_functions::get_func_name(counter) {
                Ok(x) => {
                    m.add(x, counter)?;
                    m.add(
                        &*format!("{}.__doc__", x),
                        window_functions::get_func_description(counter).unwrap(),
                    )?;
                }
                Err(_) => break,
            }
            counter += 1
        }
        Ok(())
    }
    #[pyfunction]
    fn print_available_functions() {
        let mut counter = 0usize;
        loop {
            match window_functions::get_func_name(counter) {
                Ok(x) => println!("{:?}: {:?}", counter, String::from(x)),
                Err(_) => break,
            }
            counter += 1
        }
    }

    /// Dynamic to Static array switching,
    /// TODO: write a test for this, it should be simple enough to do
    #[inline]
    fn sub_apply_window_dyn<T>(
        a: Array<T, IxDyn>,
        method: usize,
        window: &[usize],
    ) -> Array<T, IxDyn>
    where
        T: NumConv + Clone + Copy + Send + 'static,
    {
        let binding = a.dim().clone();
        let sl = binding.slice();
        match a.ndim() {
            1 => array_threading::thread_over_any_window(
                a.into_shape(Ix1::from_slice(sl)).unwrap(),
                Ix1::from_slice(window),
                window_functions::get_func::<T, Ix1>(method).unwrap(),
            )
            .into_dyn(),
            2 => array_threading::thread_over_any_window(
                a.into_shape(Ix2::from_slice(sl)).unwrap(),
                Ix2::from_slice(window),
                window_functions::get_func::<T, Ix2>(method).unwrap(),
            )
            .into_dyn(),
            3 => array_threading::thread_over_any_window(
                a.into_shape(Ix3::from_slice(sl)).unwrap(),
                Ix3::from_slice(window),
                window_functions::get_func::<T, Ix3>(method).unwrap(),
            )
            .into_dyn(),
            4 => array_threading::thread_over_any_window(
                a.into_shape(Ix4::from_slice(sl)).unwrap(),
                Ix4::from_slice(window),
                window_functions::get_func::<T, Ix4>(method).unwrap(),
            )
            .into_dyn(),
            5 => array_threading::thread_over_any_window(
                a.into_shape(Ix5::from_slice(sl)).unwrap(),
                Ix5::from_slice(window),
                window_functions::get_func::<T, Ix5>(method).unwrap(),
            )
            .into_dyn(),
            _ => panic!("Array Shape Not Implemented"),
        }
        .into_dyn()
    }

    /// ndarray_threaded_window
    #[pymodule]
    fn _nd_thread_window_subspace(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add(
            "__description__",
            "module for working with windowed sub-views of an array with builtin threading",
        )?;

        m.add_function(wrap_pyfunction!(apply_window_for_dyn_u8, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_u8<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<u8>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<u8, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }

        m.add_function(wrap_pyfunction!(apply_window_for_dyn_u16, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_u16<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<u16>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<u16, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }
        m.add_function(wrap_pyfunction!(apply_window_for_dyn_u32, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_u32<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<u32>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<u32, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }
        m.add_function(wrap_pyfunction!(apply_window_for_dyn_i8, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_i8<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<i8>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<i8, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }

        m.add_function(wrap_pyfunction!(apply_window_for_dyn_i16, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_i16<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<i16>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<i16, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }
        m.add_function(wrap_pyfunction!(apply_window_for_dyn_i32, m)?)?;
        #[pyfunction]
        fn apply_window_for_dyn_i32<'py>(
            py: Python<'py>,
            a: PyReadonlyArrayDyn<i32>,
            m: usize,
            window: Vec<usize>,
        ) -> &'py PyArray<i32, IxDyn> {
            sub_apply_window_dyn(a.to_owned_array(), m, window.as_slice()).to_pyarray(py)
        }

        Ok(())
    }

    #[pymodule]
    fn base_functions(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add(
            "__description__",
            "non-threaded, non-windowed methods that can be called on arrays",
        )?;
        fn apply_func(arr: ArrayView1<u8>, func: WinFunc<u8, Ix1>) -> u8 {
            func(arr)
        }

        m.add_function(wrap_pyfunction!(func_window_max, m)?)?;
        #[pyfunction]
        fn func_window_max(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_window_max).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_window_min, m)?)?;
        #[pyfunction]
        fn func_window_min(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_window_min).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_stdev_ddof_0, m)?)?;
        #[pyfunction]
        fn func_stdev_ddof_0(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_stdev_ddof_0).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_stdev_ddof_1, m)?)?;
        #[pyfunction]
        fn func_stdev_ddof_1(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_stdev_ddof_1).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_area_contrast, m)?)?;
        #[pyfunction]
        fn func_area_contrast(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_area_contrast).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_fast_std, m)?)?;
        #[pyfunction]
        fn func_fast_std(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_fast_std).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_fast_std_clamp, m)?)?;
        #[pyfunction]
        fn func_fast_std_clamp(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_fast_std_clamp).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_fast_population_std, m)?)?;
        #[pyfunction]
        fn func_fast_population_std(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_fast_population_std).into_py(py)
        }

        m.add_function(wrap_pyfunction!(func_fast_sample_std, m)?)?;
        #[pyfunction]
        fn func_fast_sample_std(py: Python, arr: PyReadonlyArray1<u8>) -> PyObject {
            apply_func(arr.as_array(), window_functions::func_fast_sample_std).into_py(py)
        }

        Ok(())
    }
}
