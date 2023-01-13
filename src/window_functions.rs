use ndarray::{ArrayView, Dimension, Ix1};

use crate::integer_conversion_traits::NumConv;

pub type WinFunc<T, D> = fn(ArrayView<T, D>) -> T;
/// helper to return a function from it's identifier, used for python implementation
#[cfg(feature = "default")]
pub fn get_func<T, D>(i: usize) -> Result<fn(ArrayView<T, D>) -> T, &'static str>
where
    T: NumConv + Clone,
    D: Dimension,
{
    match get_function_information(i) {
        Ok((f, _, _)) => Ok(f),
        Err(x) => Err(x),
    }
}
#[cfg(feature = "default")]
fn get_function_information<T, D>(
    i: usize,
) -> Result<(fn(ArrayView<T, D>) -> T, &'static str, &'static str), &'static str>
where
    T: NumConv + Clone,
    D: Dimension,
{
    match i {
        0 => Ok((
            func_window_max,
            "func_window_max",
            "return the maximum value of the window",
        )),
        1 => Ok((
            func_window_min,
            "func_window_min",
            "return the minimum value of the window",
        )),
        2 => Ok((
            func_stdev_ddof_0,
            "func_stdev_ddof_0",
            "apply rust ndarray::std(ddof:0) over the window (uses float64 values) \
            then round to input dtype",
        )),
        3 => Ok((
            func_stdev_ddof_1,
            "func_stdev_ddof_1",
            "apply rust ndarray::std(ddof:1) over the window (uses float64 values) \
            then round to input dtype",
        )),
        4 => Ok((
            func_area_contrast,
            "func_area_contrast",
            "TODO:Create Description",
        )),
        5 => Ok((
            func_fast_std,
            "func_fast_std",
            "similar to standard deviation, trades precision for speed, uses integer addition for \
            first stage the n float 64 for second stage before returning as input dtype",
        )),
        6 => Ok((
            func_fast_std_clamp,
            "func_fast_std_clamp",
            "run func_fast_std then double before converting back into input type",
        )),
        7 => Ok((
            func_fast_population_std,
            "func_fast_population_std",
            "TODO:Create Description",
        )),
        8 => Ok((
            func_fast_sample_std,
            "func_fast_sample_std",
            "TODO:Create Description",
        )),
        _ => Err("No Function Found for Value"),
    }
}
/// helper to get function name, used for python implementation
#[cfg(feature = "default")]
pub fn get_func_name(i: usize) -> Result<&'static str, &'static str> {
    match get_function_information::<u32, Ix1>(i) {
        Ok((_, x, _)) => Ok(x),
        Err(x) => Err(x),
    }
}
/// helper to get function description, used for python implementation
#[cfg(feature = "default")]
pub fn get_func_description(i: usize) -> Result<&'static str, &'static str> {
    match get_function_information::<u32, Ix1>(i) {
        Ok((_, _, x)) => Ok(x),
        Err(x) => Err(x),
    }
}

pub fn func_window_max<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv + Clone,
{
    w.iter().fold(T::MIN, |a: T, f: &T| a.max(f.clone()))
}

pub fn func_window_min<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv + Clone,
{
    w.iter().fold(T::MAX, |a: T, f: &T| a.min(f.clone()))
}

/// builtin rms with ndarray
pub fn func_stdev_ddof_0<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv + Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(0f64);
    T::from_f64(w)
}

/// builtin standard deviation calculations with ndarray
pub fn func_stdev_ddof_1<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv + Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(1f64);
    T::from_f64(w)
}

/// based on [wikipedia RMS](https://wikimedia.org/api/rest_v1/media/math/render/svg/0197e4c18468102bbe81e936bca27f87e03cf7f8)
pub fn func_area_contrast<T, D>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    D: Dimension,
{
    let len = w.len() as f64;
    let (mut p1, mut p2) = (0f64, 0u64);
    for (i, x) in w.iter().enumerate() {
        p2 = p2 + x.as_u64();
        p1 = p1 + x.as_f64().powi(i as i32) - p2 as f64;
    }

    let std = p1.sqrt() / len;
    T::from_f64(std)
}

/// faster but less precise than the builtin standard deviation calculation,
/// uses unsigned integer addition
/// overflows are possible on u64 arrays, or u32 arrays with more than `4*10^6` numbers
pub fn func_fast_std<T, D>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    D: Dimension,
{
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = T::larger_int_as_f64(
        w.iter()
            .fold(T::L_ZERO, |a: T::LargerInt, x: &T| a + x.as_larger_int()),
    ) * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64() - mean).abs().powi(2));

    T::from_f64((flt * len_inv).sqrt())
}

/// almost equivalent to ``` ndarray::array.std(ddof=0) ```
///
/// ```
/// use ndarray::{aview0, aview1, arr1};
/// let array = arr1(&[1, 2, 3, 4, 5, 6]);
/// let pop_std = func_fast_population_std(array.view());
/// let nd_std = array.std(0);
/// assert_eq![pop_std, nd_std] // this is not always correct 2023-01-13
/// ```
/// uses unsigned integer addition
/// overflows are possible on u64 arrays, or u32 arrays with more than `4*10^6` numbers
pub fn func_fast_population_std<T, D>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    D: Dimension,
{
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = T::larger_int_as_f64(
        w.iter()
            .fold(T::L_ZERO, |a: T::LargerInt, x: &T| a + x.as_larger_int()),
    ) * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64() - mean).powi(2));

    T::from_f64((flt * len_inv).sqrt())
}

/// almost equivalent to ```ndarray::array.std(ddof=1))```
/// ```
/// use ndarray::{aview0, aview1, arr1};
/// let array = arr1(&[1, 2, 3, 4, 5, 6]);
/// let pop_std = func_fast_population_std(array.view());
/// let nd_std = array.std(1);
/// assert_eq![pop_std, nd_std] // this is not always correct 2023-01-13
/// ```
/// uses unsigned integer addition
/// overflows are possible on u64 arrays, or u32 arrays with more than `4*10^6` numbers
pub fn func_fast_sample_std<T, D>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    D: Dimension,
{
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = T::larger_int_as_f64(
        w.iter()
            .fold(T::L_ZERO, |a: T::LargerInt, x: &T| a + x.as_larger_int()),
    ) * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64().abs() - mean).powi(2));

    T::from_f64((flt * ((w.len() - 1) as f64).recip()).sqrt())
}

/// standard deviation calculated with [func_fast_std] then doubled
/// doubling means that the maximum deviation from an array of [u8] values is [u8::MAX]
pub fn func_fast_std_clamp<T, D>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    D: Dimension,
{
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = T::larger_int_as_f64(
        w.iter()
            .fold(T::L_ZERO, |a: T::LargerInt, x: &T| a + x.as_larger_int()),
    ) * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64() - mean).abs().powi(2));
    T::clamp_rms_max((flt * len_inv).sqrt())
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time;

    use ndarray::{Array3, Ix3};

    use crate::window_functions::{
        func_area_contrast, func_fast_std, func_fast_std_clamp, func_stdev_ddof_0,
        func_stdev_ddof_1, func_window_max, func_window_min, WinFunc,
    };

    fn not_a_hash((a, b, c): (usize, usize, usize)) -> u8 {
        ((476579u64 % (a * b * c + 1) as u64) % 256) as u8
    }

    fn generate_array3() -> Array3<u8> {
        Array3::from_shape_fn((8, 8, 1), not_a_hash)
    }

    #[test]
    fn simple_timeit_testing() {
        fn work(iters: i32, f: WinFunc<u8, Ix3>, f_name: &str) {
            let a = Array3::from_shape_fn((8, 8, 3), not_a_hash);
            let t = time::Instant::now();
            for _ in 0..iters {
                let _b = f(a.view());
            }
            let t1 = t.elapsed();
            println!("{: <20}{: >9}", f_name, t1.as_micros());
        }

        let iterations = 100000;

        work(iterations, func_area_contrast, "area_contrast");
        work(iterations, func_fast_std, "fast_std");
        work(iterations, func_fast_std_clamp, "fast_std_clamp");
        work(iterations, func_stdev_ddof_0, "stdev_ddof_0");
        work(iterations, func_stdev_ddof_1, "stdev_ddof_1");
        work(iterations, func_window_min, "window_min");
        work(iterations, func_window_max, "window_max");

        // below is just so it will print, this isnt a test
        //assert_eq!(0, 1)
    }

    #[test]
    fn test_stdev_ddof_1() {
        let a = generate_array3();
        let b = func_stdev_ddof_1(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_stdev_ddof_0() {
        let a = generate_array3();
        let b = func_stdev_ddof_0(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_faster_rms_u64_adding() {
        let a = generate_array3();
        let b = func_fast_std(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }
}
