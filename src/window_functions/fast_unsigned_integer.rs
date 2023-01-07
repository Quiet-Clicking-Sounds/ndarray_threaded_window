use ndarray::{ArrayView, Dimension, Ix3};
use crate::window_functions::WinFunc;


pub trait NumConv {
    fn from_f64(f: f64) -> Self;
    fn as_u64(&self) -> u64;
    fn as_f64(&self) -> f64;
    fn clamp_rms_max(f: f64) -> Self;
}

impl NumConv for u8 {
    fn from_f64(f: f64) -> Self {
        f as u8
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u16 {
    fn from_f64(f: f64) -> Self {
        f as u16
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u32 {
    fn from_f64(f: f64) -> Self {
        f as u32
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

impl NumConv for u64 {
    fn from_f64(f: f64) -> Self {
        f as u64
    }
    fn as_u64(&self) -> u64 {
        *self
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    fn clamp_rms_max(f: f64) -> Self {
        Self::from_f64(f * 2.0)
    }
}

/// builtin rms with ndarray
pub fn stdev_ddof_0<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    T: Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(0f64);
    T::from_f64(w)
}

/// builtin standard deviation calculations with ndarray
pub fn stdev_ddof_1<T, D: Dimension>(w: ArrayView<T, D>) -> T
where
    T: NumConv,
    T: Clone,
{
    let w = w.mapv(|elem: T| elem.as_f64());
    let w = w.std(1f64);
    T::from_f64(w)
}

/// based on [wikipedia RMS](https://wikimedia.org/api/rest_v1/media/math/render/svg/0197e4c18468102bbe81e936bca27f87e03cf7f8)
pub fn user_rms<T: NumConv, D: Dimension>(w: ArrayView<T, D>) -> T {
    let len = w.len() as f64;
    let sum_sq: f64 = w.iter().fold(0u64, |a: u64, x: &T| a + x.as_u64().pow(2)) as f64;
    let stdev = (sum_sq / len).sqrt();
    T::from_f64(stdev)
}

/// faster but less precise than the builtin standard deviation calculation,
/// uses unsigned integer addition
/// overflows are possible on u64 arrays, or u32 arrays with more than `4*10^6` numbers
pub fn fast_std<T: NumConv, D: Dimension>(w: ArrayView<T, D>) -> T {
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = w.iter().fold(0u64, |a: u64, x: &T| a + x.as_u64()) as f64 * len_inv;
    let flt: f64 = w
        .iter()
        .fold(0f64, |a: f64, x: &T| a + (x.as_f64() - mean).abs().powi(2));

    T::from_f64((flt * len_inv).sqrt())
}

/// standard deviation calculated with [fast_std] then doubled
/// doubling means that the maximum deviation from an array of [u8] values is [u8::MAX]
pub fn fast_std_clamp<T: NumConv, D: Dimension>(w: ArrayView<T, D>) -> T {
    let len_inv = (w.len() as f64).recip();
    let mean: f64 = w.iter().fold(0u64, |a: u64, x: &T| a + x.as_u64()) as f64 * len_inv;
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
    use crate::window_functions::WinFunc;
    use ndarray::Array3;

    use window::window_methods::*;

    use window_functions::fast_unsigned_integer::*;

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
                let b = f(a.view());
            }
            let t1 = t.elapsed();
            println!("{: <20}{: >9}", f_name, t1.as_micros());
        }

        let iterations = 100000;

        work(iterations, user_rms, "user_rms");
        work(iterations, fast_std, "fast_std");
        work(iterations, fast_std_clamp, "fast_std_clamp");
        work(iterations, stdev_ddof_0, "stdev_ddof_0");
        work(iterations, stdev_ddof_1, "stdev_ddof_1");

        // below is just so it will print, this isnt a test
        //assert_eq!(0, 1)
    }

    #[test]
    fn test_stdev_ddof_1() {
        let a = generate_array3();
        let b = stdev_ddof_1(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_stdev_ddof_0() {
        let a = generate_array3();
        let b = stdev_ddof_0(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }

    #[test]
    fn test_faster_rms_u64_adding() {
        let a = generate_array3();
        let b = fast_std(a.view());

        let mut hasher = DefaultHasher::new();
        b.hash(&mut hasher);
        assert_eq!(7541581120933061747, hasher.finish());
    }
}
