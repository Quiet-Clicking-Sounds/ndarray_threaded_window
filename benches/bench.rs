#![cfg(test)]
#![feature(test)]
extern crate alloc;
extern crate test;

use std::ops::BitXor;
use test::Bencher;

use ndarray::Array1;
use ndarray::{Array, Ix1, Ix2, Ix3, Ix4, Ix5};

use crate::array_shape_traits::WinSh;
use crate::array_threading::apply_over_any_window;
use crate::array_threading::thread_over_any_window;
use crate::integer_conversion_traits::IntConv;
use crate::window_functions::func_fast_std;
use crate::window_functions::*;

#[path = "../src/array_shape_traits.rs"]
mod array_shape_traits;
#[path = "../src/array_threading.rs"]
mod array_threading;
#[path = "../src/integer_conversion_traits.rs"]
mod integer_conversion_traits;
#[path = "../src/window_functions.rs"]
mod window_functions;

mod bench_window_functions {
    use super::*;

    fn standard_array() -> Array1<u16> {
        Array1::from_shape_fn(512, |c| c as u16)
    }

    #[bench]
    fn bench_window_max(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_window_max(arr.view()))
    }
    #[bench]
    fn bench_window_min(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_window_min(arr.view()))
    }
    #[bench]
    fn bench_stdev_ddof_0(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_stdev_ddof_0(arr.view()))
    }
    #[bench]
    fn bench_stdev_ddof_1(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_stdev_ddof_1(arr.view()))
    }
    #[bench]
    fn bench_area_contrast(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_area_contrast(arr.view()))
    }
    #[bench]
    fn bench_fast_std(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_fast_std(arr.view()))
    }
    #[bench]
    fn bench_fast_std_clamp(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_fast_std_clamp(arr.view()))
    }
    #[bench]
    fn bench_fast_std_pure_int(b: &mut Bencher) {
        let arr = standard_array();
        b.iter(|| func_fast_std_pure_int(arr.view()))
    }
}
mod threading_any_bench {
    use super::*;

    const BENCH_WINDOW: usize = 25;
    const L4K: usize = 4 * 1000;
    const L1M: usize = 1 * 1000 * 1000;
    fn gen1_len(size: usize) -> Array<u8, Ix1> {
        Array::from_shape_fn(size, |a| a as u8)
    }
    #[bench]
    fn benchmark_array_4k_win_25_single_thread(b: &mut Bencher) {
        let win = Ix1::from_slice(&[BENCH_WINDOW]);
        let ar_a = gen1_len(L4K);
        println!(
            "MultiThread - Array Size {:?} - Window Size {:?} - Threads {:?}",
            ar_a.len(),
            BENCH_WINDOW,
            1
        );
        b.iter(|| {
            let _b = apply_over_any_window(ar_a.clone(), win, func_fast_std_pure_int);
        })
    }
    #[bench]
    fn benchmark_array_4k_win_25_multi_thread(b: &mut Bencher) {
        let win = Ix1::from_slice(&[BENCH_WINDOW]);
        let ar_a = gen1_len(L4K);
        println!(
            "MultiThread - Array Size {:?} - Window Size {:?} - Threads {:?}",
            ar_a.len(),
            BENCH_WINDOW,
            array_shape_traits::get_proc_count()
        );
        b.iter(|| {
            let _b = thread_over_any_window(ar_a.clone(), win, func_fast_std_pure_int);
        })
    }
    #[bench]
    fn benchmark_array_1m_win_25_single_thread(b: &mut Bencher) {
        let win = Ix1::from_slice(&[BENCH_WINDOW]);
        let ar_a = gen1_len(L1M);
        println!(
            "MultiThread - Array Size {:?} - Window Size {:?} - Threads {:?}",
            ar_a.len(),
            BENCH_WINDOW,
            1
        );
        b.iter(|| {
            let _b = apply_over_any_window(ar_a.clone(), win, func_fast_std_pure_int);
        })
    }
    #[bench]
    fn benchmark_array_1m_win_25_multi_thread(b: &mut Bencher) {
        let win = Ix1::from_slice(&[BENCH_WINDOW]);
        let ar_a = gen1_len(L1M);
        println!(
            "MultiThread - Array Size {:?} - Window Size {:?} - Threads {:?}",
            ar_a.len(),
            BENCH_WINDOW,
            array_shape_traits::get_proc_count()
        );
        b.iter(|| {
            let _b = thread_over_any_window(ar_a.clone(), win, func_fast_std_pure_int);
        })
    }
}

mod array_access_bench {
    use std::ops::Deref;

    use ndarray::iter::Iter;
    use ndarray::{Array1, ArrayBase, ArrayView1, AssignElem, Dimension, OwnedRepr};

    use super::*;

    const SH1: usize = 3000;
    const W: usize = 200;
    const SH2: usize = SH1 - (W - 1);

    fn func1(t: ArrayView1<u8>) -> u8 {
        t.iter().fold(0u8, |a, b| a.min(*b))
    }
    fn make_array() -> Array1<u8> {
        Array1::ones(SH1)
    }
    #[bench]
    fn func_1_assign(b: &mut Bencher) {
        b.iter(|| {
            let ar2 = make_array();
            let ar2 = ar2.windows(W);
            let mut ar1: Array1<u8> = Array1::zeros(SH2);
            for (a, w) in ar1.iter_mut().zip(ar2.clone().into_iter()) {
                a.assign_elem(func1(w))
            }
        })
    }
    #[bench]
    fn func_1_deref(b: &mut Bencher) {
        b.iter(|| {
            let ar2 = make_array();
            let ar2 = ar2.windows(W);
            let mut ar1: Array1<u8> = Array1::zeros(SH2);
            for (a, w) in ar1.iter_mut().zip(ar2.into_iter()) {
                *a = func1(w)
            }
        })
    }
}
