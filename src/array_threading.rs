use std::sync::mpsc;
use std::thread;

use ndarray::{Array, ArrayView, AssignElem, Dimension, RemoveAxis};


use crate::array_shape_traits::{ArraySplitter,WinSh};
use crate::integer_conversion_traits::NumConv;
use crate::window_functions::WinFunc;


///
///
/// see [Array::windows] for an explanation of how windows work
///
/// # example:
/// ```
/// use ndarray::{Array2, Ix2};
/// use ndarray_threaded_window::window_functions::func_window_min; // an already implemented function 
/// let array: Array2<u8>= Array2::zeros((50, 5)); // make array
/// let window: Ix2 = Ix2::from_slice(&[2,2]); // shape of the window
/// let out = apply_over_any_window(array, window, func_window_min);
/// assert_eq!(out.shape(), &[49,4])
/// ```
pub fn apply_over_any_window<T, D>(arr: Array<T, D>, win_size: D, func: WinFunc<T, D>) -> Array<T, D>
where
    T: NumConv + Clone,
    D: Dimension + WinSh,
{
    let new_size = arr.raw_dim().size_sub_shape(&win_size);

    // create windowed parts of the array
    let win = arr.windows(win_size);
    // create an uninitiated base array for the output, shape descried by windowed_array_size
    let mut un_arr = Array::<T, D>::zeros(new_size);

    // iter through the output array and the windowed array
    for (a, w) in un_arr.iter_mut().zip(win.into_iter()) {
        a.assign_elem(func(w)); // assignments for some reason, I think = was being unhelpful
    }
    un_arr
}

/// # args
/// `input_array` * an n dimensional array, `func` will be applied over windows of this array
/// `win_size` * the window shape to move over the array
/// `func` * the function applied to each window in the form `fn(ArrayView<T, D>) -> T` see [ndarray_threaded_window::window_functions]
///
/// - see [Array::windows] for an explanation of how windows work
/// - apply multi threaded version of [apply_over_any_window]
/// - any use of [apply_over_any_window] can be directly replaced with [thread_over_any_window]  without changing the output
///
/// # example:
/// ```
/// use ndarray::{Array2, Ix2};
/// use ndarray_threaded_window::window_functions::func_window_min; // an already implemented function 
/// let array: Array2<u8>= Array2::zeros((50, 5)); // make array
/// let window: Ix2 = Ix2::from_slice(&[2,2]); // shape of the window
/// let out = thread_over_any_window(array, window, func_window_min);
/// assert_eq!(out.shape(), &[49,4])
/// ```
///
pub fn thread_over_any_window<T, D>(
    input_array: Array<T, D>,
    win_size: D,
    func: WinFunc<T, D>,
) -> Array<T, D>
where
    T:  NumConv + Clone + Copy + Send + 'static,
    D: Dimension + WinSh + RemoveAxis + Clone + Copy + 'static,
{
    let splitter = ArraySplitter::new(&input_array, &win_size);

    // thread return items keeper. Reminder: order here is important,
    // I'm not sending ordering information
    let mut thread_workers: Vec<_> = vec![];
    for (a, b, d) in splitter.slice_position_vec().iter() {
        let (tx, rx) = mpsc::channel();
        // create a slice view of the array before sending it
        let pre_compute_sliced_array: ArrayView<T, D> = D::slice_array(&input_array, a, b, d);
        // needs ownership, probably possible to refactor that out
        let pre_compute_slice = pre_compute_sliced_array.to_owned();
        thread::spawn(move || {
            // thread open, do compute and send to `rx`
            let computed_array_output =
                apply_over_any_window(pre_compute_slice, win_size.clone(), func);
            tx.send(computed_array_output).unwrap();
        });
        // attach the new thread receiver to the worker vec
        thread_workers.push(rx);
    }

    // export all the threads once they're finished, must wait for all finished
    // otherwise we end up with things out of order
    let array_stacks: Vec<_> = thread_workers.iter().map(|rx| rx.recv().unwrap()).collect();
    // views, concat doesn't like actual arrays
    splitter.restack(array_stacks)
}

#[cfg(test)]
mod tests {
    use std::ops::BitXor;


    use ndarray::{Array, Ix1, Ix2, Ix3, Ix4, Ix5};

    use crate::array_shape_traits::WinSh;

    use crate::array_threading::{
        apply_over_any_window, thread_over_any_window
    };
    use crate::window_functions::func_fast_std;
    use crate::integer_conversion_traits::NumConv;

    const SL1: usize = 24;
    const SL2: usize = 60;
    const SL3: usize = 12;
    const SL4: usize = 6;
    const SL5: usize = 3;
    const WIN_SHAPE: &[usize] = &[2, 4, 3, 2, 1];

    fn gen1<T: NumConv + BitXor>() -> Array<T, Ix1> {
        Array::from_shape_fn(SL1, |a| T::from_f64(a as f64))
    }

    fn gen2<T: NumConv + BitXor>() -> Array<T, Ix2> {
        Array::from_shape_fn((SL1, SL2), |(a, b)| T::from_f64((a ^ b) as f64))
    }

    fn gen3<T: NumConv + BitXor>() -> Array<T, Ix3> {
        Array::from_shape_fn((SL1, SL2, SL3), |(a, b, c)| T::from_f64((a ^ b ^ c) as f64))
    }

    fn gen4<T: NumConv + BitXor>() -> Array<T, Ix4> {
        Array::from_shape_fn((SL1, SL2, SL3, SL4), |(a, b, c, d)| {
            T::from_f64((a ^ b ^ c ^ d) as f64)
        })
    }

    fn gen5<T: NumConv + BitXor>() -> Array<T, Ix5> {
        Array::from_shape_fn((SL1, SL2, SL3, SL4, SL5), |(e, a, b, c, d)| {
            T::from_f64((a ^ b ^ c ^ d ^ e) as f64)
        })
    }

    #[test]
    fn all_for_1() {
        let win = Ix1::from_slice(WIN_SHAPE);
        let ar_a = gen1::<u8>();
        let ar_b = gen1::<u16>();
        let ar_c = gen1::<u32>();
        let oa = thread_over_any_window(ar_a.clone(), win, func_fast_std);
        let sa = apply_over_any_window(ar_a, win, func_fast_std);
        assert_eq!(oa, sa);
        let ob = thread_over_any_window(ar_b.clone(), win, func_fast_std);
        let sb = apply_over_any_window(ar_b, win, func_fast_std);
        assert_eq!(ob, sb);
        let oc = thread_over_any_window(ar_c.clone(), win, func_fast_std);
        let sc = apply_over_any_window(ar_c, win, func_fast_std);
        assert_eq!(oc, sc);
    }

    #[test]
    fn all_for_2() {
        let win = Ix2::from_slice(WIN_SHAPE);
        let ar_a = gen2::<u8>();
        let ar_b = gen2::<u16>();
        let ar_c = gen2::<u32>();
        let oa = thread_over_any_window(ar_a.clone(), win, func_fast_std);
        let sa = apply_over_any_window(ar_a, win, func_fast_std);
        assert_eq!(oa, sa);
        let ob = thread_over_any_window(ar_b.clone(), win, func_fast_std);
        let sb = apply_over_any_window(ar_b, win, func_fast_std);
        assert_eq!(ob, sb);
        let oc = thread_over_any_window(ar_c.clone(), win, func_fast_std);
        let sc = apply_over_any_window(ar_c, win, func_fast_std);
        assert_eq!(oc, sc);
    }

    #[test]
    fn all_for_3() {
        let win = Ix3::from_slice(WIN_SHAPE);
        let ar_a = gen3::<u8>();
        let ar_b = gen3::<u16>();
        let ar_c = gen3::<u32>();
        let oa = thread_over_any_window(ar_a.clone(), win, func_fast_std);
        let sa = apply_over_any_window(ar_a, win, func_fast_std);
        assert_eq!(oa, sa);
        let ob = thread_over_any_window(ar_b.clone(), win, func_fast_std);
        let sb = apply_over_any_window(ar_b, win, func_fast_std);
        assert_eq!(ob, sb);
        let oc = thread_over_any_window(ar_c.clone(), win, func_fast_std);
        let sc = apply_over_any_window(ar_c, win, func_fast_std);
        assert_eq!(oc, sc);
    }

    #[test]
    fn all_for_4() {
        let win = Ix4::from_slice(WIN_SHAPE);
        let ar_a = gen4::<u8>();
        let ar_b = gen4::<u16>();
        let ar_c = gen4::<u32>();
        let oa = thread_over_any_window(ar_a.clone(), win, func_fast_std);
        let sa = apply_over_any_window(ar_a, win, func_fast_std);
        assert_eq!(oa, sa);
        let ob = thread_over_any_window(ar_b.clone(), win, func_fast_std);
        let sb = apply_over_any_window(ar_b, win, func_fast_std);
        assert_eq!(ob, sb);
        let oc = thread_over_any_window(ar_c.clone(), win, func_fast_std);
        let sc = apply_over_any_window(ar_c, win, func_fast_std);
        assert_eq!(oc, sc);
    }

    #[test]
    fn all_for_5() {
        let win = Ix5::from_slice(WIN_SHAPE);
        let ar_a = gen5::<u8>();
        let ar_b = gen5::<u16>();
        let ar_c = gen5::<u32>();
        let oa = thread_over_any_window(ar_a.clone(), win, func_fast_std);
        let sa = apply_over_any_window(ar_a, win, func_fast_std);
        assert_eq!(oa, sa);
        let ob = thread_over_any_window(ar_b.clone(), win, func_fast_std);
        let sb = apply_over_any_window(ar_b, win, func_fast_std);
        assert_eq!(ob, sb);
        let oc = thread_over_any_window(ar_c.clone(), win, func_fast_std);
        let sc = apply_over_any_window(ar_c, win, func_fast_std);
        assert_eq!(oc, sc);
    }

}
