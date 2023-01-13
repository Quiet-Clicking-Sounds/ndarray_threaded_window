use std::thread;

use ndarray::{
    s, Array, ArrayView, Axis, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, RemoveAxis, SliceInfo,
    SliceInfoElem,
};

pub trait WinSh {
    type SliceType;
    /// generate window shape from slice
    ///
    /// - window shape arguments must be >= 1
    /// - window shape arguments which are shorter than the array dimension will have 1's inserted at the end
    /// - window shape arguments longer than the dimension will be ignored
    ///
    /// # examples
    /// ```
    /// let win: &[usize] = Ix1::from_slice(&[1,2,3,4,5]).slice();
    /// assert_eq!( win, &[1]);
    /// let win: &[usize] = Ix3::from_slice(&[1,2,3,4,5]).slice();
    /// assert_eq!( win, &[1,2,3]);
    /// let win: &[usize] = Ix5::from_slice(&[1,2,3,4,5]).slice();
    /// assert_eq!( win, &[1,2,3,4,5]);
    /// let win: &[usize] = Ix5::from_slice(&[5]).slice(); //
    /// assert_eq!( win, &[5,1,1,1,1,]);
    /// ```
    ///
    fn from_slice(sh: &[usize]) -> Self;
    /// calculate how much smaller the output array will be than the input array
    /// used to work out how the overlaps will be made
    fn size_sub_shape(&self, win: &Self) -> Self;
    /// return a slice argument `ndarray:s![]`
    /// * `a` - slice start position
    /// * `b` - slice end position
    /// * `d` - dimension to be sliced
    /// # Equivalents
    /// ```
    /// let slc = Ix3::slice_convert(100,300,2);
    /// let sl_ = ndarray:s!s![.., 100..300, ..];
    /// // slc and sl_ should act the same way
    /// ```
    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType;
    /// helper for slicing an array using [WinSh::slice_convert]
    /// * `a` - slice start position
    /// * `b` - slice end position
    /// * `d` - dimension to be sliced
    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Self>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Self>
    where
        Self: Sized;
    /// helper function,
    /// equivalent to `array.shape()[d]`
    /// * `d` - dimension to find the length of
    fn get_dimension_length(&self, d: &usize) -> usize;
    /// concatenate an array over the specified dimension
    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D>;
}

impl WinSh for Ix1 {
    type SliceType = SliceInfo<[SliceInfoElem; 1], Ix1, Ix1>;
    fn from_slice(sh: &[usize]) -> Self {
        let mut shi = sh.into_iter();
        Ix1(*shi.next().unwrap_or(&1usize))
    }

    fn size_sub_shape(&self, win: &Self) -> Self {
        Ix1(self[0].saturating_sub(win[0].saturating_sub(1)))
    }
    #[allow(unused_variables)]
    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType {
        s![*a..*b]
    }

    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Ix1>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Ix1> {
        input_array.slice(Self::slice_convert(a, b, d))
    }

    fn get_dimension_length(&self, d: &usize) -> usize {
        self[*d]
    }

    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D> {
        ndarray::concatenate(Axis(*d), arr_vec.as_slice()).unwrap()
    }
}

impl WinSh for Ix2 {
    type SliceType = SliceInfo<[SliceInfoElem; 2], Ix2, Ix2>;

    fn from_slice(sh: &[usize]) -> Self {
        let mut shi = sh.into_iter();
        Ix2(
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
        )
    }

    fn size_sub_shape(&self, win: &Self) -> Self {
        Ix2(
            self[0].saturating_sub(win[0].saturating_sub(1)),
            self[1].saturating_sub(win[1].saturating_sub(1)),
        )
    }

    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType {
        match d {
            0 => s![*a..*b, ..],
            1 => s![.., *a..*b],
            _ => panic!("not supported"),
        }
    }

    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Ix2>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Ix2> {
        input_array.slice(Self::slice_convert(a, b, d))
    }

    fn get_dimension_length(&self, d: &usize) -> usize {
        self[*d]
    }

    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D> {
        ndarray::concatenate(Axis(*d), arr_vec.as_slice()).unwrap()
    }
}

impl WinSh for Ix3 {
    type SliceType = SliceInfo<[SliceInfoElem; 3], Ix3, Ix3>;
    fn from_slice(sh: &[usize]) -> Self {
        let mut shi = sh.into_iter();
        Ix3(
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
        )
    }

    fn size_sub_shape(&self, win: &Self) -> Self {
        Ix3(
            self[0].saturating_sub(win[0].saturating_sub(1)),
            self[1].saturating_sub(win[1].saturating_sub(1)),
            self[2].saturating_sub(win[2].saturating_sub(1)),
        )
    }

    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType {
        match d {
            0 => s![*a..*b, .., ..],
            1 => s![.., *a..*b, ..],
            2 => s![.., .., *a..*b],
            _ => panic!("not supported"),
        }
    }

    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Ix3>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Ix3> {
        input_array.slice(Self::slice_convert(a, b, d))
    }

    fn get_dimension_length(&self, d: &usize) -> usize {
        self[*d]
    }

    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D> {
        ndarray::concatenate(Axis(*d), arr_vec.as_slice()).unwrap()
    }
}

impl WinSh for Ix4 {
    type SliceType = SliceInfo<[SliceInfoElem; 4], Ix4, Ix4>;
    fn from_slice(sh: &[usize]) -> Self {
        let mut shi = sh.into_iter();
        Ix4(
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
        )
    }

    fn size_sub_shape(&self, win: &Self) -> Self {
        Ix4(
            self[0].saturating_sub(win[0].saturating_sub(1)),
            self[1].saturating_sub(win[1].saturating_sub(1)),
            self[2].saturating_sub(win[2].saturating_sub(1)),
            self[3].saturating_sub(win[3].saturating_sub(1)),
        )
    }

    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType {
        match d {
            0 => s![*a..*b, .., .., ..],
            1 => s![.., *a..*b, .., ..],
            2 => s![.., .., *a..*b, ..],
            3 => s![.., .., .., *a..*b],
            _ => panic!("not supported"),
        }
    }

    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Ix4>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Ix4> {
        input_array.slice(Self::slice_convert(a, b, d))
    }

    fn get_dimension_length(&self, d: &usize) -> usize {
        self[*d]
    }

    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D> {
        ndarray::concatenate(Axis(*d), arr_vec.as_slice()).unwrap()
    }
}

impl WinSh for Ix5 {
    type SliceType = SliceInfo<[SliceInfoElem; 5], Ix5, Ix5>;
    fn from_slice(sh: &[usize]) -> Self {
        let mut shi = sh.into_iter();
        Ix5(
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
            *shi.next().unwrap_or(&1usize),
        )
    }

    fn size_sub_shape(&self, win: &Self) -> Self {
        Ix5(
            self[0].saturating_sub(win[0].saturating_sub(1)),
            self[1].saturating_sub(win[1].saturating_sub(1)),
            self[2].saturating_sub(win[2].saturating_sub(1)),
            self[3].saturating_sub(win[3].saturating_sub(1)),
            self[4].saturating_sub(win[4].saturating_sub(1)),
        )
    }

    fn slice_convert(a: &usize, b: &usize, d: &usize) -> Self::SliceType {
        match d {
            0 => s![*a..*b, .., .., .., ..],
            1 => s![.., *a..*b, .., .., ..],
            2 => s![.., .., *a..*b, .., ..],
            3 => s![.., .., .., *a..*b, ..],
            4 => s![.., .., .., .., *a..*b],
            _ => panic!("not supported"),
        }
    }

    fn slice_array<'a, T: Clone>(
        input_array: &'a Array<T, Ix5>,
        a: &usize,
        b: &usize,
        d: &usize,
    ) -> ArrayView<'a, T, Ix5> {
        input_array.slice(Self::slice_convert(a, b, d))
    }

    fn get_dimension_length(&self, d: &usize) -> usize {
        self[*d]
    }

    fn concat_over_dim<T: Clone, D: Dimension + RemoveAxis>(
        arr_vec: Vec<ArrayView<T, D>>,
        d: &usize,
    ) -> Array<T, D> {
        ndarray::concatenate(Axis(*d), arr_vec.as_slice()).unwrap()
    }
}
static THREAD_ENVIRONMENT_VARIABLE_NAME: &str = "SET_THREADS";
fn proc_from_env() -> Result<usize,  &'static str> {
    let c = match std::env::var(THREAD_ENVIRONMENT_VARIABLE_NAME){
        Ok(u) => u,
        Err(_) => return Err("Could Not Read")
    };
    let c = match  c.parse::<usize>() {
        Ok(u) => u,
        Err(_) => return Err("Did Not Parse")
    };
    if (c > 0) & (c < 100) {
        return Ok(c);
    }
    return Err("core env set out of bounds");
}

/// set the number of threads to be used
pub fn set_thread_env_var(threads: usize) {
    std::env::set_var(THREAD_ENVIRONMENT_VARIABLE_NAME, format!("{}", threads));
    match std::env::var(THREAD_ENVIRONMENT_VARIABLE_NAME) {
        Ok(x) => println!(
            "Environment Variable {:?} set to: {:?}",
            THREAD_ENVIRONMENT_VARIABLE_NAME, x
        ),
        Err(_) => println!("Warning: Environment Variable Not Set"),
    }
}

/// try to get available threads, or return 12
pub fn get_proc_count() -> usize {
    match proc_from_env() {
        Ok(c) => return c,
        Err(_) => match thread::available_parallelism() {
                Ok(x) => usize::from(x),
                Err(_) => {
                    println!("Thread Count not available, defaulting to 12");
                    12usize
                }

        }
    }

}
/// generate split positions over the largest dimension of an array
/// `arr` * Array to be split
/// `split_size` * splitting shape to be used
///
/// ```
/// use ndarray::{Array3,Ix3};
/// use ndarray_threaded_window::array_shape_traits::set_thread_env_var;
/// use ndarray_threaded_window::array_shape_traits::{ArraySplitter, WinSh};
///
/// set_thread_env_var(2); // output shape will change based on available threads
/// let array: Array3<u8>= Array3::zeros((5,50,5));
/// let window: Ix3 = Ix3::from_slice(&[2,2,2]);
/// let splitter = ArraySplitter::new(&array, &window);
/// // usually returns a vec, we only care about looking at the first item
/// let binding = splitter.slice_position_vec();
/// let (a,b,d) = binding.first().unwrap();
/// let array_split_one = Ix3::slice_array(&array,a,b,d);
/// assert_eq!(array_split_one.shape(), &[5,26,5])
/// ```
pub struct ArraySplitter {
    mod_dim: usize,
    mod_vec: Vec<(usize, usize)>,
}

impl ArraySplitter {
    /// array dimensions and split must be the same type
    ///
    /// ``` rust
    /// let array: Array3<u8>= Array3::zeros((5,50,5));
    /// let window: Ix3 = Ix3::from_slice(&[2,2,2]);
    /// let splitter = ArraySplitter::new(&array, &window);
    /// ```
    pub fn new<T, D>(arr: &Array<T, D>, split_size: &D) -> Self
    where
        D: Dimension + WinSh,
    {
        // this is (dimension number, dimension size) for now
        let (mut array_dimension_length, mut modifier_dim) = (0usize, 0usize);
        for (dim_num, &dim_len) in arr.shape().into_iter().enumerate() {
            // keep the largest dimension (we want to iterate over this dim)
            if dim_len > array_dimension_length {
                (modifier_dim, array_dimension_length) = (dim_num, dim_len)
            }
        }

        let split_dimension_length = split_size.get_dimension_length(&modifier_dim);
        let cores = get_proc_count();
        let size_of_new_split = (array_dimension_length - split_dimension_length.saturating_sub(1))
            as f32
            / cores as f32;

        let split_vec: Vec<(usize, usize)> = (0..cores)
            .map(|index_u| {
                let index_f32 = index_u as f32;
                let begin_position = index_f32 * size_of_new_split;
                let end_position = (index_f32 + 1f32) * size_of_new_split;
                // last split needs to grab any leftover stuff, this just makes sure rounding errors
                // don't break things again, could be refactored out at some point
                (
                    begin_position.round() as usize,
                    match index_u == cores {
                        true => array_dimension_length,
                        false => {
                            end_position.round() as usize + split_dimension_length.saturating_sub(1)
                        }
                    },
                )
            })
            .collect();
        Self {
            mod_dim: modifier_dim,
            mod_vec: split_vec,
        }
    }

    pub fn slice_position_vec(&self) -> Vec<(usize, usize, usize)> {
        self.mod_vec
            .iter()
            .map(|(a, b)| (*a, *b, self.mod_dim))
            .collect()
    }

    /// restack arrays along the axis they were split
    pub fn restack<T, D>(&self, data: Vec<Array<T, D>>) -> Array<T, D>
    where
        T: Clone,
        D: WinSh + Dimension + RemoveAxis,
    {
        let arr_stack: Vec<ArrayView<T, D>> = data.iter().map(|a| a.view()).collect();
        D::concat_over_dim(arr_stack, &self.mod_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_env_setting() {
        let thread = 7usize;
        set_thread_env_var(thread);
        assert_eq!(thread, get_proc_count());
        set_thread_env_var(0);
        let _ = get_proc_count();
    }

}
