use ndarray::ArrayView;

pub mod fast_unsigned_integer;

pub type WinFunc<T, D> = fn(ArrayView<T, D>) -> T;

pub use fast_unsigned_integer::*;
