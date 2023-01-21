
use num_traits::{ PrimInt, Zero};


pub trait IntConv: Ord + Sized + Clone + Zero {
    type LargerInt: PrimInt;
    /// Integer value which is at minimum 32bits larger than Self
    /// with unsigned or signed the same as Self
    const L_ZERO: Self::LargerInt;
    /// Largest possible value
    const MAX: Self;
    /// Smallest possible value
    const MIN: Self;
    fn from_f64(f: f64) -> Self;
    /// [Self::LargerInt] into [f64]
    fn larger_int_as_f64(f:Self::LargerInt)->f64;
    /// see [Self::LargerInt] used for addition.
    fn as_larger_int(&self) -> Self::LargerInt;
    fn as_f64(&self) -> f64;
}
impl NumConv for u8 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u8::MAX;
    const MIN: Self = u8::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as u8}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl IntConv for i8 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i8::MAX;
    const MIN: Self = i8::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as i8}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl NumConv for u16 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u16::MAX;
    const MIN: Self = u16::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as u16}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl IntConv for i16 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as i16}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl NumConv for u32 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u32::MAX;
    const MIN: Self = u32::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as u32}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl IntConv for i32 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as i32}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl NumConv for u64 {
    type LargerInt = i128;
    const L_ZERO: Self::LargerInt = 0i128;
    const MAX: Self = u64::MAX;
    const MIN: Self = u64::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round() as u64}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

impl IntConv for i64 {
    type LargerInt = i128;
    const L_ZERO: Self::LargerInt = 0i128;
    const MAX: Self = i64::MAX;
    const MIN: Self = i64::MIN;
    #[inline]
    fn from_f64(f: f64) -> Self {f.round()as i64}
    #[inline]
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    #[inline]
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    #[inline]
    fn as_f64(&self) -> f64 {*self as f64}
}

// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

pub trait SignedInt: Sized + Clone{
    type LargerS: PrimInt;
    const L_ZERO: Self::LargerS;
    fn to_larger(&self) ->Self::LargerS;
    fn from_f64(f:f64) ->Self;
    fn larger_to_f64(f:Self::LargerS) ->f64;
    fn usize_to_larger(f:usize) ->Self::LargerS;
    fn larger_pow2(f:Self::LargerS) -> Self::LargerS;
}
impl SignedInt for u8 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as u8}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
impl SignedInt for i8 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as i8}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
impl SignedInt for u16 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as u16}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
impl SignedInt for i16 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as i16}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
impl SignedInt for u32 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as u32}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
impl SignedInt for i32 {
    type LargerS = i64;
    const L_ZERO:Self::LargerS = 0i64;
    fn to_larger(&self)->Self::LargerS{*self as i64}
    fn from_f64(f:f64)->Self{f as i32}
    fn larger_to_f64(f:Self::LargerS)->f64{f as f64}
    fn usize_to_larger(f:usize)->Self::LargerS{f as i64}
    fn larger_pow2(f: Self::LargerS) -> Self::LargerS {f.pow(2)}
}
pub trait FloatConv{

}
impl FloatConv for f32{

}
impl FloatConv for f64{

}
