use num_traits::{PrimInt, Zero};

pub trait NumConv: Ord + Sized + Clone + Zero {
    type LargerInt: PrimInt;
    const L_ZERO: Self::LargerInt;
    const MAX: Self;
    const MIN: Self;
    fn from_f64(f: f64) -> Self;
    fn larger_int_as_f64(f:Self::LargerInt)->f64;
    fn as_larger_int(&self) -> Self::LargerInt;
    fn as_u64(&self) -> u64;
    fn as_f64(&self) -> f64;
    fn clamp_rms_max(f: f64) -> Self;
}
impl NumConv for u8 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u8::MAX;
    const MIN: Self = u8::MIN;
    fn from_f64(f: f64) -> Self {f.round()as u8}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for i8 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i8::MAX;
    const MIN: Self = i8::MIN;
    fn from_f64(f: f64) -> Self {f.round()as i8}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for u16 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u16::MAX;
    const MIN: Self = u16::MIN;
    fn from_f64(f: f64) -> Self {f.round()as u16}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for i16 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;
    fn from_f64(f: f64) -> Self {f.round()as i16}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for u32 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = u32::MAX;
    const MIN: Self = u32::MIN;
    fn from_f64(f: f64) -> Self {f.round()as u32}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for i32 {
    type LargerInt = i64;
    const L_ZERO: Self::LargerInt = 0i64;
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;
    fn from_f64(f: f64) -> Self {f.round()as i32}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for u64 {
    type LargerInt = i128;
    const L_ZERO: Self::LargerInt = 0i128;
    const MAX: Self = u64::MAX;
    const MIN: Self = u64::MIN;
    fn from_f64(f: f64) -> Self {f.round() as u64}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}

impl NumConv for i64 {
    type LargerInt = i128;
    const L_ZERO: Self::LargerInt = 0i128;
    const MAX: Self = i64::MAX;
    const MIN: Self = i64::MIN;
    fn from_f64(f: f64) -> Self {f.round()as i64}
    fn larger_int_as_f64(f:Self::LargerInt)->f64{f as f64}
    fn as_larger_int(&self) -> Self::LargerInt {*self as Self::LargerInt}
    fn as_u64(&self) -> u64 {*self as u64}
    fn as_f64(&self) -> f64 {*self as f64}
    fn clamp_rms_max(f: f64) -> Self {Self::from_f64(f * 2.0)}
}
