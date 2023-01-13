import base_functions

base_functions = base_functions


# func_area_contrast: int
# """ func_area_contrast currently does not work as rms   """
#
# func_fast_std: int
# """ func_fast_std standard deviation function which uses integer addition.
#
#     fast_std is around twice as fast as func_stdev_ddof_0,
#     for most inputs of u8, u16, and u32 it should be within rounding error
#
#     note: may overflow on u64 dtypes"""
#
# func_fast_std_clamp: int
# """ func_fast_std_clamp the same as :py:obj:`func_area_contrast` * 2
#
#     this ensures the minimum and maximum return from an unsigned integer type are between u::MIN and u::MAX
# """
#
# func_stdev_ddof_0: int
# """ func_stdev_ddof_0 converts to float64 then apply Welford one-pass algorithm from (rust ndarray)[https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.std] """
#
# func_stdev_ddof_1: int
# """ func_stdev_ddof_1 converts to float64 then apply Welford one-pass algorithm from (rust ndarray)[https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.std]"""
#

def set_thread_count_envar(threads: int):
    ...


def get_thread_count_envar() -> int:
    ...


def print_available_functions():
    """
    print available window functions
    """
    ...
