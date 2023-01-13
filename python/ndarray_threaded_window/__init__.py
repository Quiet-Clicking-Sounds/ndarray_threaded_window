import numpy
from ndarray_threaded_window.ndarray_threaded_window import *


def apply_window(array: numpy.ndarray, method: int, window_shape: list[int]) -> numpy.ndarray:
    """
    method can be chosen by using the included options or from the list below with `int`
        0: "func_window_max"
        1: "func_window_min"
        2: "func_stdev_ddof_0"
        3: "func_stdev_ddof_1"
        4: "func_area_contrast"
        5: "func_fast_std"
        6: "func_fast_std_clamp"
        7: "func_fast_population_std"
        8: "func_fast_sample_std"

    :param array: input ndarray
    :param method: see method notes above
    :param window_shape: size of the window; expects a list of int values the same length as [array.shape]
    :return: numpy array with the same number of dimensions as the input
    """
    match str(array.dtype):
        case "uint8":
            return _nd_thread_window_subspace.apply_window_for_dyn_u8(array, method, window_shape)
        case "uint16":
            return _nd_thread_window_subspace.apply_window_for_dyn_u16(array, method, window_shape)
        case "uint32":
            return _nd_thread_window_subspace.apply_window_for_dyn_u32(array, method, window_shape)
        case "int8":
            return _nd_thread_window_subspace.apply_window_for_dyn_i8(array, method, window_shape)
        case "int16":
            return _nd_thread_window_subspace.apply_window_for_dyn_i16(array, method, window_shape)
        case "int32":
            return _nd_thread_window_subspace.apply_window_for_dyn_i32(array, method, window_shape)
        case x:
            raise NotImplementedError(f"Arrays of dtype {x} are not supported")
