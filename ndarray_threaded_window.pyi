import numpy
import base_functions
base_functions = base_functions

fn_area_contrast: int
""" fn_area_contrast currently does not work as rms   """

fn_fast_std: int
""" fn_fast_std standard deviation function which uses integer addition.
    
    note: may overflow on u64 dtypes"""

fn_fast_std_clamp: int
""" fn_fast_std_clamp the same as :py:obj:`fn_area_contrast` * 2 

    this ensures the minimum and maximum return from an unsigned integer type are between u::MIN and u::MAX 
"""

fn_stdev_ddof_0: int
""" fn_stdev_ddof_0 converts to float64 then apply Welford one-pass algorithm from (rust ndarray)[https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.std] """

fn_stdev_ddof_1: int
""" fn_stdev_ddof_1 converts to float64 then apply Welford one-pass algorithm from (rust ndarray)[https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#method.std]"""

def apply_window(py_img: numpy.ndarray, method: int, win_size_x: int, win_size_y: int,
           win_size_z: int) -> numpy.ndarray:
    """
    method can be chosen by using the included options or from the list below with `int`
         0 = area_contrast ,
         1 = fast_std ,
         2 = fast_std_clamp ,
         3 = stdev_ddof_0 ,
         4 = stdev_ddof_1 ,

    :param py_img: input image, must be a numpy.ndarray with 3 dimensions `len(array.shape)==3`
    :param method: see method notes above

    :param win_size_x: length of window in 'x' dimension [0]
    :param win_size_y: length of window in 'y' dimension [1]
    :param win_size_z: length of window in 'z' dimension [2]
    :return: numpy array with dimensions of the input[x,y,z] - [w-1, w-1, 0]
    """
