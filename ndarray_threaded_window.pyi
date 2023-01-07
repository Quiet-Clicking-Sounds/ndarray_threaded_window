import numpy


user: int
fast_std: int
fast_std_clamp: int
stdev_ddof_0: int
stdev_ddof_1: int

def apply_window(py_img: numpy.ndarray, method: int, win_size_x: int, win_size_y: int,
           win_size_z: int) -> numpy.ndarray:
    """
    method can be chosen by using the included options or from the list below with `int`
         0 = user rms ,
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
