"""
run with:
python -mtimeit -s'import examples.speed_tests as st' 'st.use_rs()'
python -mtimeit -s'import examples.speed_tests as st' 'st.use_py()'
"""
import numpy as np

import ndarray_threaded_window as ntw

np.random.seed(19680801)
window_len = 50
array = array = np.random.randint(0, 255, (10000000, window_len), np.uint8)


def use_rs():
    ntw.apply_window(array, ntw.func_fast_population_std, [1, window_len])


def use_py():
    np.std(array, axis=1, dtype=np.float64)


if __name__ == '__main__':
    print("this should not be run directly")
