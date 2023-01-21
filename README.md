

<table>
<tr><td colspan="2"></td> </tr>

<tr>
<td>Base image</td>
<td>output of: <code>main.py -g</code></td>
</tr>
<tr>
<td><img src="img/H5.jpg"></td>
<td><img src="img/H5-fast_std.jpg"></td>
</tr>

</table>

### Python usage

```python
import numpy # numpy arrays are used to send and receive data from ntw
import ndarray_threaded_window as ntw
import imageio.v3 as iio  # for importing images

# load images
image: numpy.ndarray = iio.imread('img/H5.jpg')

modified_image: numpy.ndarray = ntw.apply_window(image, ntw.fn_fast_std, [5,5,1])

iio.imwrite('img/H5-fast_std.jpg', modified_image)
```

### Rounding note: 
rounding does differ between the numpy implementation of `int(np.std)` and `ntw.func_fast_population_std`

## Build
requires python environment, `maturin` python package, and rust nightly
```commandline
maturin build -f -r
```

Building for use in rust only:
```commandline
cargo build --no-default-features
```


## /examples/main.py Commandline example
Generating the readme images shown above is done using
```commandline
py python/main.py -g
```
run a basic benchmark over the Base image shown above

```
(venv) PS D:\Rust\ndarray_threaded_window> py examples/main.py -b
Namespace(test_kinda=False, generate_images=False, benchmark=True, shape=5, run_fast_std_montecarlo_test=False)
Environment Variable "SET_THREADS" set to: "3"
runningrun_timers with shape: [5, 5, 1] over array shape: (1200, 1200, 3)
func_window_max           time: 0.194s | window time: 0.162ms
func_window_min           time: 0.212s | window time: 0.177ms
func_stdev_ddof_0         time: 0.756s | window time: 0.632ms
func_stdev_ddof_1         time: 0.789s | window time: 0.660ms
func_area_contrast        time: 1.123s | window time: 0.939ms
func_fast_std             time: 0.413s | window time: 0.346ms
func_fast_std_clamp       time: 0.354s | window time: 0.296ms
func_fast_population_std  time: 0.370s | window time: 0.310ms
func_fast_sample_std      time: 0.378s | window time: 0.316ms

```
the above was run on a Ryzen 5 3600x @ 4.2GHz with 32bg DDR4 @ 3200 MHz 



### Python Installation
See [releases](https://github.com/Quiet-Clicking-Sounds/ndarray_threaded_window/releases)
