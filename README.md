

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

## build
requires python environment, `maturin` python package, and rust nightly
```commandline
maturin build -f -r
```

## Commandline usage
Generating the readme images shown above is done using
```commandline
py python/main.py -g
```
run a basic benchmark over the Base image shown above
```commandline
py python/main.py -b
Namespace(test_kinda=False, generate_images=False, benchmark=True, shape=5)
fn_area_contrast     time: 0.397s | window time: 0.332ms
fn_fast_std          time: 0.096s | window time: 0.080ms
fn_fast_std_clamp    time: 0.097s | window time: 0.081ms
fn_stdev_ddof_0      time: 0.216s | window time: 0.181ms
fn_stdev_ddof_1      time: 0.236s | window time: 0.198ms
```
note: the above was run on a Ryzen 5 3600x @ 4.2GHz with 32bg DDR4 @ 3200 MHz 

### Python usage

```python
import numpy # numpy arrays are used to send and receive data from ntw
import ndarray_threaded_window as ntw
import imageio.v3 as iio  # for importing images

# load images
image: numpy.ndarray = iio.imread('img/H5.jpg')

modified_image: numpy.ndarray = ntw.apply_window(
    py_img=image, method=ntw.fn_fast_std,
    win_size_x=5, win_size_y=5, win_size_z=1
)  # currently only supports 3 dimension arrays

iio.imwrite('img/H5-fast_std.jpg', modified_image)
```
