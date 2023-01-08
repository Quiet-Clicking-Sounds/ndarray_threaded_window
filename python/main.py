import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import ndarray_threaded_window as ntw
import numpy as np
from ndarray_threaded_window import base_functions

img_folder_path = Path().resolve()
if img_folder_path.name == "python":
    img_folder_path = img_folder_path.parent
img_folder_path = img_folder_path / "img"
base_image_path = img_folder_path / "H5.jpg"


def load_image(file: Path) -> np.ndarray:
    """get img from disk using cv2"""
    return iio.imread(file)


def export_image(url: Path, data: np.ndarray):
    """
    Export img to disk using cv2.imwrite
    :param url: url to place the file, including extension "here/this.jpg"
    :param data: numpy img containing img data
    """
    url.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(url, data)


def run_checks_against_np_functions():
    def test_stdev_ddof_0(array: np.ndarray) -> int:
        rs = base_functions.stdev_ddof_0(array)
        py = array.std()
        return abs(rs - int(py))

    def test_stdev_ddof_1(array: np.ndarray) -> int:
        rs = base_functions.stdev_ddof_1(array)
        py = array.std(ddof=1)
        return abs(rs - int(py))

    def test_fast_std(array: np.ndarray) -> int:
        rs = base_functions.fast_std(array)
        py = array.std()
        return abs(rs - int(py))

    def test_fast_std_clamp(array: np.ndarray) -> int:
        rs = base_functions.fast_std_clamp(array)
        py = array.std() * 2
        return abs(rs - int(py))

    def test_area_contrast(array: np.ndarray) -> int:
        rs = base_functions.area_contrast(array)
        py = np.sqrt(np.mean(array ** 2))
        return abs(rs - int(py))

    tests = [
        test_stdev_ddof_0,
        test_stdev_ddof_1,
        test_fast_std,
        test_fast_std_clamp,
        test_area_contrast,
    ]

    arrays = [np.random.randint(0, 255, 150, np.uint8) for _ in range(100)]
    for t in tests:
        print(f"{t.__name__} - {sum([t(a) for a in arrays])}")


def run_readme_image_creator(x: int):
    tar1 = img_folder_path / "H5-fast_std.jpg"
    base_img = load_image(base_image_path)
    sh: tuple[int, int, int] = x, x, 1
    out1: np.ndarray = ntw.apply_window(base_img, ntw.fn_fast_std, *sh)
    export_image(tar1, out1)


def run_timers(x: int):
    """
    expect 2023-01-08 - Ryzen 5 3600x 6c/12t 32bg DDR3  3200 MHz
    fn_area_contrast     time: 0.383s | window time: 0.320ms
    fn_fast_std          time: 0.098s | window time: 0.082ms
    fn_fast_std_clamp    time: 0.097s | window time: 0.081ms
    fn_stdev_ddof_0      time: 0.222s | window time: 0.186ms
    fn_stdev_ddof_1      time: 0.216s | window time: 0.181ms
    """
    base_img = load_image(base_image_path)
    sh: tuple[int, int, int] = x, x, 1

    def timer(ft: int):
        t1 = time.time_ns()
        ln = len(ntw.apply_window(base_img, ft, *sh))
        t2 = time.time_ns()
        t = t2 - t1
        t_per = t / ln
        return t, t_per

    test_items = {
        'fn_area_contrast': ntw.fn_area_contrast,
        'fn_fast_std': ntw.fn_fast_std,
        'fn_fast_std_clamp': ntw.fn_fast_std_clamp,
        'fn_stdev_ddof_0': ntw.fn_stdev_ddof_0,
        'fn_stdev_ddof_1': ntw.fn_stdev_ddof_1,
    }
    for s, i in test_items.items():
        timed, per_iter = timer(i)
        print(f"{s: <20} time: {timed / 1e9:.3f}s | window time: {per_iter / 1e6:.3f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test-kinda', action="store_true")
    parser.add_argument('-g', '--generate-images', action="store_true",
                        help="generate the images used in the README, use -sh to modify the shape arg")
    parser.add_argument('-b', '--benchmark', action='store_true',
                        help="run benchmark on the README image, use -sh to modify the shape arg")
    parser.add_argument('-sh', '--shape', default=5, type=int,
                        help="shape of generated image window x > (x, x, 1)")
    args = parser.parse_args()
    print(args)

    if args.test_kinda:
        run_checks_against_np_functions()
    if args.generate_images:
        run_readme_image_creator(args.shape)
    if args.benchmark:
        run_timers(args.shape)
