# 用于部分代码编译成二进制文件.so
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

# 定义需要编译的模块
extensions = [
    Extension("utils.normal_loss", ["utils/normal_loss.py"]),
    Extension("utils.slam_backend", ["utils/slam_backend.py"]),
]

setup(
    ext_modules = cythonize(extensions, language_level=3)
)