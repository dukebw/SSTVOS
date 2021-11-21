from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gridattention",
    ext_modules=[
        CUDAExtension(
            "gridattention",
            ["src/lib_cffi.cpp", "src/gridattn.cu"],
            extra_compile_args=["-std=c++11"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
