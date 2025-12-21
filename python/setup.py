%%writefile setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuboid_ops',
    ext_modules=[
        CUDAExtension(
            name='cuboid_ops',  # This becomes the import name: import cuboid_ops
            sources=['cuboid_ops.cu'],
            # Optional: extra flags for better compatibility/performance
            # extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)