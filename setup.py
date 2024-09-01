import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name), sources=[os.path.join("src", *module.split("."), src) for src in sources]
    )
    return cuda_ext


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name="ray_box_intersect_cuda",
                module="map4d.cuda",
                sources=[
                    "src/ray_box_intersect.cpp",
                    "src/ray_box_intersect_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="video_embedding_cuda",
                module="map4d.cuda",
                sources=[
                    "src/video_embedding.cpp",
                    "src/video_embedding_kernel.cu",
                ],
            ),
        ]
        if not BUILD_NO_CUDA
        else [],
    )
