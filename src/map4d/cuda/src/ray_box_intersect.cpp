/*
Author: Tobias Fischer
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


#define DEVICE_GUARD(_ten) const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void ray_box_intersect_Launcher(int num_rays, int boxes_num, const float *origins, const float *dirs,
            const float *boxes3d, float *local_origins_data, float *local_directions_data, float *near_fars_data, bool *hit_mask_data);


int ray_box_intersect_gpu(at::Tensor origins, at::Tensor dirs, at::Tensor boxes3d, at::Tensor local_origins, at::Tensor local_directions, at::Tensor near_fars, at::Tensor hit_mask){
    // params origins: (N, 3)
    // params dirs: (N, 3)
    // params boxes3d: (N, M, 7)
    // params local_origins: (N, M, 3)
    // params local_directions: (N, M, 3)
    // params near_fars: (N, M, 2)
    // params hit_mask: (N, M)
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    CHECK_INPUT(boxes3d);
    CHECK_INPUT(local_origins);
    CHECK_INPUT(local_directions);
    CHECK_INPUT(near_fars);
    CHECK_INPUT(hit_mask);

    DEVICE_GUARD(origins);

    int num_rays = origins.size(0);
    int boxes_num = boxes3d.size(1);

    const float * origins_data = origins.data<float>();
    const float * dirs_data = dirs.data<float>();
    const float * boxes3d_data = boxes3d.data<float>();
    float * local_origins_data = local_origins.data<float>();
    float * local_directions_data = local_directions.data<float>();
    float * near_fars_data = near_fars.data<float>();
    bool * hit_mask_data = hit_mask.data<bool>();

    ray_box_intersect_Launcher(num_rays, boxes_num, origins_data, dirs_data, boxes3d_data, local_origins_data, local_directions_data, near_fars_data, hit_mask_data);

    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ray_box_intersect_gpu, "ray_box_intersect forward (CUDA)");
}
