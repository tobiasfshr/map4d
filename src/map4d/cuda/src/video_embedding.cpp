#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_LAST_DIM(x, y) TORCH_CHECK(x.size(-1) == y, #x " should have last dim = " #y)

torch::Tensor video_embedding_forward_cuda(
    torch::Tensor time,
    torch::Tensor video_id,
    torch::Tensor weights,
	const int num_frequencies);

torch::Tensor video_embedding_backward_cuda(
    torch::Tensor d_loss_embedding,
    torch::Tensor time,
    torch::Tensor video_id,
	const int num_sequences,
    const int num_frequencies);

torch::Tensor video_embedding_forward(
    torch::Tensor time,
    torch::Tensor video_id,
    torch::Tensor weights,
	const int num_frequencies) {
  CHECK_INPUT(time);
  CHECK_INPUT(video_id);
  CHECK_INPUT(weights);
  CHECK_LAST_DIM(weights, num_frequencies * 2 + 1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(time));
  return video_embedding_forward_cuda(time, video_id, weights, num_frequencies);
}

torch::Tensor video_embedding_backward(
    torch::Tensor d_loss_embedding,
    torch::Tensor time,
    torch::Tensor video_id,
    const int num_sequences,
	const int num_frequencies) {
  CHECK_INPUT(d_loss_embedding);
  CHECK_INPUT(time);
  CHECK_INPUT(video_id);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(time));
  return video_embedding_backward_cuda(d_loss_embedding, time, video_id, num_sequences, num_frequencies);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("video_embedding_forward", &video_embedding_forward);
  m.def("video_embedding_backward", &video_embedding_backward);
}