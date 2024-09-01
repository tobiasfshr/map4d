#include <torch/extension.h>

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define N_BLOCKS_NEEDED(Q, N_CUDA_THREADS) ((Q - 1) / N_CUDA_THREADS + 1)
#define CUDA_GET_THREAD_ID(tid, Q)                         \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q) return

// Automatically choose number of CUDA threads based on HW CUDA kernel count
int cuda_n_threads = -1;

__host__ int get_sp_cores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2:  // Fermi
            if (devProp.minor == 1)
                cores = mp * 48;
            else
                cores = mp * 32;
            break;
        case 3:  // Kepler
            cores = mp * 192;
            break;
        case 5:  // Maxwell
            cores = mp * 128;
            break;
        case 6:  // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2))
                cores = mp * 128;
            else if (devProp.minor == 0)
                cores = mp * 64;
            break;
        case 7:  // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            break;
        case 8:  // Ampere
            if (devProp.minor == 0)
                cores = mp * 64;
            else if (devProp.minor == 6)
                cores = mp * 128;
            break;
        default:
            break;
    }
    return cores;
}

__host__ void auto_cuda_threads() {
    if (~cuda_n_threads) return;
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    const int n_cores = get_sp_cores(dev_prop);
    // Optimize number of CUDA threads per block
    if (n_cores < 2048) {
        cuda_n_threads = 256;
    }
    if (n_cores < 8192) {
        cuda_n_threads = 512;
    } else {
        cuda_n_threads = 1024;
    }
}

template <typename scalar_t>
__global__ void video_embedding_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> time,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> video_id,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> embedding,
    const int appearance_dim,
	const int num_frequencies,
    size_t num_items
) {
    CUDA_GET_THREAD_ID(idx, num_items);

    const int i = idx / appearance_dim;
	const int j = idx - i * appearance_dim;

    scalar_t result = weights[video_id[i]][j][0] * time[i];
    for (uint32_t log2_frequency = 0; log2_frequency < num_frequencies; log2_frequency++) {
		const scalar_t x = scalbn(time[i], log2_frequency);
		result += weights[video_id[i]][j][log2_frequency * 2 + 1] * sin(x);
		result += weights[video_id[i]][j][log2_frequency * 2 + 2] * cos(x);
	}

    embedding[i][j] = result;
}

template <typename scalar_t>
__global__ void video_embedding_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_loss_embedding,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> time,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> video_id,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_loss_weights,
    const int appearance_dim,
	const int num_frequencies,
    size_t num_items
) {
    CUDA_GET_THREAD_ID(idx, num_items);

    const int i = idx / appearance_dim;
	const int j = idx - i * appearance_dim;

    atomicAdd(&d_loss_weights[video_id[i]][j][0], d_loss_embedding[i][j] * time[i]);
    for (uint32_t log2_frequency = 0; log2_frequency < num_frequencies; log2_frequency++) {
		const scalar_t x = scalbn(time[i], log2_frequency);
        atomicAdd(&d_loss_weights[video_id[i]][j][log2_frequency * 2 + 1], d_loss_embedding[i][j] * sin(x));
		atomicAdd(&d_loss_weights[video_id[i]][j][log2_frequency * 2 + 2], d_loss_embedding[i][j] * cos(x));
	}
}

torch::Tensor video_embedding_forward_cuda(
    torch::Tensor time,
    torch::Tensor video_id,
    torch::Tensor weights,
	const int num_frequencies) {
    auto embedding = torch::empty({time.size(0), weights.size(1)},
        torch::TensorOptions().device(weights.device()).dtype(weights.scalar_type()));

    auto_cuda_threads();
    const int blocks = N_BLOCKS_NEEDED(time.size(0) * weights.size(1), cuda_n_threads);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "video_embedding_forward_cuda", ([&] {
        video_embedding_forward_kernel<<<blocks, cuda_n_threads>>>(
            time.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            video_id.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            embedding.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weights.size(1),
            num_frequencies,
            time.size(0) * weights.size(1));
    }));

    return embedding;
}

torch::Tensor video_embedding_backward_cuda(
    torch::Tensor d_loss_embedding,
    torch::Tensor time,
    torch::Tensor video_id,
	const int num_sequences,
    const int num_frequencies) {
    auto d_loss_weights = torch::zeros({num_sequences, d_loss_embedding.size(1), num_frequencies * 2 + 1}, 
        torch::TensorOptions().device(d_loss_embedding.device()).dtype(d_loss_embedding.scalar_type()));

    auto_cuda_threads();
    const int blocks = N_BLOCKS_NEEDED(time.size(0) * d_loss_embedding.size(1), cuda_n_threads);

    AT_DISPATCH_FLOATING_TYPES(d_loss_embedding.scalar_type(), "video_embedding_backward_cuda", ([&] {
        video_embedding_backward_kernel<<<blocks, cuda_n_threads>>>(
            d_loss_embedding.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            time.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            video_id.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            d_loss_weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            d_loss_embedding.size(1),
            num_frequencies,
            time.size(0) * d_loss_embedding.size(1));
    }));

    return d_loss_weights;
}