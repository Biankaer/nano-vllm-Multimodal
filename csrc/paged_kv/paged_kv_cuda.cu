#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

#include "paged_kv_runtime.h"

namespace nanovllm::paged_kv {

template <typename scalar_t>
__device__ bool is_vector_aligned(const scalar_t* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % sizeof(uint4) == 0;
}

template <typename scalar_t>
__device__ bool is_vector_aligned(scalar_t* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % sizeof(uint4) == 0;
}

template <typename scalar_t>
__global__ void store_kernel(const scalar_t* key, const scalar_t* value,
                             scalar_t* key_cache, scalar_t* value_cache,
                             const int32_t* slot_mapping,
                             int64_t key_token_stride,
                             int64_t value_token_stride, int64_t d) {
  const int64_t token = blockIdx.x;
  const int32_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int64_t output_base = static_cast<int64_t>(slot) * d;
  const scalar_t* key_source = key + token * key_token_stride;
  const scalar_t* value_source = value + token * value_token_stride;
  scalar_t* key_destination = key_cache + output_base;
  scalar_t* value_destination = value_cache + output_base;
  constexpr int64_t elements_per_vector = sizeof(uint4) / sizeof(scalar_t);
  int64_t scalar_start = 0;
  if (is_vector_aligned(key_source) && is_vector_aligned(value_source) &&
      is_vector_aligned(key_destination) &&
      is_vector_aligned(value_destination)) {
    const int64_t vector_count = d / elements_per_vector;
    const uint4* key_vectors = reinterpret_cast<const uint4*>(key_source);
    const uint4* value_vectors = reinterpret_cast<const uint4*>(value_source);
    uint4* key_cache_vectors = reinterpret_cast<uint4*>(key_destination);
    uint4* value_cache_vectors = reinterpret_cast<uint4*>(value_destination);
    for (int64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x) {
      key_cache_vectors[vector] = key_vectors[vector];
      value_cache_vectors[vector] = value_vectors[vector];
    }
    scalar_start = vector_count * elements_per_vector;
  }
  for (int64_t offset = scalar_start + threadIdx.x; offset < d;
       offset += blockDim.x) {
    key_destination[offset] = key_source[offset];
    value_destination[offset] = value_source[offset];
  }
}

template <typename scalar_t>
__global__ void gather_kernel(const scalar_t* cache,
                              const int32_t* block_table, scalar_t* output,
                              int64_t sequence_length, int64_t block_size,
                              int64_t d) {
  const int64_t token = blockIdx.x;
  if (token >= sequence_length) {
    return;
  }
  const int64_t logical_block = token / block_size;
  const int64_t block_offset = token % block_size;
  const int32_t physical_block = block_table[logical_block];
  const int64_t source_base =
      (static_cast<int64_t>(physical_block) * block_size + block_offset) * d;
  const int64_t output_base = token * d;
  const scalar_t* source = cache + source_base;
  scalar_t* destination = output + output_base;
  constexpr int64_t elements_per_vector = sizeof(uint4) / sizeof(scalar_t);
  int64_t scalar_start = 0;
  if (is_vector_aligned(source) && is_vector_aligned(destination)) {
    const int64_t vector_count = d / elements_per_vector;
    const uint4* source_vectors = reinterpret_cast<const uint4*>(source);
    uint4* destination_vectors = reinterpret_cast<uint4*>(destination);
    for (int64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x) {
      destination_vectors[vector] = source_vectors[vector];
    }
    scalar_start = vector_count * elements_per_vector;
  }
  for (int64_t offset = scalar_start + threadIdx.x; offset < d;
       offset += blockDim.x) {
    output[output_base + offset] = cache[source_base + offset];
  }
}

void store_cuda(const at::Tensor& key, const at::Tensor& value,
                at::Tensor key_cache, at::Tensor value_cache,
                const at::Tensor& slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && slot_mapping.is_cuda(),
              "Paged KV CUDA Store requires CUDA tensors");
  TORCH_CHECK(key.scalar_type() == value.scalar_type() &&
                  key.scalar_type() == key_cache.scalar_type() &&
                  key.scalar_type() == value_cache.scalar_type(),
              "Paged KV CUDA Store dtype mismatch");
  const int64_t tokens = key.size(0);
  if (tokens == 0) {
    return;
  }
  int64_t d = key.size(1) * key.size(2);
  int64_t key_token_stride = key.stride(0);
  int64_t value_token_stride = value.stride(0);
  const int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(),
      "paged_kv_store_cuda", [&] {
        const scalar_t* key_pointer = key.data_ptr<scalar_t>();
        const scalar_t* value_pointer = value.data_ptr<scalar_t>();
        scalar_t* key_cache_pointer = key_cache.data_ptr<scalar_t>();
        scalar_t* value_cache_pointer = value_cache.data_ptr<scalar_t>();
        const int32_t* slots_pointer = slot_mapping.data_ptr<int32_t>();
        void* arguments[] = {&key_pointer, &value_pointer, &key_cache_pointer,
                             &value_cache_pointer, &slots_pointer,
                             &key_token_stride, &value_token_stride, &d};
        NANO_CUDA_CHECK(cudaLaunchKernel(
            reinterpret_cast<const void*>(&store_kernel<scalar_t>),
            dim3(static_cast<unsigned int>(tokens)), dim3(threads), arguments, 0,
            stream));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor gather_cuda(const at::Tensor& cache, const at::Tensor& block_table,
                       int64_t sequence_length) {
  TORCH_CHECK(cache.is_cuda() && block_table.is_cuda(),
              "Paged KV CUDA Gather requires CUDA tensors");
  int64_t block_size = cache.size(1);
  const int64_t heads = cache.size(2);
  const int64_t head_dim = cache.size(3);
  int64_t d = heads * head_dim;
  at::Tensor output = at::empty({sequence_length, heads, head_dim},
                                cache.options());
  if (sequence_length == 0) {
    return output;
  }
  const int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, cache.scalar_type(),
      "paged_kv_gather_cuda", [&] {
        const scalar_t* cache_pointer = cache.data_ptr<scalar_t>();
        const int32_t* table_pointer = block_table.data_ptr<int32_t>();
        scalar_t* output_pointer = output.data_ptr<scalar_t>();
        void* arguments[] = {&cache_pointer, &table_pointer, &output_pointer,
                             &sequence_length, &block_size, &d};
        NANO_CUDA_CHECK(cudaLaunchKernel(
            reinterpret_cast<const void*>(&gather_kernel<scalar_t>),
            dim3(static_cast<unsigned int>(sequence_length)), dim3(threads),
            arguments, 0, stream));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace nanovllm::paged_kv

TORCH_LIBRARY_IMPL(nanovllm_paged_kv, CUDA, m) {
  m.impl("store", TORCH_FN(nanovllm::paged_kv::store_cuda));
  m.impl("gather", TORCH_FN(nanovllm::paged_kv::gather_cuda));
}
