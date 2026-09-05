#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

#include <deque>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "paged_kv_runtime.h"

namespace nanovllm::paged_kv {

at::ScalarType parse_dtype(const std::string& name) {
  if (name == "float16") {
    return at::ScalarType::Half;
  }
  if (name == "bfloat16") {
    return at::ScalarType::BFloat16;
  }
  if (name == "float32") {
    return at::ScalarType::Float;
  }
  TORCH_CHECK(false, "PagedKVCache does not support dtype ", name);
}

int64_t checked_numel(const std::vector<int64_t>& sizes) {
  TORCH_CHECK(!sizes.empty(), "PagedKVCache shape must not be empty");
  int64_t numel = 1;
  for (const int64_t size : sizes) {
    TORCH_CHECK(size > 0, "PagedKVCache dimensions must be positive");
    TORCH_CHECK(numel <= std::numeric_limits<int64_t>::max() / size,
                "PagedKVCache shape overflows int64");
    numel *= size;
  }
  return numel;
}

class PagedKVCache : public torch::CustomClassHolder {
 public:
  PagedKVCache(std::vector<int64_t> sizes, std::string dtype_name,
               int64_t device_index)
      : sizes_(std::move(sizes)), device_index_(device_index) {
    TORCH_CHECK(device_index_ >= 0, "CUDA device index must be non-negative");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index_));
    const at::ScalarType dtype = parse_dtype(dtype_name);
    const int64_t numel = checked_numel(sizes_);
    const size_t bytes = static_cast<size_t>(numel) * c10::elementSize(dtype);
    void* pointer = nullptr;
    NANO_CUDA_CHECK(cudaMalloc(&pointer, bytes));
    auto options = at::TensorOptions()
                       .dtype(dtype)
                       .device(at::Device(at::kCUDA, device_index_));
    tensor_ = at::from_blob(
        pointer, sizes_,
        [device_index = device_index_](void* allocation) {
          if (allocation == nullptr) {
            return;
          }
          c10::cuda::CUDAGuard allocation_guard(
              static_cast<c10::DeviceIndex>(device_index));
          cudaFree(allocation);
        },
        options);
    NANO_CUDA_CHECK(
        cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));
    NANO_CUDA_CHECK(
        cudaEventCreateWithFlags(&transfer_done_, cudaEventDisableTiming));
    NANO_CUDA_CHECK(
        cudaEventCreateWithFlags(&reuse_ready_, cudaEventDisableTiming));
  }

  ~PagedKVCache() override {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index_));
    if (transfer_stream_ != nullptr) {
      cudaStreamSynchronize(transfer_stream_);
    }
    for (const auto& pending : pending_host_copies_) {
      cudaEventDestroy(pending.done);
    }
    pending_host_copies_.clear();
    if (transfer_done_ != nullptr) {
      cudaEventDestroy(transfer_done_);
    }
    if (reuse_ready_ != nullptr) {
      cudaEventDestroy(reuse_ready_);
    }
    if (transfer_stream_ != nullptr) {
      cudaStreamDestroy(transfer_stream_);
    }
  }

  at::Tensor tensor() const { return tensor_; }

  int64_t pending_host_copy_count() const {
    return static_cast<int64_t>(pending_host_copies_.size());
  }

  at::Tensor copy_slots_async(const at::Tensor& host_slots) {
    TORCH_CHECK(host_slots.device().is_cpu(),
                "slot mapping source must be a CPU tensor");
    TORCH_CHECK(host_slots.scalar_type() == at::ScalarType::Int,
                "slot mapping source must be int32");
    TORCH_CHECK(host_slots.dim() == 1 && host_slots.is_contiguous(),
                "slot mapping source must be contiguous and one-dimensional");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index_));
    release_completed_host_copies();
    auto options = at::TensorOptions()
                       .dtype(at::ScalarType::Int)
                       .device(at::Device(at::kCUDA, device_index_));
    if (host_slots.numel() == 0) {
      return at::empty({0}, options);
    }
    TORCH_CHECK(host_slots.is_pinned(),
                "slot mapping source must use pinned CPU memory");
    const size_t bytes = static_cast<size_t>(host_slots.numel()) * sizeof(int32_t);
    if (slot_capacity_ < host_slots.numel()) {
      void* pointer = nullptr;
      NANO_CUDA_CHECK(cudaMalloc(&pointer, bytes));
      slot_buffer_ = at::from_blob(
          pointer, {host_slots.numel()},
          [device_index = device_index_](void* allocation) {
            if (allocation == nullptr) {
              return;
            }
            c10::cuda::CUDAGuard allocation_guard(
                static_cast<c10::DeviceIndex>(device_index));
            cudaFree(allocation);
          },
          options);
      slot_capacity_ = host_slots.numel();
    }
    cudaStream_t compute_stream =
        at::cuda::getCurrentCUDAStream(device_index_).stream();
    NANO_CUDA_CHECK(cudaEventRecord(reuse_ready_, compute_stream));
    NANO_CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, reuse_ready_, 0));
    NANO_CUDA_CHECK(cudaMemcpyAsync(slot_buffer_.data_ptr(), host_slots.data_ptr(), bytes,
                                    cudaMemcpyHostToDevice, transfer_stream_));
    cudaEvent_t host_copy_done = nullptr;
    NANO_CUDA_CHECK(
        cudaEventCreateWithFlags(&host_copy_done, cudaEventDisableTiming));
    NANO_CUDA_CHECK(cudaEventRecord(host_copy_done, transfer_stream_));
    pending_host_copies_.push_back({host_slots, host_copy_done});
    NANO_CUDA_CHECK(cudaEventRecord(transfer_done_, transfer_stream_));
    NANO_CUDA_CHECK(cudaStreamWaitEvent(compute_stream, transfer_done_, 0));
    return slot_buffer_.narrow(0, 0, host_slots.numel());
  }

 private:
  struct PendingHostCopy {
    at::Tensor source;
    cudaEvent_t done;
  };

  void release_completed_host_copies() {
    while (!pending_host_copies_.empty()) {
      const cudaError_t status =
          cudaEventQuery(pending_host_copies_.front().done);
      if (status == cudaErrorNotReady) {
        return;
      }
      NANO_CUDA_CHECK(status);
      NANO_CUDA_CHECK(cudaEventDestroy(pending_host_copies_.front().done));
      pending_host_copies_.pop_front();
    }
  }

  std::vector<int64_t> sizes_;
  int64_t device_index_;
  at::Tensor tensor_;
  at::Tensor slot_buffer_;
  int64_t slot_capacity_{0};
  cudaStream_t transfer_stream_{nullptr};
  cudaEvent_t transfer_done_{nullptr};
  cudaEvent_t reuse_ready_{nullptr};
  std::deque<PendingHostCopy> pending_host_copies_;
};

void store_cuda(const at::Tensor& key, const at::Tensor& value,
                at::Tensor key_cache, at::Tensor value_cache,
                const at::Tensor& slot_mapping);
at::Tensor gather_cuda(const at::Tensor& cache, const at::Tensor& block_table,
                       int64_t sequence_length);

}  // namespace nanovllm::paged_kv

static auto paged_kv_cache_class =
    torch::class_<nanovllm::paged_kv::PagedKVCache>("nanovllm_paged_kv",
                                                     "PagedKVCache")
        .def(torch::init<std::vector<int64_t>, std::string, int64_t>())
        .def("tensor", &nanovllm::paged_kv::PagedKVCache::tensor)
        .def("pending_host_copy_count",
             &nanovllm::paged_kv::PagedKVCache::pending_host_copy_count)
        .def("copy_slots_async",
             &nanovllm::paged_kv::PagedKVCache::copy_slots_async);

TORCH_LIBRARY(nanovllm_paged_kv, m) {
  m.def("store(Tensor key, Tensor value, Tensor(a!) key_cache, "
        "Tensor(b!) value_cache, Tensor slot_mapping) -> ()");
  m.def("gather(Tensor cache, Tensor block_table, int sequence_length) -> Tensor");
}
