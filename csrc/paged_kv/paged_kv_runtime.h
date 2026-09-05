#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#define NANO_CUDA_CHECK(expression)                                           \
  do {                                                                        \
    const cudaError_t error = (expression);                                   \
    TORCH_CHECK(error == cudaSuccess, #expression, " failed with CUDA error ", \
                static_cast<int>(error), ": ", cudaGetErrorString(error));   \
  } while (false)
