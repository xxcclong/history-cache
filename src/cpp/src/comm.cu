#include "comm.h"

torch::Tensor p2psend(torch::Tensor t0, int target_device) {
  auto opt = torch::TensorOptions().device(torch::kCUDA, target_device);
  auto output = t0.new_empty({t0.size(0), t0.size(1)}, opt);
  cudaSetDevice(target_device);
  cudaMemcpyAsync(output.data<float>(), t0.data<float>(),
                  t0.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
  return output;
}