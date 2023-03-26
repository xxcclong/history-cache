#include "comm.h"

void p2psend(torch::Tensor tsrc, torch::Tensor tdst, int num_element,
             int target_device) {
  //   auto opt = torch::TensorOptions().device(torch::kCUDA, target_device);
  //   auto output = torch::empty_like(t0, opt);
  cudaSetDevice(target_device);
  cudaMemcpyAsync(tdst.data<float>(), tsrc.data<float>(),
                  num_element * sizeof(float), cudaMemcpyDeviceToDevice);
  //   return tdst;
}

Tensor p2psend_gentensor(Tensor tsrc, int num_element, int target_device) {
  auto opt = torch::TensorOptions().device(torch::kCUDA, target_device);
  auto tdst = torch::empty_like(tsrc, opt);
  cudaSetDevice(target_device);
  cudaMemcpyAsync(tdst.data<float>(), tsrc.data<float>(),
                  num_element * sizeof(float), cudaMemcpyDeviceToDevice);
  return tdst;
}