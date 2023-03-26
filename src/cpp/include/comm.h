#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using torch::Tensor;

void p2psend(Tensor tsrc, Tensor tdst, int num_element, int target_device);

Tensor p2psend_gentensor(Tensor tsrc, int num_element, int target_device);