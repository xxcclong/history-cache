#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using torch::Tensor;

Tensor p2psend(Tensor t0, int target_device);