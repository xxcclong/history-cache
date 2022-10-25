#pragma once

#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

using Index = int64_t;
using torch::Tensor;

using namespace torch::autograd;

class AggrHistoryFunction : public Function<AggrHistoryFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               torch::Tensor history_map,
                               torch::Tensor history_buffer, int history_size,
                               Index num_node);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class AggrHistoryEdgeValueFunction
    : public Function<AggrHistoryEdgeValueFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               torch::Tensor edge_value,
                               torch::Tensor history_map,
                               torch::Tensor history_buffer, int history_size,
                               Index num_node);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

torch::Tensor aggr_forward_history(torch::Tensor input, torch::Tensor ptr,
                                   torch::Tensor idx, torch::Tensor history_map,
                                   torch::Tensor history_buffer,
                                   int history_size, Index num_node);

torch::Tensor aggr_forward_history_edge_value(
    torch::Tensor input, torch::Tensor ptr, torch::Tensor idx,
    torch::Tensor edge_value, torch::Tensor history_map,
    torch::Tensor history_buffer, int history_size, Index num_node);