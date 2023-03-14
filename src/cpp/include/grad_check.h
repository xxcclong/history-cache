#ifndef GRAD_CHECK_H
#define GRAD_CHECK_H
#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <vector>

using Index = int64_t;
using torch::Tensor;

void record_history(Tensor buffer_new, Tensor buffer_all, Tensor used_mask,
                    Tensor sub_to_full, Tensor sub_to_history,
                    Tensor history_to_full, Tensor full_to_history,
                    Tensor header, Tensor score, Tensor thres,
                    Tensor checkin_iter, Index glb_iter);

std::vector<torch::Tensor> count_history_reconstruct(
    torch::Tensor ptr, torch::Tensor idx, torch::Tensor history_maps,
    Index num_node, Index num_seed, int num_layer);

torch::Tensor get_graph_structure_score(torch::Tensor ptr, torch::Tensor idx,
                                        Index num_node, Index num_seed,
                                        int num_layer);

torch::Tensor count_num(torch::Tensor arr, Index mmax);

#endif