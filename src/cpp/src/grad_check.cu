#include <assert.h>
#include <c10/cuda/CUDAStream.h>

#include "grad_check.h"

__global__ void record_history_kernel_used_mask(
    float *buffer_new, Index *sub_to_full, Index *history_to_full,
    Index *full_to_history, Index *sub_to_history, bool *used_mask,
    float *buffer_all, Index *header_ptr, float *score, float *threshold,
    Index *checkin_iter, int history_num, int history_size, int num_to_record,
    Index glb_iter) {
  int sub = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (sub >= num_to_record) return;
  int lane = threadIdx.x & 31;
  int header;
  float thres = threshold[0];
  if (sub_to_history[sub] != -1 /* using history */ ||
      (thres != 0 && score[sub] > thres) /*gradient is too large*/ ||
      used_mask[sub] == 0 /*not used*/) {
    return;
  }
  if (lane == 0) {
    // set global header_ptr to next, the current header stay unchanged, waiting
    // to be updated
    header = atomicInc((unsigned int *)header_ptr, history_num - 1);
    int full = history_to_full[header];
    if (full != -1) {  // invalidate the old record, full2history = full->header
      atomicCAS((unsigned long long *)(full_to_history + full),
                (unsigned long long)header, (unsigned long long)-1);
    }
  }
  header = __shfl_sync(0xffffffff, header, 0, 32);
  // move in
  int base_from = sub * history_size;
  int base_to = header * history_size;
  for (int i = lane; i < history_size; i += 32) {
    buffer_all[base_to + i] = buffer_new[base_from + i];
  }
  // set his2full and full2his
  if (lane == 0) {
    Index full = sub_to_full[sub];
    full_to_history[full] = header;
    history_to_full[header] = full;
    checkin_iter[header] = glb_iter;
  }
}

void record_history(Tensor buffer_new, Tensor buffer_all, Tensor used_mask,
                    Tensor sub_to_full, Tensor sub_to_history,
                    Tensor history_to_full, Tensor full_to_history,
                    Tensor header, Tensor score, Tensor thres,
                    Tensor checkin_iter, Index glb_iter) {
  int history_num = buffer_all.sizes()[0];
  int history_size = buffer_all.sizes()[1];
  int num_to_record = buffer_new.sizes()[0];

  int block_size = 512;
  int processed_per_block = block_size / 32;
  record_history_kernel_used_mask<<<(num_to_record + processed_per_block - 1) /
                                        processed_per_block,
                                    block_size>>>(
      buffer_new.data<float>(), sub_to_full.data<Index>(),
      history_to_full.data<Index>(), full_to_history.data<Index>(),
      sub_to_history.data<Index>(), used_mask.data<bool>(),
      buffer_all.data<float>(), header.data<Index>(), score.data<float>(),
      thres.data<float>(), checkin_iter.data<Index>(), history_num,
      history_size, num_to_record, glb_iter);
}

__global__ void count_history_kernel(Index *ptr, Index *idx, Index *hmap,
                                     bool *use_in, bool *use_out,
                                     Index num_node, Index num_seed,
                                     int layer_id, int num_layer) {
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  if (row >= num_node) return;
  // seed nodes
  if (row < num_seed && layer_id == num_layer - 1) {
    if (lane == 0) {
      use_in[row] = 1;
    }
  } else if (use_in[row] != 1)
    return;
  Index begin = ptr[row], end = ptr[row + 1];
  // using histroy
  if (hmap != nullptr && hmap[row] != -1) {
    // use_out[row] = 1; // if with historical embedding, not involved in any
    // computation
    return;
  } else {
    if (lane == 0) {
      use_out[row] = 1;  // due to self loop
    }
  }
#pragma unroll
  for (Index i = begin + lane; i < end; i += 32) {
    Index nodeid = idx[i];
    use_out[nodeid] = 1;
  }
}

std::vector<torch::Tensor> count_history_reconstruct(
    torch::Tensor ptr, torch::Tensor idx, torch::Tensor history_maps,
    Index num_node, Index num_seed, int num_layer) {
  std::vector<torch::Tensor> output_tensors;
  for (int i = 0; i < num_layer + 1; ++i) {
    output_tensors.push_back(
        ptr.new_zeros({num_node}, torch::dtype(torch::kBool)));
  }
  // int num_history_layer = history_maps.size();
  // in/out mask: num_layer + 1; needed ones: 1;
  int block_size = 512;
  int num_per_block = block_size / 32;
  Index ptr_size = ptr.sizes()[0] - 1;
  auto stream =
      at::cuda::getCurrentCUDAStream(ptr.device().index()).stream();
  for (int i = 0; i < num_layer; ++i) {
    count_history_kernel<<<(ptr_size + num_per_block - 1) / num_per_block,
                           block_size, 0, stream>>>(
        ptr.data<Index>(), idx.data<Index>(),
        i != 1  // (num_layer - i > num_history_layer)
            ? nullptr
            : history_maps.data<Index>() /* starting from seed node layer */,
        output_tensors[num_layer - i].data<bool>(),
        output_tensors[num_layer - 1 - i].data<bool>(), ptr_size, num_seed,
        num_layer - 1 - i, num_layer);
  }
  return output_tensors;
}

__global__ void get_graph_structure_score_kernel(Index *ptr, Index *idx,
                                                 Index *output, Index ptr_size,
                                                 Index num_seed) {
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  if (row >= ptr_size) return;
  if (row < num_seed) output[row] = 1;
  if (output[row] == 0) return;
  Index begin = ptr[row], end = ptr[row + 1], num_neighbor = end - begin;
#pragma unroll
  for (Index i = begin + lane; i < end; i += 32) {
    Index nodeid = idx[i];
    if (output[nodeid] == 0) output[nodeid] = output[row] * num_neighbor;
  }
}

torch::Tensor get_graph_structure_score(torch::Tensor ptr, torch::Tensor idx,
                                        Index num_node, Index num_seed,
                                        int num_layer) {
  auto output = ptr.new_zeros({num_node});
  int block_size = 512;
  int num_per_block = block_size / 32;
  Index ptr_size = ptr.sizes()[0] - 1;
  for (int i = 0; i < num_layer; ++i) {
    get_graph_structure_score_kernel<<<
        (ptr_size + num_per_block - 1) / num_per_block, block_size>>>(
        ptr.data<Index>(), idx.data<Index>(), output.data<Index>(), ptr_size,
        num_seed);
  }
  return output;
}