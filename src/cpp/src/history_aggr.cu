#include <c10/cuda/CUDAStream.h>

#include "common.h"
#include "history_aggr.h"

// if a center node has history, no neighbor searching, use (copy) history value
__global__ void gen_fwd_history_v2(Index *ptr, Index *idx, float *vin,
                                   Index *hmap, float *hbuf, float *vout,
                                   int num_node, int INFEATURE, int hsize) {
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float rs = 0.0f;
  // float local = vin[row * INFEATURE + col];
  int theidx;
  int jlimit;
  Index hid = hmap[row];
  if (hid != -1) {
    if (col < INFEATURE) vout[row * INFEATURE + col] = hbuf[row * hsize + col];
    return;
  }
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (col < INFEATURE) rs += vin[neighbor_id * INFEATURE + col];
    }
  }
  if (col < INFEATURE && num_neighbor != 0)
    vout[row * INFEATURE + col] = rs / num_neighbor;
}

__global__ void gen_fwd_history_v2_edge_value(Index *ptr, Index *idx,
                                              float *edge_value, float *vin,
                                              Index *hmap, float *hbuf,
                                              float *vout, int num_node,
                                              int INFEATURE, int hsize) {
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  // float local = vin[row * INFEATURE + col];
  int theidx;
  float theval;
  int jlimit;
  Index hid = hmap[row];
  if (hid != -1) {
    if (col < INFEATURE) vout[row * INFEATURE + col] = hbuf[row * hsize + col];
    return;
  }
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) rs += vin[neighbor_id * INFEATURE + col] * val;
    }
  }
  if (col < INFEATURE) vout[row * INFEATURE + col] = rs;
}

__global__ void gen_fwd_history_v2_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *vin, Index *hmap,
    float *hbuf, float *vout, int num_node, int INFEATURE, int hsize,
    int num_head) {
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;  // center node id
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
  Index hid = hmap[row];
  if (hid != -1) {
    if (hsize == INFEATURE) {
      if (col < INFEATURE)
        vout[target_id * INFEATURE + col] = hbuf[row * hsize + col];
    } else if (hsize == INFEATURE * num_head) {
      if (col < INFEATURE)
        vout[target_id * INFEATURE + col] =
            hbuf[row * hsize + head_id * INFEATURE + col];
    } else {
      assert(false);
    }
    return;
  }
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE)
        rs += vin[(neighbor_id * num_head + head_id) * INFEATURE + col] * val;
    }
  }
  if (col < INFEATURE) vout[target_id * INFEATURE + col] = rs;
}

__global__ void gen_bwd_history_grad_v2(
    Index *ptr, Index *idx, float *grads_in, Index *hmap, float *vout_fwd,
    float *grads_out, float *grads_history, int num_node, int INFEATURE,
    int hsize)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float grad = 0.0f;
  if (col < INFEATURE && num_neighbor > 0) {
    grad = grads_in[row * INFEATURE + col] / num_neighbor;
  }

  Index hid = hmap[row];
  if (hid != -1) {
    grad = grads_in[row * INFEATURE + col];  // no div in forward history
    if (col < INFEATURE) atomicAdd(grads_history + row * hsize + col, grad);
    return;
  }

  int theidx;
  int jlimit;

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id * INFEATURE + col, grad);
      }
    }
  }
}

__global__ void gen_bwd_history_grad_v2_edge_value(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, Index *hmap,
    float *vout_fwd, float *grads_out, float *grads_history, int num_node,
    int INFEATURE, int hsize)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }

  Index hid = hmap[row];
  if (hid != -1) {
    grad = grads_in[row * INFEATURE + col];  // no div in forward history
    if (col < INFEATURE) atomicAdd(grads_history + row * hsize + col, grad);
    return;
  }

  int theidx;
  float theval;
  int jlimit;

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id * INFEATURE + col, grad * val);
      }
    }
  }
}

__global__ void gen_bwd_history_grad_v2_edge_value_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, Index *hmap,
    float *vout_fwd, float *grads_out, float *grads_history, int num_node,
    int INFEATURE, int hsize,
    float *edge_grad)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row =
      (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // center node id
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }

  Index hid = hmap[row];
  if (hid != -1) {
    grad = grads_in[row * INFEATURE + col];  // no div in forward history
    // grad * 1e6);
    if (col < INFEATURE) atomicAdd(grads_history + row * hsize + col, grad);
    return;
  }

  int theidx;
  float theval;
  int jlimit;

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      float tmp_edge_grad = 0.0f;
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id * INFEATURE + col, grad * val);
        tmp_edge_grad = grad * vout_fwd[neighbor_id * INFEATURE + col];
        // atomicAdd(edge_grad + i + j, grad * vout_fwd[neighbor_id * INFEATURE
        // + col]);
      }
      for (int k = 16; k > 0; k >>= 1) {
        tmp_edge_grad += __shfl_down_sync(0xffffffff, tmp_edge_grad, k);  // sum
      }
      if (lane == 0) {
        atomicAdd(edge_grad + i + j, tmp_edge_grad);
      }
    }
  }
}

__global__ void gen_bwd_history_grad_v2_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, Index *hmap,
    float *vout_fwd, float *grads_out, float *grads_history, int num_node,
    int INFEATURE, int hsize,
    int num_head)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[target_id * INFEATURE + col];
  }

  Index hid = hmap[row];
  if (hid != -1) {
    grad = grads_in[target_id * INFEATURE + col];
    if (hsize == INFEATURE) {
      if (col < INFEATURE) atomicAdd(grads_history + row * hsize + col, grad);
    } else if (hsize == INFEATURE * num_head) {
      if (col < INFEATURE)
        atomicAdd(grads_history + row * hsize + head_id * INFEATURE + col,
                  grad);
    } else {
      assert(false);
    }
    return;
  }

  int theidx;
  float theval;
  int jlimit;

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        atomicAdd(
            grads_out + (neighbor_id * num_head + head_id) * INFEATURE + col,
            grad * val);
      }
    }
  }
}

__global__ void gen_bwd_history_grad_v2_edge_value_multi_head_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, Index *hmap,
    float *vout_fwd, float *grads_out, float *grads_history, int num_node,
    int INFEATURE, int hsize, int num_head,
    float *edge_grad)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[target_id * INFEATURE + col];
  }

  Index hid = hmap[row];
  if (hid != -1) {
    grad = grads_in[target_id * INFEATURE + col];
    if (hsize == INFEATURE) {
      if (col < INFEATURE) atomicAdd(grads_history + row * hsize + col, grad);
    } else if (hsize == INFEATURE * num_head) {
      if (col < INFEATURE)
        atomicAdd(grads_history + row * hsize + head_id * INFEATURE + col,
                  grad);
    } else {
      assert(false);
    }
    return;
  }

  int theidx;
  float theval;
  int jlimit;

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      float tmp_edge_grad = 0.f;
      if (col < INFEATURE) {
        atomicAdd(
            grads_out + (neighbor_id * num_head + head_id) * INFEATURE + col,
            grad * val);
        tmp_edge_grad =
            grad *
            vout_fwd[(neighbor_id * num_head + head_id) * INFEATURE + col];
        // atomicAdd(edge_grad + (i + j) * num_head + head_id,
        //           grad * vout_fwd[(neighbor_id * num_head + head_id) *
        //           INFEATURE + col]);
      }
      for (int k = 16; k > 0; k >>= 1) {
        tmp_edge_grad += __shfl_down_sync(0xffffffff, tmp_edge_grad, k);  // sum
      }
      if (lane == 0) {
        atomicAdd(edge_grad + (i + j) * num_head + head_id, tmp_edge_grad);
      }
    }
  }
}

torch::Tensor historyForwardImpl(AutogradContext *ctx, torch::Tensor input,
                                 torch::Tensor ptr, torch::Tensor idx,
                                 torch::Tensor history_map,
                                 torch::Tensor history_buffer, int history_size,
                                 Index num_node,
                                 torch::Tensor edge_value = torch::Tensor()) {
  ctx->save_for_backward({input, history_buffer, edge_value});
  ctx->saved_data["ptr"] = (int64_t)ptr.data<Index>();
  ctx->saved_data["idx"] = (int64_t)idx.data<Index>();
  ctx->saved_data["hmap"] = (int64_t)history_map.data<Index>();
  ctx->saved_data["hsize"] = (int64_t)history_size;
  if (num_node == 0) num_node = ptr.sizes()[0] - 1;
  ctx->saved_data["num_node"] = (int64_t)num_node;

  auto stream = at::cuda::getCurrentCUDAStream(ptr.device().index()).stream();

  int feat_len = input.sizes().back();
  ASSERT(input.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(input.device().index()));
  int num_head = 1;
  auto output = torch::Tensor();
  if (edge_value.sizes()[0] != 0 && edge_value.sizes().size() > 1) {
    num_head = edge_value.sizes()[1];
    ASSERT(num_head == input.sizes()[1]);
    output = input.new_zeros({num_node, num_head, feat_len});
  } else {
    output = input.new_zeros({num_node, feat_len});
  }
  output.requires_grad_(true);

  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  int block_size = 512;
  block_size = std::max(block_size, ceil_feat_len);

  dim3 grid, block;
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  // Evaluating doesn't use history
  if (edge_value.sizes()[0] == 0) {
    if (history_size == 0) {
      ASSERT(false);
    } else
      gen_fwd_history_v2<<<grid, block, 0, stream>>>(
          ptr.data<Index>(), idx.data<Index>(), input.data<float>(),
          history_map.data<Index>(), history_buffer.data<float>(),
          output.data<float>(), num_node, feat_len, history_size);
  } else if (edge_value.sizes().size() == 1) {
    if (history_size == 0) {
      ASSERT(false);
    } else {
      gen_fwd_history_v2_edge_value<<<grid, block, 0, stream>>>(
          ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
          input.data<float>(), history_map.data<Index>(),
          history_buffer.data<float>(), output.data<float>(), num_node,
          feat_len, history_size);
    }
  } else if (edge_value.sizes().size() == 2) {
    if (history_size == 0) {
      ASSERT(false);
    } else {
      grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
               (block_size / ceil_feat_len);
      gen_fwd_history_v2_edge_value_multi_head<<<grid, block, 0, stream>>>(
          ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
          input.data<float>(), history_map.data<Index>(),
          history_buffer.data<float>(), output.data<float>(), num_node,
          feat_len, history_size, num_head);
    }
  } else {
    ASSERT(false);
  }
  return output;
}

torch::Tensor AggrHistoryFunction::forward(AutogradContext *ctx,
                                           torch::Tensor input,
                                           torch::Tensor ptr, torch::Tensor idx,
                                           torch::Tensor history_map,
                                           torch::Tensor history_buffer,
                                           int history_size, Index num_node) {
  return historyForwardImpl(ctx, input, ptr, idx, history_map, history_buffer,
                            history_size, num_node);
}

torch::Tensor AggrHistoryEdgeValueFunction::forward(
    AutogradContext *ctx, torch::Tensor input, torch::Tensor ptr,
    torch::Tensor idx, torch::Tensor edge_value, torch::Tensor history_map,
    torch::Tensor history_buffer, int history_size, Index num_node) {
  return historyForwardImpl(ctx, input, ptr, idx, history_map, history_buffer,
                            history_size, num_node, edge_value);
}

tensor_list historyBackwardImpl(AutogradContext *ctx,
                                tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto edge_value = saved[2];
  Index *ptr = (Index *)(ctx->saved_data["ptr"].toInt());
  Index *idx = (Index *)(ctx->saved_data["idx"].toInt());
  Index *history_map = (Index *)(ctx->saved_data["hmap"].toInt());
  int history_size = ctx->saved_data["hsize"].toInt();
  auto grad_output = grad_outputs[0];
  auto grad_input = torch::zeros_like(input);
  auto grad_history = torch::zeros_like(saved[1]);

  auto stream =
      at::cuda::getCurrentCUDAStream(grad_input.device().index()).stream();

  int num_node = ctx->saved_data["num_node"].toInt();
  int hsize = ctx->saved_data["hsize"].toInt();
  int feat_len = input.sizes().back();

  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  int block_size = 512;
  block_size = std::max(block_size, ceil_feat_len);

  dim3 grid, block;
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  ASSERT(block.x % 32 == 0);
  if (edge_value.sizes()[0] == 0) {
    if (history_size == 0) {
      ASSERT(false);
    } else {
      gen_bwd_history_grad_v2<<<grid, block, 0, stream>>>(
          ptr, idx, grad_output.data<float>(), history_map, input.data<float>(),
          grad_input.data<float>(), grad_history.data<float>(), num_node,
          feat_len, hsize);
    }
  } else if (edge_value.sizes().size() == 1) {
    if (history_size == 0) {
      ASSERT(false);
    } else {
      if (edge_value.requires_grad()) {
        auto edge_grad = torch::zeros_like(edge_value);
        gen_bwd_history_grad_v2_edge_value_edge_grad<<<grid, block, 0,
                                                       stream>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            history_map, input.data<float>(), grad_input.data<float>(),
            grad_history.data<float>(), num_node, feat_len, hsize,
            edge_grad.data<float>());
        return {grad_input, grad_history, edge_grad};
      } else {
        gen_bwd_history_grad_v2_edge_value<<<grid, block, 0, stream>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            history_map, input.data<float>(), grad_input.data<float>(),
            grad_history.data<float>(), num_node, feat_len, hsize);
      }
    }
  } else if (edge_value.sizes().size() == 2) {
    int num_head = edge_value.sizes()[1];
    if (history_size == 0) {
      ASSERT(false);
    } else {
      grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
               (block_size / ceil_feat_len);
      if (edge_value.requires_grad()) {
        auto edge_grad = torch::zeros_like(edge_value);
        gen_bwd_history_grad_v2_edge_value_multi_head_edge_grad<<<grid, block,
                                                                  0, stream>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            history_map, input.data<float>(), grad_input.data<float>(),
            grad_history.data<float>(), num_node, feat_len, hsize, num_head,
            edge_grad.data<float>());
        return {grad_input, grad_history, edge_grad};
      } else {
        gen_bwd_history_grad_v2_edge_value_multi_head<<<grid, block, 0,
                                                        stream>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            history_map, input.data<float>(), grad_input.data<float>(),
            grad_history.data<float>(), num_node, feat_len, hsize, num_head);
      }
    }
  } else {
    ASSERT(false);
  }
  return {grad_input, grad_history};
}

tensor_list AggrHistoryFunction::backward(AutogradContext *ctx,
                                          tensor_list grad_outputs) {
  auto outputs = historyBackwardImpl(ctx, grad_outputs);
  return {outputs[0], torch::Tensor(), torch::Tensor(), torch::Tensor(),
          outputs[1], torch::Tensor(), torch::Tensor()};
}

tensor_list AggrHistoryEdgeValueFunction::backward(AutogradContext *ctx,
                                                   tensor_list grad_outputs) {
  auto outputs = historyBackwardImpl(ctx, grad_outputs);
  return {outputs[0],      torch::Tensor(),
          torch::Tensor(), outputs.size() == 3 ? outputs[2] : torch::Tensor(),
          torch::Tensor(), outputs[1],
          torch::Tensor(), torch::Tensor()};
}

torch::Tensor aggr_forward_history(torch::Tensor input, torch::Tensor ptr,
                                   torch::Tensor idx, torch::Tensor history_map,
                                   torch::Tensor history_buffer,
                                   int history_size, Index num_node) {
  auto t = AggrHistoryFunction::apply(input, ptr, idx, history_map,
                                      history_buffer, history_size, num_node);
  t.requires_grad_(true);  // IMPORTANT
  return t;
}

torch::Tensor aggr_forward_history_edge_value(
    torch::Tensor input, torch::Tensor ptr, torch::Tensor idx,
    torch::Tensor edge_value, torch::Tensor history_map,
    torch::Tensor history_buffer, int history_size, Index num_node) {
  auto t = AggrHistoryEdgeValueFunction::apply(input, ptr, idx, edge_value,
                                               history_map, history_buffer,
                                               history_size, num_node);
  t.requires_grad_(true);  // IMPORTANT
  return t;
}