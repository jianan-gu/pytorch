# mypy: allow-untyped-defs
import contextlib
import logging
import re
from typing import List, Optional
from unittest.mock import patch

import sympy

import torch
import torch.utils

from .. import ir
from ..ir import TensorBox
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from ..virtualized import V
from .cpp_template import CppTemplate


log = logging.getLogger(__name__)

FLEX_ATTENTION_TEMPLATE = r"""
{{template.header().getvalue()}}
#include <ATen/native/CPUBlas.h>
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "kv_num_blocks": kv_num_blocks, "kv_indices": kv_indices, "full_kv_num_blocks": full_kv_num_blocks} %}
{%- set kernel_args = template.update_kernel_args(kernel_args) %}

extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output}, extra_sizevars=template.extra_sizevars)}}
{
  int64_t kvBlockSize = {{kvBlockSize}};
  kvBlockSize = kvBlockSize>{{kernel.size(key, 1)}} ? {{kernel.size(key, 1)}}
                                                    : kvBlockSize;
  int64_t num_thread = {{num_thread}};

  // dtypes of kernel and internal buffers
  using scalar_t = {{kernel.dtype(query)}};
  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<{{kernel.dtype(query)}}>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = {{scale}};
  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  int64_t batchSize_k = {{kernel.size(key, 0)}};
  int64_t num_head_k = {{kernel.size(key, 2)}};
  int64_t headSize_v = {{kernel.size(value, 3)}};
  bool is_broadcast_bs_kv = batchSize != batchSize_k;
  bool is_broadcast_head_kv = num_head != num_head_k;
  int64_t gqa_shards = num_head / num_head_k;
  int64_t bs_shards = batchSize / batchSize_k;

  int64_t batchSize_kvi = {{kernel.size(kv_indices, 0)}};
  int64_t num_head_kvi = {{kernel.size(kv_indices, 1)}};
  int64_t block_num_kvi = {{kernel.size(kv_indices, 3)}};
  bool is_broadcast_bs_kvi = batchSize != batchSize_kvi;
  bool is_broadcast_head_kvi = num_head != num_head_kvi;
  int64_t gqa_shards_kvi = num_head / num_head_kvi;
  int64_t bs_shards_kvi = batchSize / batchSize_kvi;
  int64_t kviStrideB = {{kernel.stride(kv_indices, 0)}};
  int64_t kviStrideH = {{kernel.stride(kv_indices, 1)}};
  int64_t kviStrideQ = {{kernel.stride(kv_indices, 2)}};
  auto  kv_indices_data = kv_indices;

  // Strides
  int64_t qStrideB = {{kernel.stride(query, 0)}};
  int64_t qStrideM = {{kernel.stride(query, 1)}};
  int64_t qStrideH = {{kernel.stride(query, 2)}};
  int64_t kStrideB = {{kernel.stride(key, 0)}};
  int64_t kStrideN = {{kernel.stride(key, 1)}};
  int64_t kStrideH = {{kernel.stride(key, 2)}};
  int64_t vStrideB = {{kernel.stride(value, 0)}};
  int64_t vStrideN = {{kernel.stride(value, 1)}};
  int64_t vStrideH = {{kernel.stride(value, 2)}};
  int64_t oStrideB = {{kernel.stride(output, 0)}};
  int64_t oStrideM = {{kernel.stride(output, 2)}};
  int64_t oStrideH = {{kernel.stride(output, 1)}};

  // Check total kv block number for kv value.
  int64_t block_num_kv_count = 0;
  bool has_block_indice_zero = true;
  for (int64_t kv_count = 0; kv_count < block_num_kvi; kv_count++) {
    if (*(kv_indices + kv_count) > 0) {
      block_num_kv_count++;
    } else if (*(kv_indices + kv_count) == 0) {
      if (has_block_indice_zero) {
        has_block_indice_zero = false;
        block_num_kv_count++;
      } else {
        break;
      }
    }
  }
  // Check to use kv_indice if total block size is bigger than kv length, e.g.,
  // in PagedAttention case.
  bool use_kv_indice = false;
  if (block_num_kvi != block_num_kv_count && batchSize_k == 1) {
    use_kv_indice = true;
  }
  int64_t kvSize = use_kv_indice ? block_num_kv_count * kvBlockSize
                                 : {{kernel.size(key, 1)}};
  int64_t qSplitSize = 32;
  int64_t kvSplitSize = 512;
  if (qSize >= 768) {
    qSplitSize = 256;
    kvSplitSize = 512;
  } else if (qSize >= 192) {
    qSplitSize = 64;
    kvSplitSize = 512;
  }
  if (kvBlockSize < kvSplitSize) {
    kvSplitSize = kvBlockSize;
  }

  qSplitSize = qSplitSize > qSize ? qSize : qSplitSize;
  kvSplitSize = kvSplitSize > kvSize ? kvSize : kvSplitSize;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  bool need_pack = false;
  if (std::is_same_v<scalar_t, at::BFloat16>) {
    need_pack = at::native::cpublas::need_pack(at::kBFloat16);
  }
  // Pad is needed for packing when K is not even
  bool headSize_even = headSize % 2 == 0;
  int64_t eheadSize = need_pack && !headSize_even ? headSize + 1: headSize;
  int64_t ekvSplitSize = need_pack && (kvSplitSize % 2 != 0) ? kvSplitSize + 1 : kvSplitSize;
  int64_t ekvTail = need_pack && (kvTail % 2 != 0) ? kvTail + 1 : kvTail;

  // Allocate per thread temp buf (accumulate type)
  int64_t _size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize_v;

{%- set acc_buf_name = "buf" %}
  {{ kernel.define_buffer(acc_buf_name, [ "num_thread", "_size_per_thread" ], dtype = accumulate_dtype) }}
{%- set acc_reduced_buf_name = "buf_reduced" %}
  {{ kernel.define_buffer(acc_reduced_buf_name, [ "num_thread", "qSplitSize", "ekvSplitSize" ], dtype = query_dtype) }}

  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;

  scalar_t* out_data = output;
  accum_t* buf_data = buf;
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced : nullptr;

  // Buffer to store padding query and packing key/value
  scalar_t* key_reorder_ptr = nullptr;
  scalar_t* value_reorder_ptr = nullptr;
  scalar_t* query_padding_ptr = nullptr;
  int64_t kv_padding_size = (kvSize - 1) / kvSplitSize * ekvSplitSize + ekvTail;

{%- set key_t_reorder_buf_name = "key_t_reorder" %}
  {{ kernel.define_buffer(key_t_reorder_buf_name,
     [ "batchSize", "num_head", "eheadSize", "kvSize" ], dtype = query_dtype) }}
  key_reorder_ptr = key_t_reorder;
{%- set value_t_reorder_buf_name = "value_t_reorder" %}
  {{ kernel.define_buffer(value_t_reorder_buf_name,
     [ "batchSize", "num_head", "kv_padding_size", "headSize_v"], dtype = query_dtype) }}
  value_reorder_ptr = value_t_reorder;

  if (!headSize_even && need_pack) {
{%- set qeury_t_padding_buf_name = "qeury_t_padding" %}
    {{ kernel.define_buffer(qeury_t_padding_buf_name, [ "num_thread", "qSplitSize", "eheadSize"], dtype = query_dtype) }}
    query_padding_ptr = qeury_t_padding;
  }
  scalar_t* transpose_buffer_ptr = nullptr;

  // Reorder K, V and transpose K
  if (need_pack) {
{%- set tranpose_t_reorder_buf_name = "tranpose_t_reorder" %}
    {{ kernel.define_buffer(tranpose_t_reorder_buf_name, [ "num_thread", "kvSplitSize", "headSize"], dtype = query_dtype) }}
    transpose_buffer_ptr = tranpose_t_reorder;
  }
  at::parallel_for(0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
    int ompIdx = at::get_thread_num();
    int64_t i = 0, j = 0, l = 0, n = 0;
    scalar_t* transpose_ptr = need_pack? transpose_buffer_ptr + ompIdx * kvSplitSize * headSize : nullptr;
    at::native::data_index_init(begin, i, batchSize, j, num_head, l, kvSlice);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      n = l * kvSplitSize;
      int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
      auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
      auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;
      auto kv_block_num = n / cur_kvSplitSize;
      auto kv_block_offset = n - kv_block_num * cur_kvSplitSize;
      // getting kv indices by [BS, Head, 1, kv_block_num]
      auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
      auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
      auto kv_logical_data = kv_indices_data + i_kvi * kviStrideB +
                              j_kvi * kviStrideH + kv_block_num;
      auto k_addr =
            k_data + i_kv * kStrideB + j_kv * kStrideH + n * kStrideN;
      auto v_addr =
            v_data + i_kv * vStrideB + j_kv * vStrideH + n * vStrideN;
      if (use_kv_indice) {
          k_addr =
              k_data + i_kv * kStrideB + j_kv * kStrideH +
              (*kv_logical_data * cur_kvSplitSize + kv_block_offset) * kStrideN;
          v_addr =
              v_data + i_kv * vStrideB + j_kv * vStrideH +
              (*kv_logical_data * cur_kvSplitSize + kv_block_offset) * vStrideN;
      }
      if (need_pack) {
        // transpose [cur_kvSplitSize, headSize] -> [headSize, cur_kvSplitSize]
        at::native::utils::transpose<uint16_t>(
          cur_kvSplitSize,
          headSize,
          /* src_ptr */
          reinterpret_cast<const uint16_t*>(k_addr),
          /* ld_src */ kStrideN,
          /* dst */ reinterpret_cast<uint16_t*>(transpose_ptr),
          /* ld_dst */ cur_kvSplitSize);

        // Pack [headSize, cur_kvSplitSize]
        pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(transpose_ptr),
          /* dst */ reinterpret_cast<uint16_t*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize),
          /* ld_src */ cur_kvSplitSize,
          /* K */ headSize,
          /* N */ cur_kvSplitSize);

        // Pack [cur_kvSplitSize, headSize_v]
        pack_vnni2(
          /* src */ reinterpret_cast<const uint16_t*>(v_addr),
          /* dst */ reinterpret_cast<uint16_t*>(value_reorder_ptr +
                  i * num_head * kv_padding_size * headSize_v +
                  j * kv_padding_size * headSize_v + n * headSize_v),
          /* ld_src */ vStrideN,
          /* K */ cur_kvSplitSize,
          /* N */ headSize_v);
        } else {
          if (std::is_same_v<scalar_t, at::BFloat16>) {
            // transpose [cur_kvSplitSize, headSize] -> [headSize, cur_kvSplitSize]
            at::native::utils::transpose<uint16_t>(
                  cur_kvSplitSize,
                  headSize,
                  /* src_ptr */
                  reinterpret_cast<const uint16_t*>(k_addr),
                  /* ld_src */ kStrideN,
                  /* dst */ reinterpret_cast<uint16_t*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                          j * eheadSize * kvSize + n * eheadSize),
                  /* ld_dst */ cur_kvSplitSize);
          } else {
            // transpose [cur_kvSplitSize, headSize] -> [headSize, cur_kvSplitSize]
            at::native::utils::transpose<float>(
                  cur_kvSplitSize,
                  headSize,
                  /* src_ptr */
                  reinterpret_cast<const float*>(k_addr),
                  /* ld_src */ kStrideN,
                  /* dst */ reinterpret_cast<float*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                          j * eheadSize * kvSize + n * eheadSize),
                  /* ld_dst */ cur_kvSplitSize);
          }
        }
    // Move to the next query
    at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
    }
  });

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * _size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;

    scalar_t *qk_reduced_data =
        is_reduced_type
            ? buf_reduced_data + ompIdx * qSplitSize * ekvSplitSize
            : nullptr;
    scalar_t* query_t_padding_ptr = (!headSize_even && need_pack)
            ? query_padding_ptr + ompIdx * qSplitSize * eheadSize
            : nullptr;

    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t m = k * qSplitSize;
      int64_t cur_qSplitSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), cur_qSplitSize);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), cur_qSplitSize);

      if (!headSize_even && need_pack) {
        // Pad query if headSize is not even
        copy_value_with_pad<scalar_t>(
          q_data + i * qStrideB + j * qStrideH + m * qStrideM,
          query_t_padding_ptr,
          cur_qSplitSize,
          headSize,
          cur_qSplitSize,
          eheadSize,
          qStrideM
        );
      }
      for (int64_t n = 0; n < kvSize; n += kvSplitSize) {
        int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
        int64_t cur_ekvSplitSize = (need_pack && cur_kvSplitSize % 2 != 0) ? cur_kvSplitSize + 1 : cur_kvSplitSize;

        // Calculate scale * q @ k.T
        auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
        auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;
        auto kv_block_num = n / kvBlockSize;
        auto kv_block_offset = n - kv_block_num * kvBlockSize;
        // getting kv indices by [BS, Head, 1, kv_block_num]
        auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
        auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
        auto kv_logical_data = kv_indices_data + i_kvi * kviStrideB +
                                j_kvi * kviStrideH + kv_block_num;
        if (!need_pack) {
          auto k_addr =
              key_reorder_ptr + i * num_head * eheadSize * kvSize +
              j * eheadSize * kvSize + n * eheadSize;
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              cur_kvSplitSize,
              eheadSize,
              qStrideM,
              cur_kvSplitSize,
              cur_kvSplitSize,
              false,
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              k_addr,
              qk_data);
        } else {
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              cur_kvSplitSize,
              eheadSize,
              headSize_even ? qStrideM : eheadSize,
              cur_kvSplitSize,
              cur_kvSplitSize,
              false,
              !headSize_even
                  ? query_t_padding_ptr
                  : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize,
              qk_data);
        }

        _mul_scale_kernel<accum_t>(qk_data, scaling_factor, cur_qSplitSize*cur_kvSplitSize);

{%- if score_mod and mask_mod %}
        // apply score mod function
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          for (int64_t col = 0; col < cur_kvSplitSize; col++) {
            std::vector<int64_t> b_idx = {i};
            std::vector<int64_t> h_idx = {j};
            std::vector<int64_t> q_idx = {m+row};
            int64_t phisical_kv_idx = n+col;
            if (use_kv_indice) {
                phisical_kv_idx= *kv_logical_data * kvBlockSize + col;
            }
            std::vector<int64_t> kv_idx = {phisical_kv_idx};
            accum_t* in_ptr0 = qk_data + row * cur_kvSplitSize + col;
            auto in_ptr1 = b_idx.data();
            auto in_ptr2 = h_idx.data();
            auto in_ptr3 = q_idx.data();
            auto in_ptr4 = kv_idx.data();
            {{ template.generate_other_buffer("score_others", 0, "len_score_other", kernel.args) }}
            accum_t* out_ptr{{score_buf_idx}} = in_ptr0;
            {{ template.modification(score_mod, score_buf_name, score_buf_idx) }}
          }
        }
        // Apply block mask, fill unused with -inf
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          for (int64_t col = 0; col < cur_kvSplitSize; col++) {
            std::vector<int64_t> b_idx = {i};
            std::vector<int64_t> h_idx = {j};
            std::vector<int64_t> q_idx = {m+row};
            int64_t phisical_kv_idx = n+col;
            if (use_kv_indice) {
                phisical_kv_idx= *kv_logical_data * kvBlockSize + col;
            }
            std::vector<int64_t> kv_idx = {phisical_kv_idx};
            accum_t* qk_block = qk_data + row * cur_kvSplitSize + col;
            auto in_ptr1 = b_idx.data();
            auto in_ptr2 = h_idx.data();
            auto in_ptr3 = q_idx.data();
            auto in_ptr4 = kv_idx.data();
            {{ template.generate_other_buffer("mask_others", -1, "len_mask_other", kernel.args) }}
            std::vector<int64_t> temp = {0};
            int64_t* out_ptr{{mask_buf_idx}} = temp.data();
            {{ template.modification(mask_mod, mask_buf_name, mask_buf_idx) }}
            *qk_block = *out_ptr{{mask_buf_idx}} != 0
                            ? *qk_block
                            : -std::numeric_limits<accum_t>::infinity();
          }
        }
{%- endif %}
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < cur_qSplitSize; ++row) {
          // apply scaling factor and max per row in fusion
          _mul_reduce_max_fusion_kernel(
              qk_data + row * cur_kvSplitSize,
              static_cast<accum_t>(1),
              cur_kvSplitSize,
              qk_data + row * cur_kvSplitSize,
              tmp_max);
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
            // to avoid `nan = exp2f(-inf - (-inf))`
            fill_stub(conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
              static_cast<scalar_t>(0), cur_kvSplitSize);
          } else {
            tmp_sum = tmp_max;
            // qk <- exp(qk - max) and sum per row
            _exp_reduce_sum_fusion_kernel(
              qk_data + row * cur_kvSplitSize, cur_kvSplitSize,
              conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_ekvSplitSize,
              tmp_sum);
            // exp_tmp <- exp(max[row] - max)
            exp_tmp = std::exp(qk_max_data[row] - tmp_max);
            // sum[row] <- sum + exp_tmp * sum[row]
            qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
            // max[row] <- max
            qk_max_data[row] = tmp_max;
            // dst <- dst * exp_tmp
            if (n > 0) {
              at::vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize_v,
              dst_data + row * headSize_v,
              headSize_v);
            }
          }
          if (need_pack && cur_kvSplitSize % 2 != 0) {
            // Pad: [qSplitSize, cur_kvSplitSize] -> [qSplitSize, cur_kvSplitSize + 1]
            *(qk_reduced_data + row * (1 + cur_kvSplitSize) + cur_kvSplitSize) = scalar_t(0);
          }
        }
        // Calculate Softmax(q @ k.T) @ v
        if (!need_pack) {
          auto v_addr =
              v_data + i_kv * vStrideB + j_kv * vStrideH + n * vStrideN;
          if (use_kv_indice) {
              v_addr =
                  v_data + i_kv * vStrideB + j_kv * vStrideH +
                  (*kv_logical_data * kvBlockSize + kv_block_offset) * vStrideN;
          }
          at::native::cpublas::brgemm(
                  cur_qSplitSize,
                  headSize_v,
                  cur_ekvSplitSize,
                  cur_ekvSplitSize,
                  vStrideN,
                  headSize_v,
                  n > 0,
                  conditional_data_ptr(qk_data, qk_reduced_data),
                  v_addr,
                  dst_data);
        } else {
          int64_t psize = n / kvSplitSize * ekvSplitSize;
          at::native::cpublas::brgemm(
              cur_qSplitSize,
              headSize_v,
              cur_ekvSplitSize,
              cur_ekvSplitSize,
              headSize_v,
              headSize_v,
              n > 0,
              qk_reduced_data,
              value_reorder_ptr +
                  i * num_head * kv_padding_size * headSize_v +
                  j * kv_padding_size * headSize_v + psize * headSize_v,
              dst_data);
        }
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < cur_qSplitSize; ++row) {
        // Row sums for full masked out rows are 0, we set them to 1
        // in order to avoid NaNs in the output and instead set fully
        // masked out rows to 0
        qk_max_data[row] = qk_max_data[row] == -std::numeric_limits<accum_t>::infinity() ? 0 : qk_max_data[row];
        qk_sum_data[row] = qk_sum_data[row] == 0 ? 1 : qk_sum_data[row];
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        at::vec::map<scalar_t>(
            [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
            out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
            dst_data + row * headSize_v,
            headSize_v);
      }
      // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
    if (need_pack) {
      at::native::cpublas::brgemm_release();
    }
  });
}
"""


class CppFlexAttentionTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16]
        super().__init__("flex_attention", input_nodes, layout, parallel_num_threads())
        self.scale = scale
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.score_buf_name = (
            V.graph.register_buffer(self.score_mod) if self.score_mod else None
        )
        self.mask_buf_name = (
            V.graph.register_buffer(self.mask_mod) if self.mask_mod else None
        )

        def get_idx(buf_name):
            match = re.search(r"\d+", buf_name)
            assert match, f"incorrect score buf name: {buf_name}"
            return match.group()

        self.score_buf_idx = (
            get_idx(self.score_buf_name) if self.score_buf_name else None
        )
        self.mask_buf_idx = get_idx(self.mask_buf_name) if self.mask_buf_name else None
        self.kv_block_size = kv_block_size
        self.has_other_buffer = has_other_buffer
        self.no_full_kv_block = no_full_kv_block
        self.other_buffer_input_offset = 1
        if self.no_full_kv_block:
            self.other_buffer_input_offset = 0
        self.fake_buffers = fake_buffers
        self.len_score_other = len_score_other
        self.len_mask_other = len_mask_other
        self.kernel_input_name_to_buffer = kernel_input_name_to_buffer
        self.extra_sizevars = {
            val
            for val in self.kernel_input_name_to_buffer.values()
            if isinstance(val, sympy.Symbol)
        }
        self.other_buf_start_idx = 5
        self.score_mod_other_buffers = (
            self.input_nodes[
                self.other_buf_start_idx
                + self.other_buffer_input_offset : self.other_buf_start_idx
                + self.other_buffer_input_offset
                + self.len_score_other
            ]
            if self.has_other_buffer
            else None
        )
        self.mask_mod_other_buffers = (
            self.input_nodes[
                self.other_buf_start_idx
                + self.other_buffer_input_offset
                + self.len_score_other :
            ]
            if self.has_other_buffer
            else None
        )
        self.other_ptr_data = {}  # type: ignore[var-annotated]

    def update_kernel_args(self, kernel_args):
        kernel_args.update(
            {
                key: value
                for key, value in self.kernel_input_name_to_buffer.items()
                if not isinstance(value, sympy.Symbol)
            }
        )
        return kernel_args

    def generate_other_buffer(self, buf_list, start_offset, len_attr, kernel_args):
        kernel_input_name_to_buffer_name = {
            key: value if isinstance(value, sympy.Symbol) else value.get_name()
            for key, value in self.kernel_input_name_to_buffer.items()
        }

        def get_arg(name):
            return kernel_input_name_to_buffer_name.get(name)

        def get_arg_name(name):
            if isinstance(get_arg(name), sympy.Symbol):
                return kernel_args.sizevars.get(get_arg(name))
            return kernel_args.input_buffers.get(get_arg(name))

        if not self.has_other_buffer:
            return ""

        if start_offset == -1:
            start_offset = getattr(self, len_attr)

        length = getattr(self, len_attr)
        for i in range(length):
            pointer = f"in_ptr{self.other_buf_start_idx + start_offset + i}"
            buffer_key = f"{buf_list}_{i}"
            if pointer not in self.other_ptr_data:
                self.other_ptr_data[pointer] = (
                    get_arg_name(buffer_key),
                    get_arg(buffer_key),
                )

        return "\n".join(
            f"auto {ptr} = {name};" for ptr, (name, _) in self.other_ptr_data.items()
        )

    def modification(self, subgraph_buffer, output_name, output_idx):
        assert isinstance(subgraph_buffer, ir.ComputedBuffer)
        subgraph_buffer_data = subgraph_buffer.data
        from ..loop_body import LoopBody
        from ..utils import sympy_index_symbol_with_prefix, SymT
        from ..virtualized import V
        from .cpp import CppKernelProxy, KernelGroup

        kernel_group = KernelGroup()
        kernel_input_args = {
            "score": "in_ptr0",
            "b": "in_ptr1",
            "h": "in_ptr2",
            "q_idx": "in_ptr3",
            "kv_idx": "in_ptr4",
        }
        if self.has_other_buffer:
            kernel_input_args.update(
                {arg: ptr for ptr, (_, arg) in self.other_ptr_data.items()}
            )

        kernel_output_args = {output_name: f"out_ptr{output_idx}"}

        args = kernel_group.args
        for name, inp in kernel_input_args.items():
            args.input_buffers[name] = inp

        for name, inp in kernel_output_args.items():
            args.output_buffers[name] = inp

        for name in self.extra_sizevars:
            args.sizevars[name] = f"k{name}"

        kernel_group.args = args

        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []

        var_sizes = tuple([])  # type: ignore[var-annotated]  # noqa: C409
        output_index = 0
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes)
        }

        def fn(*args):
            V.ops.store(
                output_name,
                output_index,
                subgraph_buffer_data.make_loader()(args).value,
            )

        body = LoopBody(
            fn,
            (list(var_ranges.keys())),
            var_ranges,
            list(var_ranges.keys()),
            tuple(),
        )

        from ..loop_body import MemoryUsageType

        assert all(
            mem.buffer_name in kernel_group.args.input_buffers
            for mem in body.memory_usage[MemoryUsageType.LOAD]
        ), "All the buffers in the score and mask subgraph should be in kernel_group.args.input_buffers"

        bodies.append(body)
        var_sizes_list.append((var_sizes, ()))

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)
        kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppFlexAttentionTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            score_mod=score_mod,
            mask_mod=mask_mod,
            kv_block_size=kv_block_size,
            has_other_buffer=has_other_buffer,
            no_full_kv_block=no_full_kv_block,
            fake_buffers=fake_buffers,
            len_score_other=len_score_other,
            len_mask_other=len_mask_other,
            kernel_input_name_to_buffer=kernel_input_name_to_buffer,
        )
        template.maybe_append_choice(choices)
        return template

    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        if epilogue_nodes is not None and epilogue_nodes != []:
            raise NotImplementedError(
                "Unsupported for `epilogue_nodes` in CppFlexAttentionTemplate."
            )
        # Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
        #     -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
        #  Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        #  Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)

        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])

        num_threads = parallel_num_threads()
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node
        options = dict(
            query=query,
            key=key,
            value=value,
            kv_num_blocks=self.input_nodes[3],
            kv_indices=self.input_nodes[4],
            full_kv_num_blocks=self.input_nodes[5]
            if not self.no_full_kv_block
            else None,
            score_mod_other_buffers=self.score_mod_other_buffers,
            mask_mod_other_buffers=self.mask_mod_other_buffers,
            scale=self.scale,
            accumulate_dtype=torch.float,
            query_dtype=query.layout.dtype,
            kvBlockSize=self.kv_block_size,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
            score_mod=self.score_mod,
            mask_mod=self.mask_mod,
            score_buf_name=self.score_buf_name,
            mask_buf_name=self.mask_buf_name,
            score_buf_idx=self.score_buf_idx,
            mask_buf_idx=self.mask_buf_idx,
        )
        with contextlib.ExitStack() as stack:
            for buf in self.fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(FLEX_ATTENTION_TEMPLATE).render(**options)
