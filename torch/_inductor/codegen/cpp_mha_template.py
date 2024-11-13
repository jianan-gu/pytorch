# mypy: allow-untyped-defs
import contextlib
import logging
import math
from functools import lru_cache
from typing import Any, Callable, cast, List, Optional, Set, Union
from unittest.mock import patch

import torch
import torch.utils

from ..._dynamo.utils import counters
from .. import config, ir, lowering as L
from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import (
    cache_on_self,
    has_free_symbols,
    is_same_mkldnn_tensor,
    is_same_tensor,
    parallel_num_threads,
)
from ..virtualized import ops, V
from .cpp import get_export_declaration
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm, LayoutType
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel, parse_expr_with_index_symbols
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
#include <ATen/native/CPUBlas.h>

{%- set kernel_args = {"query": query, "key": key, "value": value} %}
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{

  // scale
  using scalar_t = {{kernel.dtype(query)}};
  auto is_causal = false;
  int64_t q_split_size = 32;
  int64_t kv_split_size = 512 ;
  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<{{kernel.dtype(query)}}>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = {{scale}};

  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t kvSize = {{kernel.size(key, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};

  bool has_attn_mask = false;
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
  int64_t oStrideM = {{kernel.stride(output, 1)}};
  int64_t oStrideH = {{kernel.stride(output, 2)}};
  int64_t mStrideB = 0;
  int64_t mStrideH = 0;
  int64_t mStrideM = 0;
  int64_t mStrideN = 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;


 // const auto dtype = query.layout.dtype;
 // const auto accumulate_dtype = torch.float32; //{{kernel.acc_dtype(query)}}; //toOpMathType(dtype);

  // Whether pack is needed
  bool need_pack = false;
  // Block size of packing B matrix
  int64_t packb_size = 64;

  int64_t rHeadSize = headSize;
  int64_t rkvSplitSize = kvSplitSize;
  int64_t rkvTail = kvTail;
  int64_t rkvSize = kv_split_size > kvSize ? rkvTail : rkvSplitSize * kvSlice + rkvTail;

  // oneDNN pack does not support odd K now, we need also pad odd K
  bool headSize_even = headSize % 2 == 0;
  int64_t eheadSize = headSize;
  int64_t ekvSplitSize = kvSplitSize;
  int64_t ekvTail = kvTail;

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread = qSplitSize * rkvSplitSize + qSplitSize + qSplitSize + qSplitSize * rHeadSize;

  {%- set acc_buf_name = "buf" %}
      {{ kernel.define_buffer(acc_buf_name, [num_thread, size_per_thread], dtype=accumulate_dtype)}}

  auto q_data = query;
  auto k_data = key;
  auto v_data = value;
  int64_t* mask_data = nullptr;
  // scalar_t* out_data = output.data_ptr<scalar_t>();
  auto out_data = output;

  accum_t* buf_data = buf;//.data_ptr<accum_t>();
  scalar_t* buf_reduced_data =  nullptr; //is_reduced_type ? buf_reduced.data_ptr<scalar_t>() :

  // Buffer to store padding query
  scalar_t* query_padding_ptr = nullptr;
  std::unique_ptr<scalar_t[]> query_padding_data;

  // Buffer to store Key and Value after transforms
  scalar_t* key_reorder_ptr = nullptr;
  std::unique_ptr<scalar_t[]> key_reorder_data;
  scalar_t* value_reorder_ptr = nullptr;
  std::unique_ptr<scalar_t[]> value_reorder_data;
  int kv_padding_size = (kvSize - 1) / kvSplitSize * ekvSplitSize + ekvTail;

  auto kkk = {{template.apply_score_mod(1,1,4,123,123)}};
  std::cout<<"gjngjn"<<kkk<<std::endl;
  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
     int64_t i = 0, j = 0, k = 0;
     at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * rkvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * ekvSplitSize : nullptr;
        scalar_t* query_t_padding_ptr = nullptr;

        for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          fill_stub(qk_max_data,
              -std::numeric_limits<accum_t>::infinity(), qBlockSize);
          fill_stub(qk_sum_data,
              static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;

          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            int64_t ekvBlockSize = kvBlockSize;
            int64_t rkvBlockSize = kvBlockSize == kvSplitSize ? rkvSplitSize : rkvTail;
            // Calculate scale * q @ k.T
            at::native::cpublas::gemm(
              at::native::TransposeType::Transpose,
              at::native::TransposeType::NoTranspose,
              kvBlockSize,
              qBlockSize,
              headSize,
              static_cast<accum_t>(1),
              k_data + i * kStrideB + j * kStrideH +
                  n * kStrideN,
              kStrideN,
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              qStrideM,
              static_cast<accum_t>(0),
              qk_data,
              kvBlockSize);
            
            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t* row_ptr = qk_data + row * rkvBlockSize;
                fill_stub(row_ptr + last_col + 1,
                    -std::numeric_limits<accum_t>::infinity(),
                    kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            // And apply scaling factor
            // qk <- qk * scaling + attn_mask
            if (has_attn_mask) {
              for (int64_t row = 0; row < qBlockSize; ++row) {
    #if __GNUC__ == 11 && defined(__ARM_FEATURE_SVE)
                  _scale_attn_mask_fusion_kernel(
                    qk_data + row * rkvBlockSize,
                    mask_data + i * mStrideB + j * mStrideH +
                        (m + row) * mStrideM + (mStrideN == 0 ? 0 : n),
                    kvBlockSize,
                    qk_data + row * rkvBlockSize,
                    scaling_factor,
                    mStrideN == 0);
    #else
                  if (mStrideN == 0) {
                    _scale_attn_mask_fusion_kernel</*is_stride_0*/ true>(
                      qk_data + row * rkvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM,
                      kvBlockSize,
                      qk_data + row * rkvBlockSize,
                      scaling_factor);
                  } else {
                    _scale_attn_mask_fusion_kernel</*is_stride_0*/ false>(
                      qk_data + row * rkvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM + n,
                      kvBlockSize,
                      qk_data + row * rkvBlockSize,
                      scaling_factor);
                  }
    #endif
              }
            }
            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              // apply scaling factor and max per row in fusion
              _mul_reduce_max_fusion_kernel(
                  qk_data + row * rkvBlockSize,
                  scaling_factor,
                  kvBlockSize,
                  qk_data + row * rkvBlockSize,
                  tmp_max);
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                // to avoid `nan = exp2f(-inf - (-inf))`
                fill_stub(conditional_data_ptr(qk_data, qk_reduced_data) + row * ekvBlockSize,
                  static_cast<scalar_t>(0), kvBlockSize);
              } else {
                tmp_sum = tmp_max;
                // qk <- exp(qk - max) and sum per row
                _exp_reduce_sum_fusion_kernel(
                    qk_data + row * rkvBlockSize, kvBlockSize,
                    conditional_data_ptr(qk_data, qk_reduced_data) + row * ekvBlockSize,
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
                    dst_data + row * rHeadSize,
                    dst_data + row * rHeadSize,
                    headSize);
                }
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            at::native::cpublas::gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              headSize,
              qBlockSize,
              kvBlockSize,
              static_cast<accum_t>(1),
              v_data + i * vStrideB + j * vStrideH +
                  n * vStrideN,
              vStrideN,
              conditional_data_ptr(qk_data, qk_reduced_data),
              kvBlockSize,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data,
              headSize);
          }

          // dst <- dst / sum[row]
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            // Row sums for full masked out rows are 0, we set them to 1
            // in order to avoid NaNs in the output and instead set fully
            // masked out rows to 0
            qk_max_data[row] = qk_max_data[row] == -std::numeric_limits<accum_t>::infinity() ? 0 : qk_max_data[row];
            qk_sum_data[row] = qk_sum_data[row] == 0 ? 1 : qk_sum_data[row];
            accum_t sum_reciprocal = 1 / qk_sum_data[row];
            at::vec::map<scalar_t>(
              [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
              out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
              dst_data + row * rHeadSize,
              headSize);
          }
          // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
    });
}
"""

from ..lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
    select,
    slice_,
    empty_like,
    copy_,
    ones_like,
    clone,
)
from ..ir import (
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    StorageBox,
    Subgraph,
    TensorBox,
)
def get_padded_n(n, block_n):
    return (n + block_n - 1) // block_n * block_n
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from ..kernel.mm import tuned_mm
class CppMHATemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        block_mask
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16, torch.half, torch.uint8]
        super().__init__(
            "mha",
            input_nodes,
            layout,
            1
        )
        self.scale = scale
        self.score_mod = score_mod
        self.block_mask = block_mask
    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        block_mask
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout
        def postprocessor(output ):
            # breakpoint()
            return output
        template = DataProcessorTemplateWrapper(
            CppMHATemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale= scale,
            score_mod=score_mod,
            block_mask=block_mask
        )
        template.maybe_append_choice(choices)
        return template
    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        breakpoint()
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()
    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        query = kernel.permute(self.input_nodes[0], [0,2,1,3])
        key = kernel.permute(self.input_nodes[1], [0,2,1,3])
        value = kernel.permute(self.input_nodes[2], [0,2,1,3])
        q_split_size = 32
        kv_split_size = 512
        batchSize = query.layout.size[0]
        qSize = query.layout.size[1]
        kvSize = key.layout.size[1]
        num_head = query.layout.size[2]
        headSize = query.layout.size[3]

        qSplitSize = qSize if q_split_size > qSize else q_split_size
        kvSplitSize = kvSize if kv_split_size > kvSize else kv_split_size
        qSlice = (qSize + qSplitSize - 1) / qSplitSize
        kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize
        kvTail = (kvSize - 1) % kvSplitSize + 1
        rHeadSize = headSize
        rkvSplitSize = kvSplitSize
        rkvTail = kvTail
        rkvSize = rkvTail if kv_split_size > kvSize else rkvSplitSize * kvSlice + rkvTail

        headSize_even = headSize % 2 == 0
        eheadSize = headSize
        ekvSplitSize = kvSplitSize
        ekvTail = kvTail
        size_per_thread = qSplitSize * rkvSplitSize + qSplitSize + qSplitSize + qSplitSize * rHeadSize

        num_threads = parallel_num_threads()
        buf_out = TensorBox.create(self.output_node)

        # dtype = query.scalar_type();
        # accumulate_dtype = toOpMathType(dtype);

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            buf_out = template_buffer_node

        options = dict(
            query=query,# self.input_nodes[0],
            key=key,#self.input_nodes[1],
            value=value,#self.input_nodes[2],
            scale=self.scale,#self.input_nodes[3],
            size_per_thread=size_per_thread,
            # dtype=dtype,
            accumulate_dtype=torch.float,
            # score_mod=self.input_nodes[3],
            # block_mask=self.input_nodes[4],
            # kernel_options=self.input_nodes[6],
            # score_mod_other_buffers=self.input_nodes[7],
            # mask_mod_other_buffers=self.input_nodes[8],
            template=self,
            output = buf_out,
            kernel=kernel,
            num_thread=num_threads,
        )
        with contextlib.ExitStack() as stack:
            return self._template_from_string(GEMM_TEMPLATE).render(**options)

