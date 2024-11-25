# mypy: allow-untyped-defs
import contextlib
import logging
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.utils
from .. import ir
from ..ir import TensorBox
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import parallel_num_threads
from .cpp_template import CppTemplate
from ..virtualized import V

log = logging.getLogger(__name__)

ATTENTION_TEMPLATE = r"""
{{template.header().getvalue()}}
#include <ATen/native/CPUBlas.h>

{%- set kernel_args = {"query": query, "key": key, "value": value, "kv_indices": kv_indices, "mask_other": mask_mod_other_buffers} %}
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{
  // kv page size, q and kv split size
  int64_t kvBlockSize = {{kvBlockSize}};
  int64_t qSplitSize = {{qSplitSize}};
  int64_t kvSplitSize = {{kvSplitSize}};

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
  bool is_broadcast_bs_kv = batchSize != batchSize_k;
  bool is_broadcast_head_kv = num_head != num_head_k;
  int64_t gqa_shards = num_head / num_head_k;
  int64_t bs_shards = batchSize / batchSize_k;

  int64_t batchSize_kvi = {{kernel.size(kv_indices, 0)}};
  int64_t num_head_kvi = {{kernel.size(kv_indices, 1)}};
  bool is_broadcast_bs_kvi = batchSize != batchSize_kvi;
  bool is_broadcast_head_kvi = num_head != num_head_kvi;
  int64_t gqa_shards_kvi = num_head / num_head_kvi;
  int64_t bs_shards_kvi = batchSize / batchSize_kvi;

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
  int64_t kviStrideB = {{kernel.stride(kv_indices, 0)}};
  int64_t kviStrideH = {{kernel.stride(kv_indices, 1)}};
  int64_t kviStrideQ = {{kernel.stride(kv_indices, 2)}};

  int kv_blocks_num = 0;
  if(*(kv_indices) == 0){
    if({{kernel.size(kv_indices, 3)}} >= 1 and *(kv_indices+1) > 0){
      kv_blocks_num++;
    }
  }
  for(int kv_blocks = kv_blocks_num; kv_blocks <{{kernel.size(kv_indices, 3)}}; kv_blocks++){
    if(*(kv_indices+kv_blocks)>0 or *(kv_indices+kviStrideB+kv_blocks)>0){
      kv_blocks_num++;
    }
  }

  int64_t kvSize = {{kernel.size(key, 1)}};
  // update kvSize, incase like page attention has allocated extra buffers
  kvSize = kvSize <= kv_blocks_num * kvBlockSize? kvSize : kv_blocks_num * kvBlockSize;
  qSplitSize = qSplitSize > qSize ? qSize : qSplitSize;
  kvSplitSize = kvSplitSize > kvSize ? kvSize : kvSplitSize;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;


  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
        /* qk     */ qSplitSize * kvSplitSize +
        /* qk_max */ qSplitSize +
        /* qk_sum */ qSplitSize +
        /* dst    */ qSplitSize * headSize;

  {%- set acc_buf_name = "buf" %}
      {{kernel.define_buffer(acc_buf_name, [num_thread, size_per_thread], dtype=accumulate_dtype)}}
  {%- set acc_reduced_buf_name = "buf_reduced" %}
      {{ kernel.define_buffer(acc_reduced_buf_name, [num_thread, qSplitSize, kvSplitSize], dtype=query_dtype)}}

  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  auto  kv_indices_data = kv_indices;

  scalar_t* out_data = output;

  accum_t* buf_data = buf;
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced : nullptr;

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
     int64_t i = 0, j = 0, k = 0;
     at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;
        scalar_t* query_t_padding_ptr = nullptr;

        for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
          int64_t m = k * qSplitSize;
          int64_t cur_qSplitSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          fill_stub(qk_max_data,
              -std::numeric_limits<accum_t>::infinity(), cur_qSplitSize);
          fill_stub(qk_sum_data,
              static_cast<accum_t>(0), cur_qSplitSize);
          int64_t num_keys = kvSize;

          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t cur_kvSplitSize = std::min(kvSplitSize, kvSize - n);
            cur_kvSplitSize = cur_kvSplitSize == kvSplitSize ? kvSplitSize : kvTail;

            // Calculate scale * q @ k.T
            auto kv_block_num = n / kvBlockSize;
            auto kv_block_offset = n - kv_block_num * kvBlockSize;
            // getting kv indices by [BS, Head, 1, kv_block_num]
            auto i_kvi = is_broadcast_bs_kvi ? i/bs_shards_kvi : i;
            auto j_kvi = is_broadcast_head_kvi ? j/gqa_shards_kvi : j;
            auto kv_logical_data = kv_indices_data + i_kvi*kviStrideB + j_kvi*kviStrideH  + kv_block_num;
            auto i_kv = is_broadcast_bs_kv ? i/bs_shards : i;
            auto j_kv = is_broadcast_head_kv ? j/gqa_shards : j;
            auto k_addr = k_data + i_kv * kStrideB + j_kv * kStrideH + (*kv_logical_data * kvBlockSize)  * kStrideN;

            at::native::cpublas::gemm(
              at::native::TransposeType::Transpose,
              at::native::TransposeType::NoTranspose,
              cur_kvSplitSize,
              cur_qSplitSize,
              headSize,
              scaling_factor,
              k_addr,
              kStrideN,
              q_data + i * qStrideB + j * qStrideH +
                  m * qStrideM,
              qStrideM,
              static_cast<accum_t>(0),
              qk_data,
              cur_kvSplitSize);
            
            {%- if score_mod and mask_mod %}
            // apply score mod function
            for (int64_t row = 0; row < cur_qSplitSize; ++row) {
              for(int col = 0; col< cur_kvSplitSize; col++){
                std::vector<int64_t> b_idx = {i};
                std::vector<int64_t> h_idx = {j};
                std::vector<int64_t> q_idx = {k*cur_qSplitSize+row};
                int64_t phisical_kv_idx = *kv_logical_data * kvBlockSize + col;
                std::vector<int64_t> kv_idx = {phisical_kv_idx};
                accum_t* in_ptr0 = qk_data + row * cur_kvSplitSize + col;
                auto in_ptr1 = b_idx.data();
                auto in_ptr2 = h_idx.data();
                auto in_ptr3 = q_idx.data();
                auto in_ptr4 = kv_idx.data();
                {%- if mask_mod_other_buffers %}
                auto in_ptr5 = mask_other;
                {%- endif %}
                accum_t* out_ptr0 = in_ptr0;
                {{template.modification(score_mod)}}
                }
            }
            // Apply block mask, fill unused with -inf
            for (int64_t row = 0; row < cur_qSplitSize; ++row) {
              for(int col = 0; col< cur_kvSplitSize; col++){
                std::vector<int64_t> b_idx = {i};
                std::vector<int64_t> h_idx = {j};
                std::vector<int64_t> q_idx = {k*cur_qSplitSize+row};
                int64_t phisical_kv_idx = *kv_logical_data * kvBlockSize + col;
                std::vector<int64_t> kv_idx = {phisical_kv_idx};
                accum_t* qk_block = qk_data + row * cur_kvSplitSize + col;
                auto in_ptr1 = b_idx.data();
                auto in_ptr2 = h_idx.data();
                auto in_ptr3 = q_idx.data();
                auto in_ptr4 = kv_idx.data();
                {%- if mask_mod_other_buffers %}
                auto in_ptr5 = mask_other;
                {%- endif %}                
                std::vector<int64_t> temp = {0};
                int64_t* out_ptr0 = temp.data();
                {{template.modification(mask_mod)}}
                *qk_block = *out_ptr0!=0 ?  *qk_block : -std::numeric_limits<accum_t>::infinity();
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
                fill_stub(conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_kvSplitSize,
                  static_cast<scalar_t>(0), cur_kvSplitSize);
              } else {
                tmp_sum = tmp_max;
                // qk <- exp(qk - max) and sum per row
                _exp_reduce_sum_fusion_kernel(
                    qk_data + row * cur_kvSplitSize, cur_kvSplitSize,
                    conditional_data_ptr(qk_data, qk_reduced_data) + row * cur_kvSplitSize,
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
                    dst_data + row * headSize,
                    dst_data + row * headSize,
                    headSize);
                }
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            auto v_addr = v_data + i_kv * vStrideB + j_kv * vStrideH + (*kv_logical_data * kvBlockSize)  * vStrideN;

            at::native::cpublas::gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              headSize,
              cur_qSplitSize,
              cur_kvSplitSize,
              static_cast<accum_t>(1),
              v_addr,
              vStrideN,
              conditional_data_ptr(qk_data, qk_reduced_data),
              cur_kvSplitSize,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data,
              headSize);
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
              dst_data + row * headSize,
              headSize);
          }
          // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
    });
}
"""

class CppMHATemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        fake_buffers,
    ) -> None:
        assert layout.dtype in [torch.float, torch.float16, torch.bfloat16]
        super().__init__("mha", input_nodes, layout, parallel_num_threads())
        self.scale = scale
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.kv_block_size = kv_block_size
        self.fake_buffers = fake_buffers

    def modification(self, subgraph_buffer):
        has_other_buf = len(self.input_nodes) == 6
        if has_other_buf:
            score_other_buf_name = self.input_nodes[4].get_name()
            mask_other_buf_name = self.input_nodes[5].get_name()

        assert isinstance(subgraph_buffer, ir.ComputedBuffer)
        subgraph_buffer_data = subgraph_buffer.data
        from ..loop_body import LoopBody
        from ..utils import sympy_index_symbol_with_prefix, SymT
        from ..virtualized import ops, V

        output_name = "buf0"     
        V.graph.register_buffer(subgraph_buffer)

        from .cpp import CppKernel, CppKernelProxy, KernelGroup
        kernel_group = KernelGroup()
        kernel_input_args = {
            "score": "in_ptr0",
            "b": "in_ptr1",
            "h": "in_ptr2",
            "q_idx": "in_ptr3",
            "kv_idx": "in_ptr4",
        }
        if has_other_buf:
            kernel_input_args.update({
                score_other_buf_name: "in_ptr5",
                mask_other_buf_name: "in_ptr5",
            })

        kernel_output_args = {
            "buf0": "out_ptr0"
        }

        args = kernel_group.args
        for name, inp in kernel_input_args.items():
            args.input_buffers[name] = inp

        for name, inp in kernel_output_args.items():
            args.output_buffers[name] = inp        

        kernel_group.args = args

        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []

        var_sizes = (tuple([]))
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
        fake_buffers,
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppMHATemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            score_mod=score_mod,
            mask_mod=mask_mod,
            kv_block_size=kv_block_size,
            fake_buffers=fake_buffers,
        )
        template.maybe_append_choice(choices)
        return template

    def apply_score_mod(self, score, b, h, q_idx, kv_idx):
        return self.score_mod.graph_module(score, b, h, q_idx, kv_idx).item()

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        # Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
        #     -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
        #  Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        #  Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])

        qSize = query.layout.size[1]
        kvSize = key.layout.size[1]
        headSize = query.layout.size[3]

        if qSize >= 768:
            qSplitSize = 256
            kvSplitSize = 512
        elif qSize >= 192:
            qSplitSize = 64
            kvSplitSize = 512
        else:
            qSplitSize = 32
            kvSplitSize = 512

        if self.kv_block_size < kvSplitSize:
            kvSplitSize = self.kv_block_size

        # get size_per_thread/qSplitSize/kvSplitSize for buf/buf_reduced creation
        qSplitSize = min(qSplitSize, qSize)
        kvSplitSize = min(kvSplitSize, kvSize)
        # sizes for qk + qk_max + qk_sum + dst
        size_per_thread = (
            qSplitSize * kvSplitSize + qSplitSize + qSplitSize + qSplitSize * headSize
        )

        num_threads = parallel_num_threads()
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node
        has_other_buffer = len(self.input_nodes) == 6
        options = dict(
            query=query,
            key=key,
            value=value,
            kv_indices=self.input_nodes[3],
            score_mod_other_buffers=self.input_nodes[4] if has_other_buffer else None,
            mask_mod_other_buffers=self.input_nodes[5] if has_other_buffer else None,            
            scale=self.scale,
            accumulate_dtype=torch.float,
            query_dtype=query.layout.dtype,
            qSplitSize=qSplitSize,
            kvSplitSize=kvSplitSize,
            size_per_thread=size_per_thread,
            kvBlockSize=self.kv_block_size,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
            score_mod=self.score_mod,
            mask_mod=self.mask_mod,
        )
        with contextlib.ExitStack() as stack:
            for buf in self.fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )            
            return self._template_from_string(ATTENTION_TEMPLATE).render(**options)
