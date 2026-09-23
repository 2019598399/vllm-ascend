# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers.

On vLLM Ascend, SP currently covers the MoE path (SP MoE). The attention
`o_proj` ends with a TP all-reduce, so its inputs are replicated on every TP
rank. Feeding those replicated tokens directly into the experts duplicates
compute and communication under expert parallelism. SP MoE keeps the expert
inputs sharded by sequence and restores the expected layout at the MoE output
boundary instead.

**The original flashcomm feature overlapped functionally with the SP feature and has been deprecated since v0.27.1.**

## Principle

SP MoE shards the input along the token dimension in each
Transformer layer. Different TP ranks therefore process different tokens,
avoiding duplicate expert computation for the same tokens.

The main data flow of an MoE layer is:

```text
Sequence-parallel input sharding
  -> TP all-gather: collect tokens from all ranks
  -> attention
  -> TP reduce scatter
  -> RMS Norm
  -> Router
  -> all-to-all
  -> Moe
```

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

Steps to follow to enable SP currently:

- `tensor_parallel_size > 1` and `data_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `--additional-config '{"enable_flashcomm1": true}'` set `flashcomm1`

### Matmul reduce-scatter fusion

On supported Ascend devices, a sequence-parallel row-parallel projection can
run its matmul and token-dimension reduce-scatter as one CANN kernel. It is
opt-in through the upstream compilation switch:

```bash
vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}' \
  --compilation-config '{"pass_config": {"fuse_gemm_comms": true}}'
```

Fusion applies only to BF16/FP16 bias-free projections with TP size 2, 4, or
8 and a supported contracted dimension. Other graphs retain the unfused path.
When SP/MMRS is enabled, Ascend derives its default token threshold from the
hardware profile; profiles without a benchmarked value temporarily use an
8 MiB-per-TP-rank policy. Set `pass_config.sp_min_token_num` explicitly to
override that policy for a deployment.
To verify a real workload without a model-specific reproducer, set
`VLLM_DEBUG_DUMP_PATH` and check the emitted FX graph for
`vllm.npu_matmul_reduce_scatter`; Ascend profiling then shows the corresponding
`npu_mm_reduce_scatter_base` kernel.

### Temporary FlashComm switch (Ascend only)

Until SP support is fully validated, vLLM Ascend keeps SP MoE option by original flashcomm option.

To opt into upstream SP MoE, set one of the following (the
`additional_config` form is preferred):

```bash
# Preferred.
vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}'
```

```bash
# Kept for compatibility.
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```

This switch is temporary and deprecated. Referencing either form logs a
`FlashComm is deprecated` warning from `init_ascend_config`, and the override
carries a `TODO` to remove it once SP is supported — after that, the upstream
configuration above takes effect directly.
