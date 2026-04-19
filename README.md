# microgpt extensions

This notebook extends the original tiny, pure-Python `microgpt` with four modern transformer ideas:

- `GELU` replaces the original `ReLU` nonlinearity in the feed-forward block.
- `LoRA` adds low-rank adapters to the attention projection matrices.
- `RoPE` replaces learned absolute position embeddings with rotary position encoding in attention.
- `Mixture of Experts (MoE)` swaps the dense MLP for a small routed expert layer.

The implementation lives in [`/Users/lailasmith/Downloads/microgpt.ipynb`](/Users/lailasmith/Downloads/microgpt.ipynb).

## 1. GELU

Reference: [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

Underlying idea:
`ReLU` hard-thresholds activations at zero. `GELU` is smoother: instead of an activation being either fully kept or fully dropped, it scales values continuously according to their magnitude. In practice this often improves optimization in transformer-style networks because it keeps more nuanced information flowing through the MLP.

How it is implemented here:
- A `Value.tanh()` primitive was added to the tiny autograd engine.
- `gelu(x)` uses the standard tanh approximation:
  `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- The feed-forward block uses `GELU` when `USE_GELU = True`.

## 2. LoRA

Reference: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

Underlying idea:
Fine-tuning a full weight matrix is expensive because every parameter is updated. LoRA assumes that many useful task-specific updates can be represented as a low-rank change:

`W_adapted = W + B @ A`

where `A` and `B` are much smaller than `W`. This keeps the original model weights fixed and only learns the compact adapter matrices, dramatically reducing trainable parameters.

How it is implemented here:
- Each attention projection (`Wq`, `Wk`, `Wv`, `Wo`) can be registered with optional LoRA adapters.
- When `USE_LORA = True`, the code keeps the base matrix and adds:
  - `lora_a`: shape `(rank, input_dim)`
  - `lora_b`: shape `(output_dim, rank)`
- `linear_with_lora(...)` computes the base projection plus the scaled low-rank update.

Important note:
LoRA is most natural for fine-tuning a pretrained model. This notebook trains from scratch, so `USE_LORA` is implemented but left `False` by default. Turning it on would freeze random base weights, which is not a realistic LoRA training setup.

## 3. RoPE

Reference: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Underlying idea:
Standard learned position embeddings inject absolute position information by adding a position vector to the token vector. RoPE instead rotates the query and key vectors by a position-dependent angle. This makes attention naturally aware of relative positions because the dot product between rotated vectors reflects their positional offset.

How it is implemented here:
- Learned position embeddings are skipped when `USE_ROPE = True`.
- `apply_rope(...)` rotates pairs of features using sine/cosine angles.
- `apply_rope_by_head(...)` applies the rotation independently inside each attention head.
- The rotary transform is applied to `q` and `k` before attention scores are computed.

Why this is useful:
- RoPE gives position awareness without storing a separate learned position table.
- Relative distance information appears directly in the attention mechanism.

## 4. Mixture of Experts

Reference: [Hugging Face MoE overview](https://huggingface.co/blog/moe#a-brief-history-of-moes)

Underlying idea:
A standard transformer feed-forward layer is dense: every token passes through the same MLP. A Mixture of Experts instead keeps several specialized MLPs ("experts") and uses a gate to route each token to only a small subset of them. This increases model capacity without forcing every token to use every parameter.

How it is implemented here:
- Each transformer layer can own:
  - a gating matrix `moe_gate`
  - several expert MLPs `expert*.fc1` and `expert*.fc2`
- `moe_mlp(...)` computes gate probabilities, selects the top experts, and combines only their outputs.
- `TOP_K_EXPERTS` controls how many experts a token uses.

Implementation note:
This notebook uses a compact educational version of sparse routing. It does not add advanced production features like auxiliary load-balancing losses, expert capacity constraints, or distributed expert parallelism.

## Design choices in this notebook

- `USE_GELU = True`: enabled by default because it cleanly upgrades the dense nonlinearity.
- `USE_ROPE = True`: enabled by default because it integrates naturally with attention.
- `USE_MOE = True`: enabled by default to demonstrate routed feed-forward computation.
- `USE_LORA = False`: implemented but disabled by default because LoRA is mainly a fine-tuning method, not a scratch-training method.

## Summary

This version of `microgpt` now demonstrates four influential transformer ideas in a single minimal notebook:

- `GELU` for smoother nonlinear activations
- `LoRA` for parameter-efficient low-rank adaptation
- `RoPE` for rotary positional encoding in attention
- `MoE` for sparse routed feed-forward capacity

The goal of the implementation is educational clarity, not production efficiency. It keeps the original notebook's spirit: small, readable, and close to the underlying algorithms.
