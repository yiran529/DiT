In the cross-attention modules `cross_attn_x_to_z` and `cross_attn_z_to_x`,
add OPTIONAL support for a distance-based attention bias.

Requirements:
- Add new optional inputs: `q_pos` and `kv_pos`, each of shape [B, N, 2], normalized to [-1, 1].
- Compute squared distance d2 = ||q_pos - kv_pos||^2.
- Add bias = - d2 / (2 * sigma^2) to the attention logits BEFORE softmax.
- sigma is a learnable scalar parameter of the attention module (initialize so sigma≈0.5).
- Bias must broadcast to [B, heads, q_len, kv_len].
- If q_pos or kv_pos is None, do NOT apply any bias (behavior identical to before).
- Do NOT change existing attention behavior or signatures otherwise.

Only modify the attention module implementation. Do NOT change model forward yet.

---

In the cross-attention modules `cross_attn_x_to_z` and `cross_attn_z_to_x`,
add OPTIONAL support for a distance-based attention bias.

Requirements:
- Add new optional inputs: `q_pos` and `kv_pos`, each of shape [B, N, 2], normalized to [-1, 1].
- Compute squared distance d2 = ||q_pos - kv_pos||^2.
- Add bias = - d2 / (2 * sigma^2) to the attention logits BEFORE softmax.
- sigma is a learnable scalar parameter of the attention module (initialize so sigma≈0.5).
- Bias must broadcast to [B, heads, q_len, kv_len].
- If q_pos or kv_pos is None, do NOT apply any bias (behavior identical to before).
- Do NOT change existing attention behavior or signatures otherwise.

Only modify the attention module implementation. Do NOT change model forward yet.

---

Modify the model forward pass to replace z→x cross-attention updates with FiLM-style modulation.

Changes:
1. Keep all existing x→z cross-attention calls unchanged.
2. Wherever the code currently does:
   - z_out = cross_attn_x_to_z(z, x, c)
   - x_out = cross_attn_z_to_x(x, z_out, c)
   - x, z = x_out, z_out
   replace it with:
   - z_out = cross_attn_x_to_z(z, x, c)
   - x = film_modulate_x(x, z_out, c)
   - z = z_out

3. Implement `film_modulate_x(x, z, c)`:
   - z_pool = mean over z tokens
   - Use a small MLP to predict gamma, beta, and a scalar gate from concat(z_pool, c)
   - Apply:
     x = x + gate * (gamma[:,None,:] * LayerNorm(x) + beta[:,None,:])

4. Initialize gamma and beta to zero so the modulation is a no-op at start.
5. Do NOT add any new attention or token-level branching.

Only modify the forward logic and add the FiLM module.
