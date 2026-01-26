# DiT-LatentGate

## 核心改进（相对普通 DiT）
- 引入 **latent token 池**：基础 latent 始终存在，额外的 optional latent 由网络预测连续权重（0-1）进行“软门控”，形成可变容量的 latent 表示。
- **LatentWeightPredictor**：用 patch token 的全局平均特征 + 条件向量 c（timestep+label）+ timestep 投影预测 optional latent 权重；提供温度/偏置/目标均值等控制，并可产生稀疏性辅助损失。
- **Dense↔Latent 双向交互**：新增 AdaLN-Zero 的 cross-attention 模块在 x(密集 patch tokens) 与 z(latent tokens) 之间双向交换信息。
- **计算主干转移到 latent**：Transformer block 主要在 z 上迭代；x 仅在间隔步（cross_attn_interval）与 z 进行信息交换，形成 latent bottleneck，降低 dense token 上的计算负担。
- **多阶段融合**：初始 x→z cross-attn、周期性交互、最终 z→x cross-attn 后再解码输出。

## 可能的命名含义
- “LatentGate” 强调 optional latent 的权重门控与可变容量。
- 结构上更接近 “latent bottleneck + 双向 cross-attn” 的 DiT 变体。

## 完整计算流程（突出 latent 机制，省略通用 DiT 细节）
- 1) **得到 x 与条件向量 c**：图像先 patch-embed 得到 x（含固定 sin-cos 位置编码）；timestep 与 label embed 相加得到条件向量 c。（这部分与普通 DiT 一致）
- 2) **构造 basic latent**：从可学习参数 `latent_tokens_basic` 直接复制到 batch 维度，通过`cross_attn_z_to_x`得到 `z_basic`。这是固定容量的 latent 容器，不依赖输入。
- 3) **构造 optional latent（软门控）**：\n  - 用 `x` 的全局平均池化向量 + 条件向量 `c` 拼接，经 `LatentWeightPredictor` MLP 预测每个 optional latent 的连续权重 `w∈(0,1)`，并叠加 `t` 的线性投影；支持温度/偏置/目标均值调节。\n  - 将 `latent_tokens_optional` 按权重逐 token 缩放，得到 `z_optional = latent_tokens_optional * w`。\n  - 可选：若设置 target mean，会产生一个权重均值的辅助稀疏性损失（模型内部存成 `last_optional_aux_loss`）。
- 4) **拼接形成 latent 池**：`z = concat(z_basic, z_optional)`（若 optional=0 则只用 z_basic）。
- 5) **初始压缩（x→z）**：通过 AdaLN-Zero cross-attn（`cross_attn_x_to_z`）让 z 以查询、x 作为 key/value 进行注意力更新。直观上这是将密集 x 信息“压缩”注入 latent。
- 6) **latent 主干前向 + 周期性交互**：\n  - 主要 transformer blocks 在 z 上迭代（`z = DiTBlock(z, c)`）。\n  - 每隔 `cross_attn_interval` 步执行一次 **双向交换**：\n    - `z_out = cross_attn_x_to_z(z, x, c)`：z 读取 x（再次压缩/注入）。\n    - `x_out = cross_attn_z_to_x(x, z_out, c)`：x 读取 z_out（把 latent 信息回写到 dense）。\n    - 更新为 `(x, z) = (x_out, z_out)`。\n  - 若深度不能整除该间隔，末尾再做一次双向交换以保证信息回流。
- 7) **解压回 x（最终融合）**：在所有 latent 迭代后，再执行一次 `cross_attn_z_to_x`，让最终 latent 表示写回到 dense x。
- 8) **输出解码**：x 经过 final layer 得到 patch 输出并 unpatchify 成图像。（与普通 DiT 一致）
