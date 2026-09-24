# 图 1 修改提示词与排版规范

用途：编辑论文总体架构图，使其与当前 `core/models.py`、`core/environment.py` 和训练器一致。

参考图：`figures/generated/gnn_architecture_ai_fig1_draft_v2.png`。这是修改目标与风格参考，保留旧文件，不覆盖。

## English prompt

Edit the supplied multi-robot GNN architecture diagram for an IEEE research
paper. Preserve the six-panel left-to-right layout, blue and green panel
colors, four colored robots, obstacles, goal stars, directed workflow arrows,
and the bottom feedback loop. Use a white background, flat vector graphics,
high-contrast black labels, consistent typography, and compact margins.
Do not add new modules or change the implemented algorithm. All text is English.

Panel 1 — “Multi-robot scene”: show four robots with different colors,
goal stars, static obstacles, velocity arrows, and dashed local neighbor links.
Label the observations as position p_i, velocity v_i, goal g_i, and obstacles.

Panel 2 — “Robot observations”: remove “Candidate actions”. Show an
eight-dimensional node vector with five groups: normalized goal displacement
(g_i − p_i) ⊘ s, 2D; normalized current velocity v_i/v_max, 2D;
nearest-obstacle outward unit direction n_i^obs, 2D;
normalized obstacle distance η_i = clip(d_i^obs/4, 0, 1), 1D; and constant 1, 1D.
The goal displacement is NOT a unit goal direction and absolute position is
NOT concatenated directly into the node input. s = [W,H].

Panel 3 — “Dynamic graph input”: display
A_ij = 1{0 < ||p_i − p_j||_2 ≤ R_c},
r_ij = p_i − p_j, Δv_ij = v_i − v_j,
n_ij = r_ij/max(||r_ij||_2, ε), c_ij = −Δv_ij^T n_ij,
e_ij = A_ij [r_ij ⊘ s, Δv_ij/v_max, c_ij/v_max] ∈ R^5.
Label c_ij as closing speed and the edge direction as sender j → receiver i.
Do not include separate distance or communication-indicator feature channels.
Show X, A, E as the structured network inputs; ε = 10^−6.

Panel 4 — “Feature encoding”: use “Node encoder: Linear + ReLU” with
h_i^(0) = ReLU(W_x x_i + b_x). Use “Mean edge feature” before “Edge encoder:
Linear”, with ē_i = Σ_j A_ij e_ij / D_i and z_i = W_e ē_i + b_e.
D_i = max(1, Σ_j A_ij). The edge encoder is shared across all three layers.
Draw a visible arrow from z_i to the repeated message-passing block.
Do not depict an independent deep edge MLP or a ReLU after the edge encoder.

Panel 5 — “Shared GNN policy”: show three message-passing layers, each
containing BOTH mean neighbor aggregation and node update. Present the
common per-layer computation compactly:
h̄_i^(l) = Σ_j A_ij h_j^(l) / D_i,
m_i^(l) = ReLU(Linear_m,l(h̄_i^(l) + z_i)),
h_i^(l+1) = ReLU(Linear_u,l([h_i^(l) || m_i^(l)])).
Here each Linear includes its bias. Label the block “Repeat for l = 0, 1, 2”
and “128 hidden units”. Parameters are shared across robots; the message
and update parameters are different across layers. Connect the last layer
to a shared “Linear → ReLU → Linear” action head, a_i = f_head(h_i^(3)).
There is NO tanh or other final activation. Do not show layer 1 as only
aggregation, layer 2 as only update, and layer 3 as global connectivity.

Panel 6 — “Velocity execution”: show, in this exact order,
raw velocity u_i^raw = v_max a_i;
“Optional pairwise correction” (sequential, approaching pairs);
“Norm clipping”, ||u_i||_2 ≤ v_max;
“Executed velocity u_i”; and robot motion. Label the safety correction as
optional. Never write “ORCA correction”, “guaranteed safe”, or “safe velocity”.
It is a custom sequential pairwise correction and not an ORCA solver.

The bottom arrow feeds executed motion back to the scene and is labeled
“Next observation: rebuild graph and recompute features”. The exact training
loss equations belong in the manuscript, not in this inference diagram.
All formulas must be legible and mathematically correct, with consistent
subscripts and superscripts. Do not invent symbols, feature dimensions,
candidate-action inputs, or training stages.

## 交付与验收

- 输出新版本 `gnn_architecture_aligned_v3`，保留 `.drawio`、`.svg`、`.pdf`、`.png`。
- 首选可编辑 Draw.io 元素；公式用确定性数学排版，避免生成式文字导致符号错误。
- 从 Draw.io 布局模型导出 SVG，再由 SVG 导出横向 PDF，四周约 1 mm。
- 保留六栏风格；文字可精简，完整定义以正文为准，不挤入训练损失。
- 重点检查：8/5 维、i−j 方向、平均后边编码、每层均聚合及更新、输出无 tanh、先修正后限速。
- 本次采用本地矢量编辑与公式排版，不调用图片生成 API。此提示词同时保留给后续图像编辑工具使用。
