# 正文公式与图 1 对齐记录

## 保存的修改前版本

- GitHub：`https://github.com/XuelinHu/multi-robot-gnn-cooperative-navigation`
- 修改前提交：`8c4277a`（已推送至 `master`）。
- 旧图保留：`figures/generated/gnn_architecture_ai_fig1_draft_v2.png` 及同名 Draw.io。
- 新正文范围：`paper/ieee-manuscript/main-en.tex`、`main-zh.tex`。
- `manuscript/paper_overall.tex` 属于旧开发稿，本轮保留在快照状态，不作为最新公式依据。

## 对齐依据与结果

| 内容 | 代码依据 | 正文与图中处理 |
| --- | --- | --- |
| 节点特征 | `core/environment.py::observation` | 目标相对位移、当前速度、障碍方向、截断距离、常数，共 8 维；补充归一化与零方向情况。 |
| 图与边特征 | `build_graph`、`build_edge_features` | `0 < distance <= radius`；采用 `i-j` 方向；接近速度为第 5 维，非边位置置零。 |
| 消息传递 | `core/models.py::MessagePassingGNN.forward` | 先均值聚合，再加共享边线性编码，随后消息 Linear+ReLU 与更新 Linear+ReLU；三层均执行完整流程。 |
| 偏置与孤立节点 | 同上 | 边张量未传入时省略边编码；传入零张量仍保留编码器偏置，不把二者当作同一配置。 |
| 动作输出 | 模型动作头与 `scripts/evaluate_baselines.py::load_policy` | Linear–ReLU–Linear 输出无量纲动作，乘 1.4 得原始速度，删除 tanh。 |
| 模仿与辅助损失 | `scripts/train_policy.py` | SmoothL1 两个分量求和；进度使用归一化目标位移；平滑比较预测与观测中的当前速度；补充批级有效节点及机器人对归一化。 |
| 安全过滤 | `core/environment.py::safety_filter` | 顺序检查每个 `i<j`，使用已修正速度；0.35 为距离余量，隐含系数 1/s；结束后范数限速。不是 ORCA 求解器。 |
| 位置更新 | `rollout` | 补充工作区坐标裁剪，区分训练中的未裁剪一步预测。 |
| 耗时口径 | `rollout` 中计时边界 | 计时包括观测及策略，不包括安全过滤与环境更新，正文明确说明。 |

## 训练配置核验

三个 `results/checkpoints_multiscale_{gnn,attention,transformer}/*_n40.json`
均记录：种子 7、30 轮、128 隐藏单元、三层，碰撞/进度/平滑权重分别为
0.5/0.5/0.1。多规模数据路径为 `data/expanded/mixed_*`。
`scripts/generate_dataset.py` 的 mixed 分支按回合交替生成 ORCA-style、
RVO-style 和安全过滤后的 ORCA-style 标签。正文及摘要已区分多规模联合训练
与其他检查点，不再暗示所有对比使用同样的辅助损失。

## 图形交付

- 提示词：`figures/prompts/figure1_aligned_v3.md`。
- 构建命令：`conda run -n pyg python scripts/build_aligned_architecture.py`。
- 输出：`figures/generated/gnn_architecture_aligned_v3.{drawio,svg,pdf,png}`。
- 本轮采用确定性的可编辑矢量布局，不使用图片生成 API。
- 生成 Draw.io 模型后，从模型导出 SVG，再由 SVG 转为 PDF 和 PNG；
  独立 PDF 横向，画布四周 1 mm。公式以矢量路径保留，避免字体替代改动符号。
- 正式论文引用矢量 PDF；旧位图和旧 Draw.io 保留。

## 验证

- 数值对照：有边、无边张量、全零边张量的三层 GNN；孤立与位置重合节点；
  边特征；顺序安全修正；范数裁剪；归一化进度与当前速度一致性损失，均通过。
- 本次没有修改模型、训练器、仿真器和实验结果文件。
- 两份正式论文运行 XeLaTeX、BibTeX、XeLaTeX、XeLaTeX；英文 6 页、中文 5 页。
- 最终图形和页面进行渲染检查；无未解析引用和 Overfull。
