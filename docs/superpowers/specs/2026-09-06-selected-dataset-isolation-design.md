# 所选数据集隔离设计

## 目标

一次运行只处理 `configs/run.yaml` 明确选择的数据集。多个数据集同时选择时，攻击图、清单和特征文件分别保存；统一检测器只合并本次选择的数据集，并保留每行的 `dataset_id`。评测只统计本次选择的数据集。

## 数据流

- 生成：继续使用 `artifacts/attacks/corpus/<profile>/...` 和每个 profile 独立清单，不跨 profile 查找或复用图。
- 特征：继续使用 `artifacts/features/profiles/<profile>.pt`，每个文件只包含该 profile 的样本。
- 训练：工作流把本次 `selection.profiles` 显式传给训练函数；训练函数只读取这些 profile 文件，再合并为一个统一检测器的训练/验证张量。
- 测试：每个 profile 单独生成和保存测试图。任何一个所选 profile 失败时，本次评测跳过，避免旧测试结果冒充本次结果。
- 评测：工作流把本次 `selection.profiles` 显式传给评测函数；报告既保存所选数据集的汇总，也保存逐数据集结果。

## 兼容性

命令行直接调用 `train-detector` 或 `evaluate-detector` 且没有传入 profile 列表时，保留“读取全部已准备 profile”的旧行为。只有 `run-selected` 强制按本次选择过滤。

## 验证

- 单选 Cora 时，训练和评测不会读取 Polblogs 文件。
- 同选 Cora、Citeseer 时，只读取这两个文件，且训练只执行一次。
- 任一所选数据集测试失败时，不执行评测。
- 完整测试套件继续通过。
