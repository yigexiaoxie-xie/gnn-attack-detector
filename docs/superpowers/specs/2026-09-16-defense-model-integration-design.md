# 防御模型接入设计

## 目标

把平台目录中的十种防御模型变成可真实运行、可报告状态的防御评测功能，同时保持共享攻击检测器及其训练结果不变。

## 边界

- 防御模型只做节点分类鲁棒性评测，不参与攻击检测器的训练和推理。
- 防御产物保存到每个数据集自己的 `artifacts/profiles/<profile>/defenses/`，不覆盖 `artifacts/models/universal-attack-detector.pt`。
- 输入图必须与所选数据集拥有相同节点数，并由平台补齐该数据集的节点属性和标签契约。
- 已有 GCN、GCN-Jaccard、SimPGCN 检查点继续复用，不重复训练。

## 执行模式

| 模型 | 模式 | 运行方式 |
|---|---|---|
| GCN、GCN-Jaccard、SimPGCN | `checkpoint` | 复用现有 profile probe，直接评测输入图 |
| RGCN、MedianGCN、AirGNN、SGC、GNNGuard | `checkpoint` | 每个数据集准备一次独立防御检查点，随后直接评测 |
| GCN-SVD | `input_fit` | 对当前输入邻接矩阵执行截断 SVD 后训练并评测 |
| ProGNN | `offline` | 对当前输入图执行结构学习优化后评测，明确标记为慢任务 |

GNNGuard 不是 DeepRobust 0.2.9 标准导出项。平台只在检测到兼容实现时启用；未安装时显示缺少依赖，不伪造结果。

## 组件

1. `defenses/registry.py` 保存模型显示名、执行模式和检查点文件名。
2. `defense_service.py` 统一校验输入、加载或训练模型、计算干净图和输入图节点准确率。
3. `preparation.py` 在 `prepare-defenses` 中准备可缓存模型；某个可选模型缺少依赖时记录为 unavailable，并继续准备其他模型。
4. `platform_service.py` 返回每个数据集、每个防御模型的真实就绪状态。
5. `web_app.py` 展示全部十种模型，只允许用户运行当前环境实际可用的模型，并说明是否需要即时训练。

## 错误处理

- 缺少 DeepRobust/PyG/GNNGuard 时返回包含模型名和安装原因的错误。
- 检查点必须匹配 profile id、节点数和干净图指纹，否则拒绝加载。
- 单个可选模型准备失败不会删除已有检查点，也不会影响检测器。
- CUDA 不可用或模型不支持 CUDA 时采用 CPU；运行模式和设备写入结果。

## 验证

- 注册表测试验证十个模型及三种执行模式。
- 服务测试用轻量假模型验证检查点路径、输入图校验、结果字段和缺依赖行为。
- 平台测试验证十个模型状态不再硬编码为三种。
- 运行完整测试套件，确认攻击检测流程无回归。

