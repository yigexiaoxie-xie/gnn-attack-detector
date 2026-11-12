# DeepRobust 通用完整图检测器重构设计

## 目标

把当前由 Day-1～Day-4 临时迭代累积而成的原型，收敛为可开源的图攻击检测与防御工具集：使用一个固定的二分类检测器，对已登记数据集的一张或多张完整图分别输出 `clean` 或 `attacked`；提供 20 个可审计的 DeepRobust 攻击协议；提供鲁棒模型目录和少量互补探针。

“一个检测器”指所有数据集共享同一个最终分类器和阈值。每个数据集仍需要自己的干净参考、特征维数、标签划分和探针检查点；这些属于数据集档案，不是新的检测器。未登记且没有可信参考的数据集只能报告不支持，不能声称达到 80%。

## 保留内容

- 完整图级二分类；每个输入文件独立输出结论和概率。
- PT 与 NPZ 图输入、安全校验、批量上传和网页入口。
- DeepRobust Cora-LCC（2485 节点）作为第一个数据集档案。
- 训练/验证/测试隔离、固定随机种子、产物哈希和每种攻击的指标报告。
- CPU/Windows 脚本和相对路径部署。

## 删除内容

- GreatX、PyG 和项目自定义攻击后端及其专用安装步骤。
- 节点注入、后门、RL-S2V 占位、GraphFool/GRBCD/PRBCD 等不属于本轮 DeepRobust 核心协议的实现。
- `day4`、`victim` 等阶段性配置命名，以及只服务旧单节点或旧 22 方法语料的兼容分支。
- 过期的逐日记录、旧设计/计划、旧环境文件和被替代的测试。
- 训练数据、模型和上传缓存不进入 Git 仓库。

## 20 个攻击协议

统一登记以下 20 个协议，后端均标记为 `deeprobust`：

1. `random_add`
2. `random_delete`
3. `random_flip`
4. `dice_add`
5. `dice_delete`
6. `dice_mix`
7. `node_embedding_add`
8. `node_embedding_delete`
9. `degree_add`
10. `degree_delete`
11. `eigenvector_delete`
12. `rnd`
13. `nettack_structure`
14. `nettack_feature`
15. `nettack_hybrid`
16. `fga`
17. `sgattack`
18. `pgdattack`
19. `metattack`
20. `igattack`

Random、NodeEmbedding 和 Nettack 的子项直接使用 DeepRobust 参数；DICE add-only/delete-only 是平台在 DICE 规则上的显式协议变体；Metattack 默认只接受固定来源和校验值的预计算图，避免 CPU 在线生成。

全局结构协议使用 5%、10%、15%、20% 无向边预算。目标协议在一张完整图中累计攻击 5、10、15、20 个不同目标节点，单目标局部预算固定为 3。特征协议必须保存并检测修改后的 `x`，不能只保存邻接矩阵。

## 数据集档案与检测器

数据集档案记录数据集 ID、来源、预处理、节点/特征/类别数、干净图、划分和探针检查点。Cora 2708 与 Cora-LCC 2485 是两个不同档案，不混用模型或攻击图。

最终检测器只接收固定长度、按节点数/边数归一化的图级特征。训练样本增加 `dataset_id`，以后把 Citeseer、PubMed 等档案生成的语料合并训练；模型文件不保存某一张 Cora 图或某个 GCN 的硬绑定哈希。检测时先选择匹配的数据集档案，再抽取同一特征模式，最后调用同一个检测器。

## 鲁棒模型

模型目录登记 GCN、GCN-Jaccard、GCN-SVD、RGCN、MedianGCN、SimPGCN、ProGNN、GNNGuard、AirGNN、SGC。目录登记不等于每次检测全部运行。

常规在线探针固定为：

- GCN：基础预测变化；
- GCN-Jaccard：结构过滤与恢复信号；
- SimPGCN：相似度和预测稳定性信号；
- ProGNN：数据集登记阶段离线生成鲁棒结构参考。

其余模型放在防御评测页面供用户选择。没有标签时只报告结构处理和一致性，不能伪造分类准确率。

## 评测门槛

测试集不参与训练、阈值选择或数据归一化。报告必须同时给出：总体 accuracy/precision/recall/F1、干净图特异度、20 个攻击协议各自的召回率、四档强度召回率、样本数与随机种子。只有总体准确率和干净图特异度均不低于 80%、20 个协议分别达到 80%，并且独立可信干净图数量达标，界面才显示“达到项目门槛”。

## 开源合规

仓库使用 MIT License；DeepRobust 通过固定版本依赖安装，不把其源码复制进仓库。`THIRD_PARTY_NOTICES.md` 保留 DeepRobust 的版权和 MIT 许可证链接，README 引用 DeepRobust 论文。数据集、预训练权重和预攻击图分别记录来源，不默认提交。

## 验证

- 注册表恰好包含上述 20 个唯一协议，且没有 GreatX/PyG/项目后端。
- 配置只接受登记的数据集档案、协议和相对路径。
- 数据集张量能合并多个 `dataset_id`，检测器模型不绑定单一图哈希。
- 防御目录包含 10 个模型，在线探针集合固定为 3 个，ProGNN 标为离线参考。
- 源码可编译；轻量测试不依赖真实下载；服务器完整测试使用固定依赖环境执行。
