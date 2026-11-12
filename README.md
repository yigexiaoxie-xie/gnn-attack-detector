# 图深度学习模型对抗攻击检测与防御工具集

本项目是一个图神经网络对抗攻击检测与防御平台，支持扰动图生成、共享 GCN
检测器训练、图文件上传与批量检测，以及鲁棒防御模型的效果对比。平台支持
Cora、CiteSeer、PubMed 和 Polblogs，并可按数据集查看检测结果。

## CPU/GPU 自动选择

默认配置 `runtime.device: auto` 会在 CUDA 可用时优先使用 GPU，否则使用
CPU。遇到 CUDA 显存不足或兼容性错误时，系统只把当前尚未完成的任务用 CPU
重试一次并打印 `[GPU-FALLBACK]`；依赖 NumPy、SciPy 或谱分解的 CPU-only
攻击仍在 CPU 上运行。

Windows 环境请选择一个安装脚本：

```powershell
# NVIDIA GPU 服务器
.\scripts\setup_windows_cuda.ps1

# 只有 CPU 的电脑
.\scripts\setup_windows_cpu.ps1
```

运行 CUDA 脚本后不要再运行 CPU 脚本，因为后者会按设计安装 CPU 版
PyTorch/PyG。

## 当前设计

- 一个共享GCN检测器：根据 `configs/run.yaml` **本次选中的数据集和攻击**训练。
  GCN的参数与图级分类头联合训练，不再使用每数据集三个模型的响应差异。
  完整输入图不需要干净参考图、节点标签或数据集名称，支持不同节点数和属性维度。
- 输入为12维节点描述符，GCN 12→32→32，全图均值/最大值/95%分位数池化形成96维，
  加4维属性一致性统计，再交给分类头。借鉴ProGNN的特征平滑性思想，不运行完整ProGNN，
  不进行谱分解、低秩优化，也不将某张图“恢复后的结果”冒充真实干净图。
- 缺失、单位矩阵和常量属性屏蔽；Polblogs默认采用无内容属性模式。结构和属性规范仍需校验。
- 每次训练会自动更新当前检测器，同时保存独立运行版本和旧模型备份。仅训练Cora只更新Cora报告；其他数据集的报告保留，不把不同模型结果混成一个总指标。
- 旧攻击图仍可复用；旧252维差异特征和MLP检查点不能用于新检测器，必须重提取输入并重训。
- 23 种攻击协议：Random 加边/删边/翻边，DICE 加边/删边/混合，度中心性
  加边/删边，PRBCD，Structack，GFAttack，无目标 FGAttack，RND，Nettack
  结构/特征/混合，FGA，SGAttack，PGDAttack，Metattack，IGAttack，以及无需梯度、
  谱分解或代理训练的 Bridge Add 和 Hub Rewire 快速结构攻击。
- 10 种防御模型目录：GCN、GCN-Jaccard、GCN-SVD、RGCN、MedianGCN、
  SimPGCN、ProGNN、GNNGuard、AirGNN、SGC。防御模型不再是攻击检测的前置条件。
  平台不再预训练或复用防御检查点；每张输入图都从头训练所选防御模型。
- 防御评测使用所选 DeepRobust 数据集档案自带的 `idx_train`、`idx_val`、`idx_test`
  和可信节点标签。输入文件提供被扰动后的边及节点属性；平台分别在干净图和输入图上
  用相同随机种子、相同数据划分训练同一种模型，再比较测试节点准确率。训练参数仅用于
  当前这张输入图，评测结束后丢弃，不会覆盖共享攻击检测器。
- GCN-SVD 会在当前输入图上执行截断 SVD；ProGNN 会在当前输入图上执行耗时的结构优化。
  这些操作都属于防御模型本身的训练流程，而不是攻击检测器的组成部分。
- 标准 DeepRobust 0.2.9 不直接提供 GNNGuard，未安装兼容实现时平台会显示不可用。
  ProGNN 使用稠密邻接矩阵，为防止内存溢出，节点数超过 5000 的图（包括 Pubmed）不启用。

项目调用外部 DeepRobust 和 GreatX 包，并在本项目中独立实现 Structack
适配器及两种快速结构攻击，不把第三方仓库源代码复制进来。许可证和归属见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Windows 服务器运行

项目推荐 Python 3.10。先在项目根目录建立 `.venv`，然后执行：

```powershell
# 已有可用环境请跳过安装；GPU服务器用CUDA安装脚本，不要用CPU脚本覆盖。
.\scripts\setup_windows_cuda.ps1
```

以后正常运行只需要两步。先打开唯一的批次选择文件：

```powershell
notepad .\configs\run.yaml
```

在 `profiles` 和 `attacks` 下，把本次要运行项目开头的 `#` 删除（取消注释）；
不运行的项目保留或重新加上 `#`。一次可以取消注释多个数据集、多个攻击方法。
保存后执行：

```powershell
.\scripts\run_selected_attacks.bat
```

脚本按“准备所选数据集 → 生成/复用所选训练验证图 → 只在所选数据集上训练共享
GCN → 生成/复用所选测试图 → 用新模型评测所选数据集”的顺序运行。
一次全选四个数据集就是四者联合训练；只选Cora就是只训练Cora，不自动累计其他数据集。
其他数据集的文件与报告保留，平台会标注与当前模型不同的历史报告。
任一所选数据集的测试生成失败时，本次评测会停止，并保留上一次完整评测结果。
已存在且通过哈希、图结构和扰动预算
校验的图会自动复用；中途停止后再次执行会从缺失项继续，不要删除 `artifacts`
或 `.venv`。

**已经生成好攻击图，只升级检测器时，不要重新生成攻击：**

```powershell
.\scripts\train_detector.ps1
# 等价：python -m gnn_attack_detector.cli rebuild-detector --run-config configs/run.yaml
```

训练完成后会自动更新平台当前使用的检测器。每个数据集的评测报告独立保存；后续训练其他数据集不会删除已有数据集的报告，只有再次训练同一数据集时才会更新它。

运行 `train_detector.ps1` 并完成测试评测后，可在
`artifacts/reports/training-runs/<时间戳>/summary.txt` 查看本次各数据集的准确率及其算术平均值。
同一目录也保存原始评测报告和 CSV 汇总表。仓库提供源码、配置和运行脚本，数据集、
模型及评测结果由运行流程生成并保存在本地。

详细替换文件、迁移和输入格式见 [共享GCN迁移说明](docs/shared-gcn-migration.md)。

当前已登记四个数据集档案：`cora-lcc`、`citeseer-lcc`、`polblogs-lcc`、
`pubmed-lcc`。它们可以一起为同一个通用检测器提供训练数据。

防御模型与通用检测器完全分开，不需要提前训练防御模型。只要该数据集的
`artifacts/profiles/<数据集>/clean.pt` 已存在，就可以在平台“防御工具”页上传图并运行。
平台将对每张图按 DeepRobust 原始划分临时训练所选模型，不保存防御模型参数。

如果某个数据集尚未准备过，可运行下面的兼容命令准备干净数据档案和攻击生成所需的
普通 GCN 代理；该命令不再生成任何防御检查点：

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli prepare-defenses `
  --profiles cora-lcc citeseer-lcc polblogs-lcc pubmed-lcc
```

防御输入图必须与所选数据集保持相同的节点顺序、节点数量和属性维度。平台采用输入图的
扰动结构/属性，同时从可信数据档案恢复节点标签和 DeepRobust 划分；因此不能把来源未知、
节点重新编号的图直接冒充某个已登记数据集。缺少可选依赖时只会使对应防御模型不可运行，
不会覆盖攻击检测器。

本次计划的组合数是“数据集数 × 攻击方法数”。每种攻击每个数据集产生
48 个图槽位（4 个强度 × 训练 5 + 验证 2 + 测试 5），因此四种强度分别有
20/8/20 张训练、验证、测试图。已有且通过完整校验的图会直接复用；官方
Metattack 文件会逐个使用且每个文件只使用一次；独立预计算图用完后会自动调用
DeepRobust 实时生成缺少的图。实时 Metattack 在 CPU 上很慢，因此应尽量预先放入更多
经过校验的独立图。对于 PGD、FGA 等优化型攻击，若达到最大重试次数仍无法得到新的合格结构，任务会输出
拒绝原因并停止该槽位，不会用重复图凑数。预计攻击图总数等于
各攻击数量之和再乘所选数据集数。
运行前可只预览、不生成：

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli show-selection `
  --config configs/base.yaml --run-config configs/run.yaml
```

启动平台：

```powershell
.\scripts\start_platform.bat
```

启动脚本会在 `8501` 至 `8510` 中选择第一个空闲端口，并在终端打印地址。
服务器本机访问 `http://localhost:实际端口`；同一局域网的电脑访问
`http://服务器IP:实际端口`。

平台概览显示数据集、攻击方法、已保存图数和检测服务状态。界面中的数据集名称
为 Cora、Citeseer、Polblogs、Pubmed；底层档案 ID 和已保存的模型路径保持兼容。

攻击检测支持在线生成与多文件上传。选择攻击强度 `0 · 干净图` 时，直接将该
数据集已有的干净图交给检测器判断，只检测一次，不复制成多张样本。非零强度
在线生成的图保存到 `artifacts/platform/generated/`，不会加入训练、验证或测试
清单，并会排除与该数据集清单中已有图指纹相同的结果。

准确率评测按数据集切换，指标、攻击明细、对比图和 CSV 导出均仅对应所选数据集。
测试准确率是该数据集判断正确图数除以全部测试图数；攻击图检出率是被正确识别
的攻击图数除以攻击图数。
每种攻击每个数据集的默认测试计划仍是 20 张（4 个强度各 5 张）；不足或额外
的历史记录按实际数量显示。旧页面按攻击方法合并不同数据集，可能把 20 张和
另一数据集的 8 张显示为 28 张。现在分数据集展示，不会修改或隐藏真实样本数。

旧报告若有匹配的逐图预测记录，平台会自动读取分数据集结果。若页面提示缺少
明细，只需重新计算报告，不用重新生成图或训练：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) "src"
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli evaluate-detector --config configs/base.yaml
```

不打开网页也可以查看真实测试结果：

```powershell
.\.venv\Scripts\python.exe .\show_results.py
```

也可以在 PyCharm 中右键根目录的 `show_results.py`，选择“运行”。程序读取
`artifacts/reports/universal-detector-evaluation.json`，显示综合准确率、每个
数据集的准确率以及每种攻击的实际检出率。
`run_selected_attacks.bat` 成功结束时也会自动打印同一份结果。

## 高级分步命令

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli check-config
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli prepare
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli prepare-detector-data
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli rebuild-detector --run-config configs/run.yaml
# 如需单独评测候选模型，显式指定模型和数据集：
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli evaluate-detector --model artifacts/models/shared-gcn-runs/实际版本号/detector.pt --profiles cora-lcc
```

检测单张图：

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli detect-graph `
  --graph path\to\graph.npz
```

输入支持 `.pt` 和 `.npz`。完整NPZ包含 `x`（float32）和 `edge_index`（int64），
`y`可省略且不会进入检测模型。无属性时 `x` 为 `[N, 0]`；PT字典也可用 `num_nodes` 替代 `x`。
无向边双向存储，无自环/重复边。旧压缩PT仅存边（特征攻击另存修改后属性），必须在
有原始配套文件的项目中解码一次；CSR-only NPZ也不具备完整属性，不能当作完整输入。
用 `export-graph --graph 原图.pt --output 新完整图.npz` 可导出不依赖参考文件的完整图。

平台“攻击检测 → 上传图文件”固定先选择来源数据集，再上传文件。完整 PT/NPZ 使用
文件自身的节点属性；SciPy CSR-only NPZ 使用所选数据集 `clean.pt` 中的可信节点属性，
再与上传的扰动邻接矩阵组成完整图。CSR 邻接矩阵节点数必须与所选数据集完全一致，
平台不会按节点数猜测来源，也不会自动裁剪或重排节点。不同数据集的文件请分批上传。

## 增加数据集

在 `configs/base.yaml` 的 `profiles` 下增加档案，写明数据集名称、DeepRobust
预处理 setting、预期节点数、干净参考路径和探针目录；再把档案 ID 加到
`configs/run.yaml` 的选择列表。将新数据集与希望保留训练覆盖的旧数据集一起选择，
复用已有攻击、重提取描述符，重新训练并验证一个共享模型，然后明确激活。

平台可以读取合法的未知来源完整图，但这不意味着已验证跨未知数据集的泛化能力。
来源不明时显示 unknown，不能凭节点数量猜数据集，也不能在没有真实攻击标签时计算准确率。

## 目录

```text
configs/                         配置和数据集档案声明
src/gnn_attack_detector/         攻击、探针、语料、检测器和平台代码
artifacts/profiles/              干净参考图和每数据集探针
artifacts/attacks/corpus/        可恢复、可复用的攻击图语料
artifacts/features/shared-gcn-v4/  每图12维节点描述符与4维统计，磁盘按需读取
artifacts/features/manifests/    每数据集独立的攻击图清单
artifacts/models/                一个激活检测器及多个候选/备份版本
artifacts/reports/               最新总体报告、逐数据集报告与版本历史
scripts/                         Windows 安装、训练和启动入口
```

## 项目团队

知识智能与安全计算研究组（吴涛团队）
