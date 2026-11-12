# 共享GCN检测器：服务器迁移与运行说明

## 1. 这次到底改了什么

现在是**一个共享GCN编码器与分类头联合训练**，而不是每个数据集训练一个攻击检测器。

完整图的边与节点属性 → 12维通用节点描述符 → GCN 12→32→32 → 全图均值/最大值/95%分位数池化（96维）→ 拼接4维一致性统计 → 分类头 → 一个攻击分数和判断。

12维包括：log度数、相对度数、孤立点标记，邻居log度数均值/标准差/最大值，邻接节点属性余弦相似度均值/标准差/最小值/最大值，有效属性邻居标记及比例。4维是有效边余弦相似度均值、标准差、5%分位数和有效边比例。

借鉴ProGNN的**属性平滑性/结构与属性一致性**思想：相连节点的内容是否相容，作为检测信号交给学习器判断。没有运行完整ProGNN，没有低秩分解或修复图优化，也没有宣称不相似的边必然是攻击。GCN-Jaccard、SimPGCN不再进入检测流程；它们只保留在单独的防御评测中。

图的节点数与原始属性维度可以不同。`y`（节点类别）、攻击名称、数据集名称、干净参考图都不进入检测模型。来源名称只用于分组评测；上传来源未知时显示unknown，模型不猜来源。能够读取未知数据集，不代表已经验证其泛化准确率。

属性缺失/零向量、单位矩阵、常量占位特征会屏蔽，Polblogs默认无内容属性。非标准人工属性不能仅靠程序可靠推断，需用`none`说明。有效性标记本身会进入通用描述符。

## 2. 替换服务器哪些文件

先停止平台和本项目正在执行的Python任务，备份服务器项目。不要结束其他人的进程。

在服务器原项目目录内，按相同相对路径替换以下文件：

```text
src/gnn_attack_detector/cli.py
src/gnn_attack_detector/config.py
src/gnn_attack_detector/detector_features.py
src/gnn_attack_detector/detector_dataset.py
src/gnn_attack_detector/detector_model.py
src/gnn_attack_detector/pipeline.py
src/gnn_attack_detector/graph_io.py
src/gnn_attack_detector/workflow.py
src/gnn_attack_detector/preparation.py
src/gnn_attack_detector/probes.py
src/gnn_attack_detector/platform_service.py
src/gnn_attack_detector/result_viewer.py
src/gnn_attack_detector/web_app.py
scripts/train_detector.ps1
configs/base.yaml
```

`configs/base.yaml`若在服务器有自行调整过的路径、攻击配额等，先保留这些设置，合并下面的检测器配置变化：

```yaml
detector:
  attribute_modes:
    polblogs-lcc: none
  nonattack_manifest: null
  model:
    type: shared_gcn_reference_free
    hidden_channels: 32
    # 其余dropout、epochs、patience、学习率等参数保留。
```

同时保留 `runtime.device: auto`。不要复制本机`.venv`到服务器，不需要重装已可用的CUDA版PyTorch；本次未增加依赖。

建议同步测试和文档：

```text
tests/test_detector_core.py
tests/test_reference_free_detector.py
tests/test_selected_workflow.py
tests/test_universal_contract_source.py
README.md
docs/shared-gcn-migration.md
```

旧检测器的252维参考差异计算、旧MLP训练与缓存读取代码已在原文件中移除并替换；旧探针输出里仅供旧检测器使用的embedding冗余字段也移除了。没有删除攻击实现、防御评测实现或任何已有图数据。无需额外手工删除Python文件。

本地修改前快照在 `output/shared-gcn-prechange/`，包含原src/tests/configs/scripts/README。不要把这个快照当成新代码覆盖服务器。

**必须保留服务器已有的：**

- `artifacts/attacks/corpus/`：昂贵的已生成扰动图。
- `artifacts/features/manifests/`：攻击图清单。
- `artifacts/profiles/`：旧压缩攻击文件省略属性，需要这些原始属性来解码；攻击生成和防御评测也仍使用它。
- `artifacts/models/`、`artifacts/reports/`：旧模型与历史评测。
- 原`.venv`与服务器自己选择的`configs/run.yaml`。

旧特征缓存、旧MLP模型不会被新检测器加载。它们暂时保留用于恢复，不需要删掉，更不需要为此重新生成攻击图。

## 3. 已有攻击图，怎么训练新检测器

在服务器PowerShell中：

```powershell
Set-Location -LiteralPath '<服务器上的项目根目录>'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli check-config
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
notepad .\configs\run.yaml
```

在`profiles`里选本次希望参与训练的所有数据集；在`attacks`里选已有、希望用于训练评测的攻击。保留`train/val/test`三个划分。例如四数据集联合训练就取消四个profile前的注释。**只选Cora就只训练Cora，不自动混入此前准备好的其他数据集。**

```powershell
.\scripts\train_detector.ps1
```

等价的直接命令（PyCharm终端也能运行）：

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli rebuild-detector --config configs/base.yaml --run-config configs/run.yaml
```

这个命令只读取已存在的攻击文件，不调用攻击生成算法。缺少清单、文件或必要的训练/验证样本会报错，不会悄悄花几个小时补生成。新输入放在`artifacts/features/shared-gcn-v4/`，逐图从磁盘读取，避免一次把全部图放入内存。

GCN参数和分类头都重新训练；旧MLP权重不能继续用。输入提取、训练、评测仍需要时间，取决于样本数、图规模和设备，不承诺几分钟完成。日志会显示`[DEVICE] detector training: cuda`或`cpu`；CUDA故障会打印`[GPU-FALLBACK]`后用CPU重新训练当前候选。

## 4. 训练后如何更新检测器

每次训练完成后会自动更新平台当前使用的检测器，同时保留一个运行版本，例如：

```text
artifacts/models/shared-gcn-runs/20260914T120000-ab12cd34/detector.pt
```

以程序实际输出的`training.model_path`为准。旧模型会自动备份到`shared-gcn-runs/previous-.../`，不需要再执行激活命令。每个数据集的评测报告保存到`shared-gcn-profiles/<profile>.json`：训练其他数据集不会删除旧报告，再次训练某个数据集时只更新该数据集的报告。

需要新增攻击数据时仍可运行原`run_selected_attacks.bat`：它会生成/复用所选攻击并训练、评测；训练完成后活动检测器会自动更新。

## 5. 各数据集结果如何保存

- 最新一次运行总体结果：`artifacts/reports/universal-detector-evaluation.json`。
- 各数据集最近一次结果：`artifacts/reports/shared-gcn-profiles/<profile>.json`。
- 每次运行报告与逐图预测：`artifacts/reports/shared-gcn-history/<版本>/`。

这次只评测Cora，只更新Cora的分组报告；CiteSeer/PubMed/Polblogs报告保留。平台按数据集读取，并显示评测时间、模型哈希和历史/候选模型提醒。来自不同版本的分组结果不会被拼成某一个模型的总准确率。

不打开平台可运行`show_results.py`查看最新一次整体报告，或打开对应profile的JSON看历史分组结果。

## 6. 上传与单图检测

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli detect-graph --graph '完整图.npz'
```

无需`--profile`，也无需节点类别。NPZ内容：

- `x`：float32矩阵`[N, F]`；没有内容属性可用`[N, 0]`。
- `edge_index`：int64矩阵`[2, E]`；无向边两个方向都要有，无自环和重复边。
- `y`：可选，若携带须形状正确，但检测器不读取它。
- `attribute_mode`：可选字符串`auto/content/none`，用于保留属性语义。

也支持标准ZIP版`torch.save`的张量字典，以及受限白名单内的PyG Data；不接受可执行模型、任意Python对象或旧raw-pickle/tar格式。推荐交付NPZ。PyTorch旧版本仍应仅接收可信渠道文件，不应把开发平台裸露到公共互联网。

为限制上传引发的资源耗尽，PT/NPZ展开后最多512MB，图最多20万节点、500万条有向边。NPZ在读取数组前检查ZIP大小和NPY声明尺寸；不能用高压缩率绕过限制。当前四个数据集在这些限制内。

旧版压缩PT不是完整输入：它通常只保存边，省略了没改变的属性。可以在原项目内直接检测，程序通过其元数据找到原始属性并校验来源；这是**补齐文件内容**，不是把原图和待检图的差值交给检测器。

要发给别人独立检测，先导出：

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli export-graph --graph 'artifacts/attacks/corpus/原图.pt' --output 'output/完整扰动图.npz'
```

导出文件包含实际待检图的属性与边，以及必要的属性语义标记，后续检测不依赖原项目干净参考文件。输出文件必须不存在，以免误覆盖。来源未知且属性是人工占位时，可显式加`--attribute-mode none`；这属于输入语义说明，不是数据集身份。

## 7. 正式测评之前最重要的限制

目前每个数据集通常只有一张原始干净图。如果它同时出现在训练、验证、测试中，它只能算**重复基准对照**，不能算独立干净测试样本。默认结果会标明这一点，独立干净测试计数为0，不能据此宣称正式准确率或对任意图泛化。

可将`detector.nonattack_manifest`设为项目内JSONL路径，接入真实非攻击图。每行例如：

```json
{"sample_id":"clean-external-001","dataset_id":"cora-lcc","partition":"test","label":0,"graph_path":"artifacts/nonattack/001.npz","source_group":"independent-collection-001","independent_clean":true,"independence_evidence":"独立采集批次、来源记录或审核说明"}
```

此示例不是生成新样本的指令；必须提供实际文件、真实来源与相互独立的训练/验证/测试数据。源组不能跨划分，一个独立源组最多计一个干净测试单位。若不能证明确实独立，设置`independent_clean:false`，不得以细微修改、重新编号或更换文件名伪造独立样本。程序的重复检查无法替代真实来源审核。

提供非攻击清单后，所选数据集的各划分都需有非攻击样本。模型检查点记录参与拟合的内容指纹与源组，后续评测会阻止它们作为独立测试输入。正式指标排除重复对照，逐数据集也单独记录资格；总体通过还要求所评测各数据集具备独立干净测试资格。

## 8. 本地与服务器验证

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m pytest -q
```

本机已运行CPU单元/集成测试（包括小图真实联合训练、无参考推理、选定数据集与历史报告、上传校验和模拟CUDA故障回退）。本机没有CUDA，因此真实CUDA测试会跳过；服务器应再运行测试并确认CUDA测试实际执行。四个真实数据集的新模型性能尚需在服务器训练后测量，不能用小图测试代替实测准确率。
