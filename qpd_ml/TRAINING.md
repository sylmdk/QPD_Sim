# QPD 模型训练与测试

本目录包含独立的 PyTorch Lightning 训练与测试模块。模型输入为 QPD RAW，输出为线性相机 RGB `clean_energy_field`。

## 安装依赖

在项目根目录运行：

```powershell
python -m pip install -r qpd_ml/requirements-training.txt
```

LPIPS 首次运行会下载所选主干网络的预训练权重。

## 配置文件

配置按用途拆分：

- `qpd_ml/configs/train.yaml`：训练、验证、TensorBoard 和 checkpoint。
- `qpd_ml/configs/test.yaml`：独立指标测试和完整尺寸重建输出。

### 训练配置

`train.yaml` 包含以下部分：

- `experiment`：实验名称、输出目录和随机种子。
- `data`：数据位置、`data_split`、裁块大小、batch size 和 DataLoader 参数。`data_split: train` 对应读取 `train.csv`；训练验证固定读取同目录的 `val.csv`。
- `model`：U-Net 通道、深度、残差块、归一化、激活、dropout 和上采样方式。
- `optimizer`：AdamW 和学习率调度参数。
- `loss`：L1/MSE 损失权重。
- `metrics`：PSNR、SSIM、LPIPS 的计算域。
- `visualization`：TensorBoard 验证图像。
- `checkpoint`：最优模型保存规则。
- `trainer`：epoch、设备和精度。

运行训练：

```powershell
python -m qpd_ml.train --config qpd_ml/configs/train.yaml
```

训练结束后 checkpoint 默认位于：

```text
outputs/qpd_training/qpd_unet/version_x/checkpoints/
```

### 指定 GPU 与多卡训练

单卡并指定 GPU 2：

```yaml
trainer:
  accelerator: gpu
  devices: [2]
  strategy: auto
```

指定 GPU 0 和 GPU 2 进行 DDP 多卡训练：

```yaml
trainer:
  accelerator: gpu
  devices: [0, 2]
  strategy: ddp
```

Lightning 会为每张卡创建独立训练进程并自动使用分布式采样器；代码会跨进程同步验证指标，TensorBoard 验证图像只由主进程写入。

### 中断后恢复训练

自动寻找当前实验目录中最新的 `last.ckpt`：

```yaml
resume:
  enabled: true
  checkpoint: auto
```

也可以明确指定 checkpoint：

```yaml
resume:
  enabled: true
  checkpoint: outputs/qpd_training/qpd_unet/version_0/checkpoints/last.ckpt
```

恢复训练会继续模型权重、优化器、学习率调度器、epoch、global step 和 callback 状态。如果 `experiment.version` 为 `null`，程序会从 checkpoint 路径识别原来的 `version_x`，继续写入同一个 TensorBoard 实验目录。`trainer.max_epochs` 表示恢复前后总 epoch 上限，而不是额外训练的 epoch 数；例如 checkpoint 已完成 20 epoch，想继续到 50 epoch，应设置 `max_epochs: 50`。

## 独立测试与完整重建

在 `test.yaml` 中设置 checkpoint：

```yaml
test:
  checkpoint: outputs/qpd_training/qpd_unet/version_0/checkpoints/last.ckpt
  run_metrics: true
```

也可以通过命令行临时指定：

```powershell
python -m qpd_ml.test --config qpd_ml/configs/test.yaml --checkpoint outputs/qpd_training/qpd_unet/version_0/checkpoints/last.ckpt
```

### 指标测试

`test.run_metrics: true` 时，通过 `data.data_split` 对应的 CSV 加载目标数据并输出：

- `test_loss`
- `test_l1`
- `test_psnr`
- `test_ssim`
- `test_lpips`

所有测试指标均由同一张完整尺寸预测和完整尺寸目标计算，不再使用测试裁块。该完整预测也同时用于保存 NPY 和 PNG，因此指标评价内容与输出内容严格一致。

如只需要生成重建结果，可设置：

```yaml
test:
  run_metrics: false
```

### 完整尺寸输出

```yaml
inference:
  enabled: true
  sample_ids: []
  output_dir: outputs/qpd_reconstruction
  save_clean_energy_field: true
  save_srgb_png: true
```

- `enabled`：是否执行完整尺寸重建。
- 测试指标和完整重建都读取 `test.yaml` 中 `data.data_split` 指定的 CSV；默认 `data_split: test` 对应 `test.csv`。
- `sample_ids`：空列表表示处理该 split 的全部样本；也可填写指定 ID。
- `save_clean_energy_field`：保存 `H×W×3 float32` 线性相机 RGB。
- `save_srgb_png`：保存经过每图 WB、CCM 和 sRGB gamma 的 8-bit RGB PNG。

整张 QPD RAW 会一次性打包成 `[1, 4, H/2, W/2]` 输入模型，不进行裁块或拼接。

每个样本输出：

```text
outputs/qpd_reconstruction/<sample_id>/
├── clean_energy_field_pred.npy
├── reconstructed_srgb.png
└── reconstruction.json
```

输出根目录还会生成 `reconstruction_summary.json`。

### 指定 GPU 与多卡测试

在 `test.yaml` 中使用与训练相同的设备写法：

```yaml
trainer:
  accelerator: gpu
  devices: [0, 2]
  strategy: auto
```

测试时每个 GPU 进程独立加载一次 checkpoint，`test.csv` 中的完整尺寸样本按 GPU 数量分片并行处理，最后由主进程合并指标与 `reconstruction_summary.json`。单个样本始终在一张 GPU 上进行完整尺寸推理；因此测试样本数大于 1 时才能从多卡获得并行收益。

## sRGB 比较域

当 `metrics.domain: srgb` 时，预测和目标会根据每张图 metadata 中的 `wb_gains` 和 `ccm_srgb_from_cam` 转换到线性 sRGB，再应用 IEC 61966-2-1 传递函数。也可设置 `camera_linear`，直接在线性相机 RGB 域计算指标。

训练损失始终在 `clean_energy_field` 的线性相机 RGB 域计算。

## TensorBoard

```powershell
tensorboard --logdir outputs/qpd_training
```

标量包含 loss、L1、PSNR、SSIM、LPIPS 和学习率。验证图像排列为：

```text
QPD 输入预览 | 模型预测 | 目标图像 | 4×绝对误差
```

`qpd_dev10` 只有 8 个训练样本、1 个验证样本和 1 个测试样本，主要用于流程验证。
