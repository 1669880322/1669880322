# BTA 二阶主模态 Physics-Informed Inverse Identification (PyTorch)

本项目是一个可直接运行的科研原型：面向 **BTA 深孔镗杆横向振动二阶降阶模型**，执行基于整段振幅序列的结构阻尼参数反演（`c_struct`）。

## Environment Compatibility

- 本项目按 **PyTorch 2.9.1+cu128** 兼容写法实现。
- 若你本地已安装兼容版本（如 `torch==2.9.1+cu128`、`torchvision==0.24.1+cu128`、`torchaudio==2.9.1+cu128`），**无需重新安装 torch 相关包**。
- `requirements.txt` 不锁定 torch 版本，避免覆盖你现有环境。
- 若需重建环境，建议与上述版本保持一致。

## 设计要点

1. **只保留二阶主方程**：主训练与主反演都只用
   \[
   \ddot q_2 + c_2 \dot q_2 + k_2 q_2 + \kappa k_2 q_2^3 = 0
   \]
   不再使用一阶 + 二阶联合训练框架，降低任务耦合复杂性。
2. **输入不是时间点 t，而是整段振幅序列** `amplitude_sequence=[a(t1),...,a(tN)]`，网络做 `sequence -> scalar` 回归。
3. **默认辨识参数为结构阻尼 `c_struct`**，不切换到流体参数或刚度参数，减少病态性并贴合当前课题目标。
4. **硬编码参数集中管理在 `config.py`**，作为 v1 标准测试基准；后续可替换为更严格物理换算值。
5. **完整系数组装保留**：`c1, c2, c11, c22, g1, k1, k2` 全部实现并记录来源；主训练只用 `c2,k2`，其余用于完整降阶背景可追溯性。
6. **1.5–5 um 振幅范围**：由动力学解 `q2(t)` 先生成，再用统一 `scale_um` 做物理一致量纲缩放，不扭曲单样本形状。
7. **第一版固定初值** `q2(0), q2_dot(0)`，降低参数-初值耦合病态，把重点集中到阻尼参数反演。
8. **与经典 PINN 教学示例的区别**：不是 `t->q(t)` 映射，而是序列编码 + 可微积分重构闭环。

## 项目结构

```text
.
├── main.py
├── config.py
├── requirements.txt
├── docs/
│   └── modeling_notes.md
├── src/
│   ├── utils.py
│   ├── losses.py
│   ├── evaluate.py
│   ├── plot_results.py
│   ├── data/
│   │   ├── generate_amplitude_dataset.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── sequence_encoder.py
│   │   └── physics_informed_identifier.py
│   ├── physics/
│   │   ├── second_mode_ode.py
│   │   └── differentiable_integrator.py
│   └── trainers/
│       └── trainer.py
├── data/
├── logs/
├── checkpoints/
└── results/
```

## 运行命令

### 1) 冒烟测试
```bash
python main.py --profile smoke_test --device cuda --regenerate
```

### 2) 完整训练
```bash
python main.py --profile full_train --device cuda --regenerate
```

### 3) 仅测试
```bash
python main.py --profile full_train --device cuda --test-only --checkpoint path/to/ckpt.pt
```

> 若 `cuda` 不可用，程序自动回退到 `cpu`。

## 真实实验数据替换说明

当替换为真实实验振幅数据时：
1. 替换 `src/data/data_loader.py` 的数据来源（保留 `amplitude_sequence` 与 `time_grid` 接口）。
2. 将 `loss_mode` 设为 `physics_only` 或 `hybrid`，并可关闭 `Lparam` 监督。
3. 保留 `src/physics/second_mode_ode.py` 与 `src/physics/differentiable_integrator.py` 的可微闭环重构。

## 输出文件

- `results/loss_curve.png`
- `results/component_losses.png`
- `results/c_est_evolution.png`
- `results/amplitude_fit_examples.png`
- `results/sample_reconstruction.png`
- `results/metrics.json`
- `results/synthetic_truth.json`
- `logs/train_history.json`
- `logs/coefficient_sanity.json`
- `checkpoints/best_<profile>.pt`

