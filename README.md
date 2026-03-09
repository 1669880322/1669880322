# PINN for BTA Deep-Hole Boring Bar Reduced Lateral Vibration Model

本项目是一个可直接运行的 Python + PyTorch 科研原型，用于在**单点位移观测**条件下训练 PINN，并反演内切削液无量纲平均流速 `u0`。

## Project Structure

- `main.py`: 训练入口（默认 smoke_test）。
- `config.py`: 所有无量纲参数、网络配置、训练配置、参数约束集中管理。
- `equations.py`: 二阶降阶动力学方程系数与观测算子。
- `generate_dataset.py`: synthetic 数据生成（`train_data.csv` 仅含 `t,w_obs`）。
- `data_loader.py`: 数据和元数据加载。
- `model.py`: PINN MLP（支持可切换 Fourier Features）。
- `losses.py`: `L_data + L_phys + L_ic` 计算。
- `trainer.py`: Adam + LBFGS 训练与 PGD 投影（`clamp_`）。
- `evaluate.py`: 评估 `w_mse`、`u0` 相对误差。
- `plot_results.py`: 损失曲线、`u0` 演化、观测对比图、可选隐藏模态对比图。
- `requirements.txt`: 依赖。

## Methodology

1. 网络输入仅为无量纲时间 `t`，输出是隐变量模态坐标 `q1_hat(t), q2_hat(t)`。
2. 物理约束来自二自由度耦合 ODE 残差：
   - `r1(t)` 和 `r2(t)` 由自动微分得到 `q_dot, q_ddot` 后代入方程构造。
3. 数据监督来自单点观测算子：
   - `w_hat(t) = sin(pi*xi_obs)*q1_hat(t) + sin(2*pi*xi_obs)*q2_hat(t)`。
4. 数据损失仅作用于 `w_hat` 与 `w_obs`：
   - `L_data = MSE(w_hat, w_obs)`。
5. `q1_true, q2_true` 不可作为监督标签，因为它们是实验不可直接测量的隐式广义坐标；直接监督会造成信息泄漏并偏离真实实验可观测性。
6. 本项目采用“隐变量 + 观测算子 + ODE 约束”双路拓扑：
   - 路径 A：ODE 物理残差约束；路径 B：观测域数据拟合。
7. 默认观测点 `xi_obs=0.33`，不能默认取 `0.5`，否则二阶模态 `sin(2*pi*xi_obs)` 为 0 导致不可观测。
8. 单点观测反演 `u0` 存在可辨识性风险（多参数耦合、局部最优、噪声敏感）；可通过多测点/多工况联合训练、先验约束、贝叶斯不确定性评估等改善。

## Dynamics and Inversion Setup

- 所有无量纲参数已在 `config.py` 硬编码，训练与数据生成共用。
- 仅反演一个参数 `u0`，并通过 PGD 方式每步后执行：
  - `u0_param.clamp_(u0_min, u0_max)`。
- 约束区间：`[2.0, 5.0]`，初值 `2.5`，真值 `3.5`。

## Data Files

运行 synthetic 生成后会产生：

- `data/train_data.csv`: 仅 `t,w_obs`（训练唯一监督源）。
- `data/hidden_truth.csv`: `q1_true,q2_true,q1_dot_true,q2_dot_true`（仅后验可视化验证，严禁用于训练损失）。
- `data/metadata.json`: `w0,w0_dot,u0_true,xi_obs,time_start,time_end,dt` 等。

## Quick Start

```bash
pip install -r requirements.txt
python main.py --profile smoke_test --regenerate
```

默认会输出：
- `logs/train_log.csv`
- `checkpoints/best_model.pt`
- `results/loss_total.png`
- `results/loss_components.png`
- `results/u0_evolution.png`
- `results/w_obs_vs_w_hat.png`
- `results/q1_true_vs_q1_hat.png`（可选验证）
- `results/q2_true_vs_q2_hat.png`（可选验证）
- `results/metrics.json`

## Switch to Full Training

```bash
python main.py --profile full_train
```

## Replace Synthetic with Experimental Single-point Displacement

1. 准备实验数据文件 `train_data.csv`（列名：`t,w_obs`）。
2. 准备 `metadata.json`（至少包含 `w0`，可选 `w0_dot`）。
3. 运行：

```bash
python main.py --data-mode file --train-csv path/to/train_data.csv --metadata path/to/metadata.json
```

> 注意：即使有后处理估计的模态，也不要将其用于训练监督。
