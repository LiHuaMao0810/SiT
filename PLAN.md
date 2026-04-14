# FM-GRPO 工程文档 v3
## Reward-Guided Flow Matching Distillation via ELBO Policy Gradients

> 基于 v3 理论文档实现。Old policy = Teacher（固定），探索样本来自 Teacher 少步 ODE，
> MSE 目标来自 Teacher 高质量 ODE（完全解耦）。

---

## 任务定义

```
Teacher：SiT-XL/2，T_exp 步 ODE，提供探索样本（reward 来源）
         SiT-XL/2，T_hq  步 ODE，提供高质量样本（MSE 蒸馏目标）
Student：SiT-XL/2，1步  ODE，从 Teacher 权重初始化
Old policy：Teacher（永远固定，充当 ELBO ratio 的分母）

目标：Student 1步生成质量逼近 Teacher 多步质量
```

**关键设计原则**（来自 v1 2D 实验验证）：
- Teacher 作为 old policy：永不漂移，trust region 语义严格成立
- 探索样本来自 Teacher 而非 Student：避免 ratio 结构性偏向 > 1
- GRPO 探索与 MSE 蒸馏完全解耦：独立采样，梯度互不干扰
- Phase 1/2 分离：Phase 1 数据冻结，Phase 2 复用 K 次

---

## 目录结构

```
FM_GRPO/SiT/
├── train.py                     # 原始脚本，不修改
├── models.py                    # 原始模型，不修改
├── transport/
│   └── path.py                  # 【微改】增加 per_sample_loss
│                                #         （其余不动）
│
├── grpo_utils.py                # 【新建】ELBO ratio / advantage / PPO-clip
├── reward_utils.py              # 【新建】Reward（ResNet classifier）
├── sit_sampler.py               # 【新建】Teacher/Student ODE 采样封装
│
├── train_elbo_grpo.py           # 【新建】主训练脚本（本方法）
├── train_sde_grpo.py            # 【新建】SDE-GRPO baseline（对比实验）
├── train_distill_only.py        # 【新建】纯蒸馏 baseline
├── evaluate.py                  # 【新建】评估脚本
│
├── pretrained_models/
│   └── SiT-XL-2-256x256.pt     # 已下载
└── results/
    ├── elbo_grpo/
    ├── sde_grpo/
    └── distill_only/
```

---

## 模块详细说明

---

### 1. `transport/path.py`（微改，+15 行）

在现有 `training_losses` 基础上新增 `per_sample_loss`，供 ELBO ratio 计算使用。

**实现要求**：
- 先读懂 `training_losses` 的插值逻辑（alpha_t, sigma_t, target 计算），**复用**，不要重写
- 返回 `(B,)` 的 per-sample loss，不做 batch mean
- 支持传入外部 `t` 和 `noise`，若为 None 则内部采样

```python
def per_sample_loss(self, model, x1, model_kwargs, t=None, noise=None):
    """
    计算每个样本的 CFM loss，返回 shape (B,)。
    
    Args:
        model:         SiT 模型（teacher 或 student）
        x1:            clean sample（数据端），shape (B, 4, 32, 32)
                       注意：SiT 约定 tau=1 是数据端，tau=0 是噪声端
        model_kwargs:  dict(y=class_labels)
        t:             flow timestep，shape (B,)，若 None 则内部 U(0,1) 采样
        noise:         高斯噪声，shape (B, 4, 32, 32)，若 None 则内部采样
    
    Returns:
        loss: shape (B,)，每个样本的 ||v_pred - u||^2 (在空间维度平均)
    
    实现步骤（参考 training_losses）：
        1. 如果 t/noise 为 None，内部采样
        2. 计算插值：x_t = alpha_t * x1 + sigma_t * noise
           （alpha_t, sigma_t 来自 transport 的 path 定义，OT schedule: alpha=tau, sigma=1-tau）
        3. 计算条件速度场真值：u = x1 - noise
           （OT-CFM 下 u(x_tau|x1) = x1 - epsilon）
        4. 模型前向：v_pred = model(x_t, t, **model_kwargs)
        5. loss = ((v_pred - u) ** 2).mean(dim=[1, 2, 3])  # (B,)，不做 batch mean
    
    注意：这个函数可能被 teacher（no_grad）和 student（with_grad）分别调用，
    调用方负责 no_grad 上下文，这里不加 @torch.no_grad()
    """
    ...
```

---

### 2. `grpo_utils.py`（新建，约 80 行）

```python
"""
ELBO ratio、advantage 计算、PPO-clip loss。

核心设计：
- compute_elbo_ratio: teacher 作为 old policy，teacher 的 ell_T 在 Phase 1 预计算并冻结
- compute_grpo_advantage: 组内归一化，支持 advantage 为 0（reward std 极小时）
- ppo_clip_loss: 标准 PPO-clip
"""
import torch
import torch.nn.functional as F


def compute_elbo_ratio(student, x1_exp, model_kwargs, transport,
                       tau_mc, eps_mc, ell_T_cached):
    """
    计算 ELBO ratio: exp(mean_j(ell_T - ell_eta))
    
    关键设计：ell_T 在 Phase 1 预计算，此处只算 student 的 ell_eta。
    teacher 和 student 使用完全相同的 (tau_mc, eps_mc, x1_exp)。
    
    Args:
        student:        当前 student（训练中，requires_grad=True）
        x1_exp:         Phase 1 教师探索终点，shape (BG, 4, 32, 32)，冻结
        model_kwargs:   dict(y=cond_g)，class label
        transport:      SiT transport 对象
        tau_mc:         Phase 1 预采样的时间步，shape (BG, N_mc)，冻结
        eps_mc:         Phase 1 预采样的噪声，shape (BG, N_mc, 4, 32, 32)，冻结
        ell_T_cached:   Phase 1 预计算的 teacher loss，shape (BG, N_mc)，冻结
    
    Returns:
        log_ratio: shape (BG,)，log r^i
        ratio:     shape (BG,)，exp(clamp(log_ratio, -5, 5))
    
    实现步骤：
        for j in range(N_mc):
            x_tau_j = tau_mc[:,j] * x1_exp + (1-tau_mc[:,j]) * eps_mc[:,j]
            ell_eta_j = transport.per_sample_loss(student, x1_exp, model_kwargs,
                                                   t=tau_mc[:,j], noise=eps_mc[:,j])
        log_ratio = (ell_T_cached - ell_eta).mean(dim=-1)  # (BG,)
        ratio = log_ratio.clamp(-5, 5).exp()
    
    注意：
    - x1_exp.detach() 确保不对探索样本求梯度
    - ell_T_cached 已经是 no_grad 的，直接用
    - 返回 log_ratio 用于 early stopping 判断，返回 ratio 用于 PPO-clip
    """
    ...


def compute_grpo_advantage(rewards, adv_clip=2.0):
    """
    GRPO 组内 advantage 归一化。
    
    Args:
        rewards:   shape (B, G)，每组 G 个 reward
        adv_clip:  advantage 截断值，默认 2.0
    
    Returns:
        advantage: shape (B, G)
    
    实现：
        R_mean = rewards.mean(dim=1, keepdim=True)   # (B, 1)
        R_std  = rewards.std(dim=1, keepdim=True)    # (B, 1)
        
        # 当组内 reward 全相同时（std ≈ 0），advantage 设为 0，不更新
        advantage = torch.where(
            R_std < 1e-8,
            torch.zeros_like(rewards),
            (rewards - R_mean) / (R_std + 1e-8)
        ).clamp(-adv_clip, adv_clip)
    
    注意：advantage 可以为负（惩罚低 reward 样本），不要 relu
    """
    ...


def ppo_clip_loss(ratio, advantage, eps_clip=0.05):
    """
    PPO-clip surrogate loss。
    
    Args:
        ratio:     shape (BG,) 或 (B, G)
        advantage: shape (BG,) 或 (B, G)，需与 ratio 形状一致
        eps_clip:  默认 0.05（FPO 论文推荐值）
    
    Returns:
        loss: scalar
    
    实现：
        clipped = ratio.clamp(1-eps_clip, 1+eps_clip)
        loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    
    注意：取负号，因为优化器做梯度下降，我们要最大化 policy gradient
    """
    ...
```

---

### 3. `reward_utils.py`（新建，约 60 行）

```python
"""
Reward model 封装。

当前使用 ResNet-50 classifier confidence 作为 reward，
不依赖真实图像，直接对 teacher 探索样本打分。

备选方案（消融用）：
  - LPIPS(x1_exp_pixel, x1_hq_pixel)：不需要外部 reward model
  - CLIP aesthetic score
"""
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T


class ClassifierReward:
    """
    用预训练 ResNet-50 的 softmax confidence 作为 reward。
    reward = p(class=c | image)
    
    在1步蒸馏初期，探索样本质量较低，reward 接近 0，
    随训练进行 reward 逐步上升，是验证训练有效性的主要监控指标。
    """
    
    def __init__(self, device):
        self.device = device
        self.model = tv_models.resnet50(weights='IMAGENET1K_V2')
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    @torch.no_grad()
    def __call__(self, images_pixel, class_labels):
        """
        Args:
            images_pixel: (B, 3, H, W)，值域 [-1, 1]（VAE decode 输出）
            class_labels: (B,) int tensor
        
        Returns:
            rewards: (B,) float，对应类别的 softmax confidence，范围 [0,1]
        
        实现：
            x = (images_pixel * 0.5 + 0.5).clamp(0, 1)
            x = normalize(x)
            x = interpolate(x, 224)  # ResNet 需要 224x224
            probs = softmax(model(x))
            rewards = probs[arange(B), class_labels]
        """
        ...


class LPIPSReward:
    """
    用 LPIPS 相似度作为 reward（备选，无需外部 classifier）。
    reward = -LPIPS(x1_exp, x1_hq)
    
    x1_hq 是 teacher 高质量（T_hq 步）生成结果，与 Phase 1 独立采样。
    相似度越高 reward 越大。
    
    优点：不需要 ImageNet 数据，纯模型内部的自监督信号
    缺点：引入 LPIPS 网络（VGG backbone），增加显存
    """
    
    def __init__(self, device):
        import lpips
        self.fn = lpips.LPIPS(net='vgg').to(device)
        for p in self.fn.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def __call__(self, x1_exp_pixel, x1_hq_pixel):
        """
        Args:
            x1_exp_pixel: (B, 3, H, W)，教师探索样本，值域 [-1,1]
            x1_hq_pixel:  (B, 3, H, W)，教师高质量样本，值域 [-1,1]
        
        Returns:
            rewards: (B,)，-LPIPS 值，越大越好
        """
        dist = self.fn(x1_exp_pixel, x1_hq_pixel).squeeze()
        return -dist
```

---

### 4. `sit_sampler.py`（新建，约 80 行）

```python
"""
SiT ODE/SDE 采样封装。

teacher_ode_sample：教师多步 ODE 采样（Phase 1 探索 + MSE 目标）
student_ode_sample：学生 N 步 ODE 采样（评估用）
"""
import torch


@torch.no_grad()
def teacher_ode_sample(teacher, vae, y, num_steps,
                       cfg_scale=1.5, latent_size=32, device='cuda',
                       return_pixel=False):
    """
    教师 ODE 采样，全程 no_grad。
    
    Args:
        teacher:      SiT teacher 模型（冻结）
        vae:          VAE decoder（冻结）
        y:            class labels, shape (B,)
        num_steps:    ODE 步数（T_exp 或 T_hq）
        cfg_scale:    CFG scale，默认 1.5
        latent_size:  latent 空间大小，默认 32（对应 256px）
        device:       GPU device
        return_pixel: 是否同时返回 pixel 图像（reward 需要时设为 True）
    
    Returns:
        x1_latent: shape (B, 4, 32, 32)，latent 空间生成结果
        x1_pixel:  shape (B, 3, 256, 256)，仅当 return_pixel=True 时返回
    
    实现参考 sample.py 第 85-109 行：
        1. z = randn(B, 4, 32, 32)
        2. CFG double batch：z = cat([z, z])，y = cat([y, y_null])
        3. sample_fn = transport_sampler.sample_ode()
        4. samples = sample_fn(z, teacher.forward_with_cfg, y=y, cfg_scale=cfg_scale)[-1]
        5. samples = samples[:B]  # 去掉 null class 那半
        6. if return_pixel: pixel = vae.decode(samples / 0.18215).sample
    
    注意：
    - 初始噪声 z 在外部固定后传入，保证 teacher_exp 和 student 使用相同初始噪声时可对比
    - 如果不需要固定 z，内部 randn 即可
    """
    ...


@torch.no_grad()
def student_ode_sample(student, vae, y, num_steps=1,
                       cfg_scale=1.5, latent_size=32, device='cuda',
                       z=None):
    """
    学生 ODE 采样，用于评估。
    
    与 teacher_ode_sample 完全相同逻辑，只是 num_steps 默认为 1。
    训练时不调用（Phase 1 探索来自 teacher，Phase 2 不需要完整推理）。
    仅在 log_every 时调用，生成样本图用于可视化。
    """
    ...
```

---

### 5. `train_elbo_grpo.py`（新建，主训练脚本）

完整训练流程，严格按照 v3 理论文档 Section 5/6 实现。

```python
"""
ELBO-GRPO 1步蒸馏训练脚本。

核心流程（每个 iteration）：

  [Phase 1：数据收集，完全 no_grad]

  Step 1：教师探索 ODE（T_exp 步，G 组）
    x0_exp = randn(B*G, 4, 32, 32)
    x1_exp = teacher_ode_sample(teacher, vae, cond_g, steps=T_exp)
                                                  # shape (BG, 4, 32, 32)

  Step 2：Reward + Advantage
    x1_exp_pixel = vae.decode(x1_exp / 0.18215).sample   # (BG, 3, 256, 256)
    R = reward_fn(x1_exp_pixel, cond_g)                   # (BG,)
    R = R.reshape(B, G)
    A = compute_grpo_advantage(R, adv_clip=2.0)           # (B, G)
    A = A.reshape(BG)

  Step 3：ELBO ratio MC 采样点（Phase 2 全程共享，不重新采样）
    tau_mc = rand(BG, N_mc)
    eps_mc = randn(BG, N_mc, 4, 32, 32)
    x_tau_mc = tau_mc * x1_exp + (1-tau_mc) * eps_mc     # (BG, N_mc, 4, 32, 32)
    u_mc = x1_exp - eps_mc                                # 条件速度场真值 (BG, N_mc, 4, 32, 32)
    # 计算并缓存 teacher loss（只算一次）
    ell_T = zeros(BG, N_mc)
    for j in range(N_mc):
        v_T_j = teacher(x_tau_mc[:,j], tau_mc[:,j], cond_g)
        ell_T[:,j] = (v_T_j - u_mc[:,j]).pow(2).mean(dim=[1,2,3])
    # ell_T: (BG, N_mc)，冻结，Phase 2 全程使用

  Step 4：MSE 目标（独立高质量 ODE，与探索完全解耦）
    x0_hq = randn(B, 4, 32, 32)                          # 独立采样，不复用 x0_exp
    x1_hq = teacher_ode_sample(teacher, vae, class_label, steps=T_hq)
    tau_hq = rand(B)
    eps_hq = randn(B, 4, 32, 32)
    x_tau_hq = tau_hq * x1_hq + (1-tau_hq) * eps_hq
    u_hq = x1_hq - eps_hq

  [Phase 2：学生更新，K 次，Phase 1 数据全部冻结]

  for k in range(K):

    Step 5：计算 student ell_eta（有梯度）
      ell_eta = zeros(BG, N_mc)
      for j in range(N_mc):
          v_eta_j = student(x_tau_mc[:,j].detach(), tau_mc[:,j], cond_g)
          ell_eta[:,j] = (v_eta_j - u_mc[:,j].detach()).pow(2).mean(dim=[1,2,3])

    Step 6：ELBO log ratio
      log_r = (ell_T - ell_eta).mean(dim=-1)             # (BG,)

    Step 7：Early stopping（防止 ratio 偏移过大）
      with no_grad:
          mean_r = log_r.clamp(-5,5).exp().mean().item()
      if mean_r > ratio_stop or mean_r < 1/ratio_stop:
          break

    Step 8：Total loss
      r = log_r.clamp(-5,5).exp()                        # (BG,)
      L_grpo = ppo_clip_loss(r, A, eps_clip=0.05)

      v_eta_hq = student(x_tau_hq.detach(), tau_hq, class_label)
      L_mse = F.mse_loss(v_eta_hq, u_hq.detach())

      loss = L_grpo + lambda_mse * L_mse

    Step 9：更新
      optimizer.zero_grad()
      if loss.isfinite():
          loss.backward()
          clip_grad_norm_(student.parameters(), 1.0)
          optimizer.step()

  [Logging]
  每 log_every 步记录：
    - reward 均值/标准差
    - mean_ratio（应接近 1）
    - clip_fraction（ratio 被 clip 的比例）
    - K_actual（实际执行了几次更新，early stop 触发时 < K）
    - L_grpo, L_mse
    - 生成样本图（student 1步推理，存 png）
"""
```

**超参**：

| 超参 | 推荐值 | 说明 |
|------|--------|------|
| `T_exp` | 5 | 教师探索步数，低质量高多样性 |
| `T_hq` | 50 | MSE 目标步数（50步 vs 250步 FID 差不多，但快5倍） |
| `G` | 8 | group 大小，每个 condition 生成 G 条探索轨迹 |
| `K` | 4 | Phase 2 学生更新次数 |
| `N_mc` | 4 | ELBO ratio MC 采样数 |
| `eps_clip` | 0.05 | PPO clip（FPO 最优值） |
| `adv_clip` | 2.0 | advantage 截断 |
| `lambda_mse` | 0.1 | MSE 权重（初始值，消融调节） |
| `ratio_stop` | 1.5 | early stopping 阈值 |
| `lr` | 1e-4 | 学习率 |
| `batch_size` | 4 | B（单 GPU，显存限制） |
| `cfg_scale` | 1.5 | CFG scale，训练推理一致 |
| `device` | `cuda:2` | 指定 GPU（GPU 0,1 被占用） |

**注意**：`T_hq=50` 而非 250，原因是 SiT 50 步 ODE 质量已经很接近 250 步（FID 差 < 0.5），但速度快 5 倍，Phase 1 的显存和时间开销更合理。如果资源充足可以改为 250。

---

### 6. `train_sde_grpo.py`（新建，SDE-GRPO baseline）

与 `train_elbo_grpo.py` **结构类似**，但 baseline 严格对齐原始 Flow-GRPO：

- old policy 不是 teacher，而是 `policy_old = deepcopy(student)` 的快照
- Phase 1 探索由 `policy_old` 进行 SDE rollout，并累计 `log_p_old`
- Phase 2 用当前 `student` 在**同一条噪声轨迹**上重算 `log_p_new`
- ratio: `log_r = log_p_new - log_p_old`
- 每隔 `old_policy_update_freq` 步更新一次 `policy_old`

**差异点（ELBO vs SDE baseline）**：

```python
# ── train_elbo_grpo.py（主方法）──
log_r = (ell_T - ell_eta).mean(dim=-1)   # ELBO ratio，teacher 做 old policy

# ── train_sde_grpo.py（baseline, Flow-GRPO style）──
# policy_old rollout: x1_exp, log_p_old, saved_noises = sde_sample_with_logprob(policy_old, ...)
# policy new recompute: log_p_new = sde_logprob_recompute(student, z_exp, saved_noises, ...)
# log_r = log_p_new - log_p_old
```

SDE log prob 的计算与噪声复用参考 `sampler_sde.py`（见下）。

---

### 7. `sampler_sde.py`（新建，SDE-GRPO baseline 专用）

```python
"""
SDE 采样 + log prob 计算，仅 SDE-GRPO baseline 使用。

核心函数：
  - `sde_sample_with_logprob`：Phase 1 rollout，返回 `(x1_exp, log_p_old, saved_noises)`
  - `sde_logprob_recompute`：Phase 2 在同一 `saved_noises` 下重算 `log_p_new`

关键点：Phase 2 必须复用 Phase 1 的噪声序列，否则 ratio 对应轨迹不一致。

注意：在1步场景下，dt = -1.0，Gaussian 近似误差 = O(dt^2) = O(1)，
这是 SDE-GRPO 在1步场景下的根本缺陷，也是我们对比实验的核心论点。
"""

def sde_sample_with_logprob(model, z, y, num_steps, cfg_scale=1.5, sigma=0.1):
    """
    用 Euler-Maruyama 从 z rollout 到 x1，并累积每步 log prob。
    返回 x1、log_prob 以及每步噪声 saved_noises（可选）。
    """
    ...

def sde_logprob_recompute(model, z, saved_noises, y, num_steps, cfg_scale=1.5, sigma=0.1):
    """
    在固定 saved_noises 下重算 log prob，用于 Phase 2 ratio。
    """
    ...
```

---

### 8. `train_distill_only.py`（新建，纯蒸馏 baseline）

与 `train_elbo_grpo.py` 相同，去掉 Phase 1 的 GRPO 部分：

```python
# 只保留 Phase 1 Step 4（MSE 目标）和 Phase 2 Step 8 的 L_mse
# 去掉：探索 ODE、reward 计算、advantage、ELBO ratio、PPO-clip

loss = F.mse_loss(v_eta_hq, u_hq.detach())
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(student.parameters(), 1.0)
optimizer.step()
```

---

### 9. `evaluate.py`（新建）

```python
"""
评估脚本：加载 checkpoint，生成 N 张图，计算指标。

用法：
  python evaluate.py \
    --ckpt results/elbo_grpo/step_3000.pt \
    --num-steps 1 \
    --num-samples 1000 \
    --cfg-scale 1.5 \
    --device cuda:2

指标：
  1. Reward：classifier confidence 均值（主要指标）
  2. FID（如果有 reference stats）
  3. 视觉样本：4x4 grid 存 png
"""
```

---

## 实验列表

### Exp 1：ELBO-GRPO + MSE（主方法）

```bash
CUDA_VISIBLE_DEVICES=2 python train_elbo_grpo.py \
  --ckpt pretrained_models/SiT-XL-2-256x256.pt \
  --T-exp 5 --T-hq 50 \
  --G 8 --K 4 --N-mc 4 \
  --eps-clip 0.05 --adv-clip 2.0 \
  --lambda-mse 0.1 \
  --ratio-stop 1.5 \
  --lr 1e-4 --batch-size 4 \
  --cfg-scale 1.5 \
  --total-iters 3000 --log-every 50 \
  --device cuda:2 \
  --results-dir results/elbo_grpo
```

### Exp 2：SDE-GRPO + MSE（baseline）

```bash
CUDA_VISIBLE_DEVICES=3 python train_sde_grpo.py \
  --ckpt pretrained_models/SiT-XL-2-256x256.pt \
  --T-exp 5 --T-hq 50 \
  --G 8 --K 4 \
  --sde-sigma 0.1 \
  --eps-clip 0.05 --adv-clip 2.0 \
  --lambda-mse 0.1 \
  --lr 1e-4 --batch-size 4 \
  --cfg-scale 1.5 \
  --total-iters 3000 --log-every 50 \
  --device cuda:3 \
  --results-dir results/sde_grpo
```

### Exp 3：纯 MSE 蒸馏（baseline）

```bash
CUDA_VISIBLE_DEVICES=4 python train_distill_only.py \
  --ckpt pretrained_models/SiT-XL-2-256x256.pt \
  --T-hq 50 \
  --lr 1e-4 --batch-size 4 \
  --cfg-scale 1.5 \
  --total-iters 3000 --log-every 50 \
  --device cuda:4 \
  --results-dir results/distill_only
```

### Exp 4（后续）：更新 old policy 消融

在 `train_elbo_grpo.py` 基础上加 `--old-policy-update-freq 100`，
验证 teacher-as-old-policy vs 定期更新学生快照的稳定性差异。

---

## 论文对比表格

| 方法 | NFE | Reward↑ | 训练稳定性 | 说明 |
|------|-----|---------|-----------|------|
| Teacher SiT-XL/2（50步） | 50 | reference | — | 上界 |
| 1步直接推理（无蒸馏） | 1 | ~0 | — | 下界，已验证是乱码 |
| 纯 MSE 蒸馏（Exp 3） | 1 | — | 稳定 | 蒸馏 baseline |
| SDE-GRPO + MSE（Exp 2） | 1 | — | 中等 | ratio 有 O(1) 误差 |
| **ELBO-GRPO + MSE（Exp 1，ours）** | 1 | — | 稳定 | 主方法 |

---

## 开发顺序

```
Day 1：基础组件 + 单元测试

  1. transport/path.py 增加 per_sample_loss
     测试：
       model_same = teacher
       loss_a = per_sample_loss(teacher, x1, kwargs, t=t, noise=eps)
       loss_b = per_sample_loss(teacher, x1, kwargs, t=t, noise=eps)
       assert allclose(loss_a, loss_b)   # 相同输入，相同输出

  2. grpo_utils.py
     测试1（identity ratio）：
       ell_T = per_sample_loss(teacher, x1, ..., t, eps)
       ell_eta = per_sample_loss(teacher, x1, ..., t, eps)   # 同一模型
       log_r = (ell_T - ell_eta).mean(dim=-1)
       assert allclose(log_r, zeros)                          # ratio 应为 1
     测试2（advantage zero-mean）：
       R = randn(4, 8)
       A = compute_grpo_advantage(R)
       assert abs(A.mean(dim=1)).max() < 1e-5

  3. reward_utils.py
     测试：
       random_latent = randn(4, 4, 32, 32)
       pixel = vae.decode(random_latent / 0.18215).sample
       r = reward_fn(pixel, labels)
       assert r.shape == (4,) and (r >= 0).all() and (r <= 1).all()
       print("随机图像 reward 均值（应接近 1/1000）:", r.mean())

  4. sit_sampler.py
     测试：
       x1, pixel = teacher_ode_sample(teacher, vae, y, num_steps=5,
                                       return_pixel=True)
       assert x1.shape == (B, 4, 32, 32)
       assert pixel.shape == (B, 3, 256, 256)
       save_image(pixel, "test_teacher_5step.png")   # 目视检查质量

Day 2：主训练脚本

  5. train_distill_only.py（最简单，先跑通 pipeline）
     跑 200 步，确认：
       - loss 单调下降
       - student 1步图像从乱码逐渐变清晰（每 50 步存一张）

  6. train_elbo_grpo.py
     跑 200 步，确认：
       - reward 均值从 ~0 开始上升
       - mean_ratio 在 0.9~1.1 范围内
       - K_actual 接近 K（early stop 不频繁触发）

Day 3：SDE-GRPO baseline

  7. sampler_sde.py（sde_1step_logprob）
     测试：
       log_p = sde_1step_logprob(teacher, z, x1_exp, y)
       assert log_p.shape == (B,)
       print("log_prob 均值（应为负数，量级合理）:", log_p.mean())

  8. train_sde_grpo.py
     跑 200 步，与 ELBO-GRPO 对比初始行为

Day 4+：完整实验

  9. 三个实验并行跑（3 个空闲 GPU），3000 步
  10. evaluate.py 收集最终指标
  11. 填论文表格
```

---

## 关键实现注意事项

### 显存管理（单卡 ~18GB 可用）

```python
# Teacher fp16，节省 ~1.5GB
teacher = teacher.half()
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Phase 1 全程 no_grad
with torch.no_grad():
    x1_exp = teacher_ode_sample(...)
    ell_T = ...

# Phase 2 内 x_tau_mc 需要 detach，防止对 Phase 1 数据求梯度
v_eta_j = student(x_tau_mc[:, j].detach(), tau_mc[:, j], cond_g)

# 梯度裁剪（防止爆炸）
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
```

### Phase 1 数据的 shape 约定

```python
B  = batch_size      # 4
G  = group_size      # 8
BG = B * G           # 32

cond_g    : (BG,)             # class label，B 组各重复 G 次
x1_exp    : (BG, 4, 32, 32)  # 探索终点
tau_mc    : (BG, N_mc)        # MC 时间步
eps_mc    : (BG, N_mc, 4, 32, 32)
x_tau_mc  : (BG, N_mc, 4, 32, 32)
u_mc      : (BG, N_mc, 4, 32, 32)
ell_T     : (BG, N_mc)        # teacher loss，冻结
R         : (B, G)            # reward
A         : (B, G) → reshape (BG,)  # advantage
```

### Early Stopping 逻辑

```python
# Phase 2 每次更新后检查
with torch.no_grad():
    mean_r = log_r.clamp(-5, 5).exp().mean().item()
if mean_r > ratio_stop or mean_r < 1.0 / ratio_stop:
    if rank == 0:
        logger.info(f"Early stop at k={k}, mean_ratio={mean_r:.3f}")
    break
```

### 监控指标健康范围

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| `reward_mean` | 初期 ~0，应单调上升 | 不上升：检查 reward fn / advantage 是否有梯度 |
| `reward_std` | 0.01~0.1 | < 0.001：探索多样性不足，增大 T_exp |
| `mean_ratio` | 0.9~1.1 | > 1.5：early stop 频繁，减小 lr |
| `clip_fraction` | 0.05~0.20 | ≈ 0：GRPO 无效；> 0.5：eps_clip 太小 |
| `K_actual` | 接近 K | 持续 = 1：lr 太大或 ratio_stop 太小 |
| `L_mse` | 单调下降 | 震荡：lambda_mse 太大干扰 GRPO |