# WAM-Flow 8×A800 项目完成流程

本文面向一台 **8 张 NVIDIA A800** 的训练机器，重新整理 WAM-Flow 从环境准备、数据准备、四阶段训练、GRPO 强化学习、NAVSIM 评测到面试复盘的完整项目完成流程。

当前版本的核心原则是：

1. **先复现官方已开源链路**：安装、模型下载、NAVSIM 数据、SFT、推理、PDM score 评测。
2. **再接入本项目新增的模拟 GRPO**：原始官方 GRPO 未开源，本仓库现在新增 `train_grpo.py`、`config/grpo_navsim.yaml`、`scripts/grpo_navsim.sh` 和 `flow_matching/rl/`，用于先把 GRPO 训练闭环跑通。
3. **最后做 metric-aligned 对齐**：当前 GRPO 先用 proxy metric 跑通训练逻辑；正式结果需要把 proxy metric 替换为 NAVSIM metric cache / PDM scorer 的真实 NC、DAC、EP、TTC、Comfort 指标。

## 0. 当前项目状态

| 模块 | 当前状态 | 入口文件 / 脚本 | 说明 |
| --- | --- | --- | --- |
| 环境安装 | 已具备 | `README.md`, `setup.py`, `requirements.txt` | 安装 `nuplan-devkit` 和本项目。 |
| 模型下载 | 已具备 | `README.md` | 下载 WAM-Flow 与 FUDOKI 预训练模型。 |
| NAVSIM 数据 | 已具备下载入口 | `download/*.sh` | 需要准备数据、地图和 metric cache。 |
| Stage 1 数值 tokenizer / embedding | 代码支持 | `train.py` | `stage=s1` 时扩展数值 token 并训练 embedding / lm head。 |
| Stage 2 VQA continued pretraining | 需要按数据补配置 | `train.py` | 仓库没有独立 VQA 预训练脚本，可基于现有 SFT 数据格式扩展。 |
| Stage 3 NAVSIM SFT | 已具备 | `scripts/sft_navsim.sh`, `config/sft_navsim.yaml`, `train.py` | 默认 8 GPU 训练，数据为 `data/navsim_668k.jsonl`。 |
| 推理 | 已具备 | `scripts/infer.sh`, `infer.py` | 默认 `discrete_fm_steps=2`。 |
| NAVSIM PDM 评测 | 已具备 | `scripts/evaluation/run_wam_flow_agent_pdm_score_evaluation.sh` | 默认 8 GPU `torchrun`。 |
| Stage 4 模拟 GRPO | 已新增 | `train_grpo.py`, `config/grpo_navsim.yaml`, `scripts/grpo_navsim.sh`, `flow_matching/rl/` | 使用 NAVSIM proxy reward 跑通 GRPO；后续替换为真实 PDM scorer。 |

## 1. 8×A800 总体执行顺序

推荐按下面顺序推进，避免一开始就直接跑完整训练：

```text
环境安装
  ↓
模型 / 数据 / metric cache 准备
  ↓
单卡 infer.py 冒烟测试
  ↓
Stage 1 Numerical tokenizer / embedding
  ↓
Stage 2 VQA continued pretraining（可小规模或按材料补齐）
  ↓
Stage 3 NAVSIM SFT（8×A800 主训练）
  ↓
NAVSIM PDM score 评测
  ↓
Stage 4 模拟 GRPO（8×A800，先 proxy metric，后 PDM metric）
  ↓
GRPO 后评测 + inference steps 消融 + 面试材料整理
```

## 2. 机器与环境规划

### 2.1 硬件假设

- GPU：8×NVIDIA A800。
- 训练方式：SFT 使用 `accelerate launch`；评测使用 `torchrun --nproc_per_node=8`。
- GRPO：建议先单机 8 卡，后续如果 rollout / PDM scorer 太慢，再做异步 rollout 或 reward cache。

### 2.2 环境安装

```bash
conda create --name wam-flow python=3.9
conda activate wam-flow
pip install -e ./nuplan-devkit
pip install -e .
```

### 2.3 路径约定

建议统一目录结构：

```text
pretrained_model/
  fudoki/
  wam-flow/
data/
  navsim_668k.jsonl
  navsim_data/
output/
  train/
    stage1_numeric/
    stage2_vqa_continue/
    navsim_sft/
    grpo_navsim/
exp/
```

## 3. 模型与数据准备

### 3.1 下载模型

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download fudan-generative-ai/WAM-Flow --local-dir ./pretrained_model/wam-flow
huggingface-cli download LucasJinWang/FUDOKI --local-dir ./pretrained_model/fudoki
mv pretrained_model/wam-flow/data/* data/
```

### 3.2 下载 NAVSIM 数据与地图

```bash
sh download/download_trainval.sh
sh download/download_maps.sh
```

### 3.3 生成 metric cache

```bash
sh scripts/evaluation/run_metric_caching.sh
```

需要在相关脚本中确认：

- `NUPLAN_MAPS_ROOT`
- `OPENSCENE_DATA_ROOT`
- `METRIC_CACHE_PATH`
- `NAVSIM_EXP_ROOT`

## 4. Stage 0：单卡推理冒烟测试

在正式训练前，先确认模型、processor、embedding、图像路径都能正常工作：

```bash
sh scripts/infer.sh
```

验收标准：

- 能成功加载 FUDOKI 和 WAM-Flow checkpoint。
- 能输出可解析的轨迹数字。
- `extract_num_list()` 或等价解析逻辑能得到 16 个数字，即 8 个 `(x, y)` waypoint。

## 5. Stage 1：Numerical embedding / tokenizer

### 5.1 目标

把连续轨迹坐标离散成数值 token，使语言模型稳定输出规划坐标。当前代码在 `use_quantize=true` 时会构造 `[-100, 100]`、间隔 `0.01` 的 `20001` 个数值 token。

### 5.2 训练配置

建议新增 `config/stage1_numeric.yaml`，核心字段：

```yaml
stage: s1
model_path: pretrained_model/fudoki
pretrain_model_path: pretrained_model/fudoki
text_embedding_path: pretrained_model/fudoki/text_embedding.pt
use_quantize: true
batch_size: 1
learning_rate: 5.0e-6
max_train_steps: 40000
```

### 5.3 8×A800 启动方式

```bash
accelerate launch \
  --config_file ./config/accelerate_config_ds2.yaml \
  --num_processes 8 \
  train.py \
  --config config/stage1_numeric.yaml \
  --output_dir output/train/stage1_numeric
```

### 5.4 验收标准

- tokenizer 新增 `20001` 个数值 token。
- embedding 维度扩展正确。
- 新增数值 token 不出现 NaN。
- 抽样 decode 时坐标格式更稳定。

## 6. Stage 2：VQA continued pretraining

### 6.1 目标

在进入规划 SFT 前，保持或增强模型对道路图像、导航语义和问答格式的理解能力。你之前补充的信息里提到可使用 **0.65M 或部分 VQA 数据**，这里建议先做小规模验证，再扩到完整数据。

### 6.2 当前限制

当前仓库没有官方单独的 VQA continued pretraining 脚本，所以这一阶段需要：

- 准备 VQA jsonl 数据；
- 对齐 `flow_matching/data/navsim.py` 支持的 image + conversations 格式；
- 从 Stage 1 checkpoint 继续训练；
- 保持数值 tokenizer 与后续 SFT 一致。

### 6.3 建议配置

新增 `config/stage2_vqa_continue.yaml`：

```yaml
stage: s2
model_path: pretrained_model/fudoki
pretrain_model_path: output/train/stage1_numeric/checkpoint-xxx
text_embedding_path: pretrained_model/fudoki/text_embedding.pt
data_list:
  - data/vqa_continue.jsonl
use_quantize: true
batch_size: 1
learning_rate: 5.0e-6
max_train_steps: 40000
```

### 6.4 启动方式

```bash
accelerate launch \
  --config_file ./config/accelerate_config_ds2.yaml \
  --num_processes 8 \
  train.py \
  --config config/stage2_vqa_continue.yaml \
  --output_dir output/train/stage2_vqa_continue
```

### 6.5 验收标准

- VQA loss 稳定下降。
- 图像问答能力没有明显退化。
- 数值 token embedding 可继续加载。
- 后续 NAVSIM SFT 初始 loss 不异常。

## 7. Stage 3：NAVSIM / nuPlan SFT

### 7.1 目标

让模型通过监督学习，从前视图像、导航命令、自车状态中生成未来 4 秒 8 个 waypoint。

### 7.2 官方配置入口

当前主配置为：

```bash
config/sft_navsim.yaml
```

核心字段包括：

```yaml
stage: s2
data_list:
  - data/navsim_668k.jsonl
batch_size: 1
learning_rate: 5e-6
max_train_steps: 40000
max_epochs: 100
checkpointing_steps: 4000
```

### 7.3 8×A800 启动方式

```bash
sh scripts/sft_navsim.sh
```

该脚本默认：

```bash
NUM_NODES=1
NUM_GPUS=8
```

### 7.4 训练过程关注点

- `ce_loss` 是否稳定下降。
- checkpoint 是否按 `checkpointing_steps` 保存。
- 是否出现坏图路径、坏 jsonl 样本或 tokenizer 长度异常。
- 8 卡显存是否稳定；如果 OOM，优先降 `batch_size` 或开 gradient accumulation。

### 7.5 验收标准

- `infer.py` 可以从 SFT checkpoint 输出轨迹。
- 输出能解析出 16 个数字。
- NAVSIM PDM score 评测可以跑通。

## 8. Stage 4：NAVSIM 模拟 GRPO

### 8.1 目标

在 SFT checkpoint 基础上进一步做 simulator-guided alignment，使优化目标从 token-level imitation 走向 metric-level driving quality。

原始官方 GRPO 代码未开源；本项目已经新增一套模拟 GRPO 训练入口，用来先完成完整 RL 闭环：

- `config/grpo_navsim.yaml`
- `scripts/grpo_navsim.sh`
- `train_grpo.py`
- `flow_matching/rl/grpo.py`
- `flow_matching/rl/reward.py`
- `flow_matching/rl/rollout.py`
- `flow_matching/rl/navsim_runner.py`

### 8.2 GRPO 配置

当前 `config/grpo_navsim.yaml` 按你补充的信息组织：

```yaml
grpo:
  train_size: 103000
  epochs: 0.5
  group_size: 3
  lr: 1.0e-6
  batch_size: 32
  warmup_steps: 500
  weight_decay: 0.01
  reward:
    safety:
      - NC
      - DAC
    performance:
      EP: 5
      TTC: 5
      Comfort: 2
  kl:
    reference_model: sft_checkpoint
    beta: null
  clip_epsilon: null
  sampling:
    denoise_steps: [1, 3, 5]
```

说明：

- `group_size=3`：每个场景采样 3 条候选轨迹。
- `denoise_steps=[1,3,5]`：三条候选可以对应不同 denoise steps，增强组内差异。
- `NC`、`DAC`：安全项，用作 safety gate。
- `EP:5`、`TTC:5`、`Comfort:2`：性能项加权。
- `beta` 和 `clip_epsilon`：论文没有给定具体值，必须调参，不能编造成固定值。

### 8.3 当前实现逻辑

1. 从 Stage 3 SFT checkpoint 加载 policy model。
2. 从同一个 SFT checkpoint 加载 frozen reference model。
3. 对每个 NAVSIM 场景采样 `group_size=3` 条候选轨迹。
4. 使用 denoise steps `1 / 3 / 5` 生成不同候选。
5. 将 token decode 成文本，解析 16 个轨迹数字。
6. 当前先用 proxy metric 计算 `NC`、`DAC`、`EP`、`TTC`、`Comfort`。
7. reward 计算为：

```text
reward = safety_gate(NC, DAC) * (5 * EP + 5 * TTC + 2 * Comfort)
```

8. 组内计算 advantage：

```text
advantage_i = reward_i - mean(reward_group)
```

9. 使用 clipped GRPO loss，并加入 reference KL：

```text
loss = clipped_policy_loss + beta * KL(policy || reference)
```

10. 保存 checkpoint，并在后续接入真实 PDM scorer 后按 PDM score 选择最优模型。

### 8.4 8×A800 启动方式

先做配置检查：

```bash
python train_grpo.py --config config/grpo_navsim.yaml --dry_run
```

正式启动建议先用保守参数：

```bash
BETA=0.03 CLIP_EPSILON=0.2 sh scripts/grpo_navsim.sh
```

如果显存压力大，优先调整：

1. `grpo.batch_size: 32 -> 16 -> 8`
2. `grpo.train_size: 103000 -> 10000 -> 5000`
3. 暂时固定 `denoise_steps=[3]`，等闭环稳定后再恢复 `[1,3,5]`

### 8.5 真实 PDM scorer 替换计划

当前 `flow_matching/rl/navsim_runner.py` 提供的是本地开发用 proxy metric。为了得到论文级结果，需要替换为：

1. token 输出 → `Trajectory`；
2. `Trajectory` → NAVSIM / PDM scorer；
3. PDM scorer 输出真实：
   - `NC`
   - `DAC`
   - `EP`
   - `TTC`
   - `Comfort`
4. `flow_matching/rl/reward.py` 继续负责加权组合 reward。

### 8.6 GRPO 验收标准

- 同一场景 3 条候选轨迹 reward 有区分度。
- reward、advantage、KL、ratio 日志稳定。
- `beta` 增大时 KL 应下降。
- `clip_epsilon` 变小时训练更保守。
- 轨迹解析失败率不能升高。
- 接入真实 PDM scorer 后，GRPO checkpoint 的 PDM score 高于 SFT baseline。

## 9. NAVSIM PDM score 评测闭环

### 9.1 启动方式

```bash
sh scripts/evaluation/run_wam_flow_agent_pdm_score_evaluation.sh
```

该脚本默认 8 GPU：

```bash
GPUS=${GPUS:-8}
torchrun --nproc_per_node=8 navsim/planning/script/run_pdm_score_wam_flow.py ...
```

### 9.2 需要确认的路径

```bash
export NUPLAN_MAPS_ROOT="/path/to/navsim_dataset/maps"
export OPENSCENE_DATA_ROOT="/path/to/navsim_dataset"
export METRIC_CACHE_PATH="/path/to/metric_cache"
```

### 9.3 必须记录的指标

| 类别 | 指标 |
| --- | --- |
| 总分 | PDM score |
| 安全 | NC, DAC, collision, drivable area |
| 效率 | EP, progress |
| 风险 | TTC |
| 舒适 | Comfort |
| 工程 | latency / scene, parse failure rate, OOM / failure case |

## 10. Inference steps 消融

### 10.1 目标

验证离散 flow matching 推理步数与效果 / 延迟的关系。

### 10.2 推荐消融

| steps | 用途 |
| --- | --- |
| 1 | 最快速度，观察性能下限。 |
| 2 | 当前脚本默认，作为主 baseline。 |
| 3 | 与 GRPO sampling 对齐。 |
| 5 | 与 GRPO sampling 对齐。 |
| 8 | 中等步数。 |
| 16 | 高质量候选。 |
| 50 | 接近充分 denoise，但延迟高。 |

### 10.3 单样本命令

```bash
torchrun --nproc_per_node 1 infer.py \
  --checkpoint_path pretrained_model/wam-flow/navsim \
  --image_path data/navsim_data/sensor_blobs/test/example.jpg \
  --processor_path pretrained_model/fudoki \
  --text_embedding_path pretrained_model/fudoki/text_embedding.pt \
  --image_embedding_path pretrained_model/fudoki/image_embedding.pt \
  --discrete_fm_steps 2 \
  --seed 123
```

### 10.4 NAVSIM 评测消融

修改：

```bash
agent.discrete_fm_steps=2
```

记录：

| checkpoint | steps | PDM score | NC | DAC | EP | TTC | Comfort | latency / scene |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT | 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 11. 8×A800 资源使用建议

### 11.1 SFT 阶段

- 默认使用 8 卡。
- 如果显存足够，优先保持 `batch_size=1`、增加 gradient accumulation，而不是盲目增 batch。
- checkpoint 每 4000 step 保存一次，避免长跑中断损失过大。

### 11.2 GRPO 阶段

GRPO 比 SFT 更吃资源，因为每个场景要采样多条候选轨迹，还要算 reward 和 reference KL。

建议分三轮：

| 轮次 | train_size | group_size | denoise_steps | 目的 |
| --- | --- | --- | --- | --- |
| Debug | 1000 | 3 | [1, 3, 5] | 验证训练闭环、日志和 checkpoint。 |
| Pilot | 10000 | 3 | [1, 3, 5] | 验证 reward 是否能拉开差距。 |
| Full | 103000 | 3 | [1, 3, 5] | 对齐论文完整设置。 |

### 11.3 `beta` / `clip_epsilon` 调参

建议从小网格开始：

| beta | clip_epsilon | 预期 |
| --- | --- | --- |
| 0.01 | 0.2 | 更新更激进，观察 reward 是否提升但 KL 是否变大。 |
| 0.03 | 0.2 | 默认起点。 |
| 0.1 | 0.1 | 更保守，适合 KL 爆炸或轨迹退化时。 |

## 12. 最终交付清单

| 类别 | 交付物 | 验收方式 |
| --- | --- | --- |
| 环境 | conda env、依赖安装记录 | 能 import `navsim`, `flow_matching`, `fudoki`。 |
| 数据 | NAVSIM 数据、地图、metric cache | PDM 评测脚本不报路径错误。 |
| Stage 1 | numeric tokenizer / embedding checkpoint | 新增数值 token 可加载。 |
| Stage 2 | VQA continued checkpoint | VQA 抽样正常，数值 token 不损坏。 |
| Stage 3 | NAVSIM SFT checkpoint | `infer.py` 输出 8 个 waypoint。 |
| Stage 4 | GRPO checkpoint | proxy reward 闭环可跑；接入 PDM 后分数高于 SFT。 |
| 评测 | SFT vs GRPO PDM score 表 | 包含 NC、DAC、EP、TTC、Comfort。 |
| 消融 | inference steps 表 | 包含 score 与 latency。 |
| 面试材料 | 项目流程、调用链、核心文件说明 | 能解释训练目标、数据流、reward 和工程取舍。 |

## 13. 面试中可直接使用的总结

我会按 8 张 A800 的资源把项目拆成四个阶段完成：第一阶段做 numerical tokenizer 和 embedding，让模型能稳定生成轨迹坐标；第二阶段做 VQA continued pretraining，保留视觉语言理解能力；第三阶段用 NAVSIM / nuPlan 数据做 SFT，让模型从前视图、导航命令和自车状态生成未来 4 秒 8 个 waypoint；第四阶段接入我自己实现的模拟 GRPO，从 SFT checkpoint 初始化 policy 和 reference model，用 10.3 万 NAVSIM 场景、0.5 epoch、group size 3、学习率 1e-6，采样 denoise steps 1/3/5 的候选轨迹，再用 NC、DAC、EP、TTC、Comfort 组成 reward 做 metric-level alignment。当前 GRPO 先用 proxy metric 跑通训练闭环，最终会替换成 NAVSIM PDM scorer 的真实指标，用 PDM score 验证是否超过 SFT baseline。
