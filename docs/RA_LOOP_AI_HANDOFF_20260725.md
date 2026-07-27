# RA-LOOP 新对话 AI 交接文档 — 2026-07-25

> 最后核对时间：2026-07-25 17:15 CST  
> 工作目录：`/home/imc/yzy/RA_LOOP`  
> 分支：`main`  
> 当前 HEAD：`d8ddbca`  
> 当前 `origin/main`：`6593009`，本地领先 6 个提交  
> **新 AI 在采取任何实验或文件操作前，应完整阅读本文。**

## 0. 一句话状态

RA-LOOP 已完成 OpenVLA-OFT/LIBERO-Spatial 基线复现、旧 fixed-reward 目标的负结果
确认、Counterfactual Recovery Advantage（CRA）与 Nominal Performance Constraint
（NPC）的实现和真实训练链路验证。

最新的 4-task A4/A5 长训练均成功结束，但 48-episode 独立评测中没有任何 checkpoint
超过 warm-start baseline。因此目前是：

```text
实现与训练链路：通过
机制行为：通过
独立正向效果：尚未证明
当前 Gate 2：未通过
```

不能把“代码跑通”写成“方法有效”，也不能把在线训练 reward 当作独立效果证据。

## 1. 最终研究目标

目标不是继续做简单的“给 proprio 扰动 + 固定 recovery reward”，而是形成一个能够
支撑 ICLR 投稿的方法：

1. 从同一个机器人初始状态构造 anchor/perturbed 成对反事实轨迹；
2. 学习 perturbation 下的 outcome-level recovery，而不是强制动作不变；
3. 在提高 recovery 的同时显式约束 nominal/anchor 性能不下降；
4. 用独立、held-out、paired evaluation 证明增益；
5. 最终需要多 task、多 seed、至少另一 LIBERO suite，并尽可能补第二个
   initialization/backbone。

完整方法动机、相关工作边界和论文实验矩阵见：

- `docs/RA_LOOP_ICLR_METHOD_REDIRECT_20260723.md`
- `docs/RA_LOOP_COUNTERFACTUAL_TRAINING_READY_20260724.md`

## 2. 当前方法定义

### 2.1 配对数据

每个 base state 产生：

- anchor：不改变机器人初始关节；
- perturbed：对机器人初始关节施加 fixed-L2 扰动；
- 两者共享 `pair_id`，当前训练强度为 `0.1 rad`。

当前每个训练 step 有 8 条 rollout，即 4 对 anchor/perturbed。

### 2.2 CRA

Counterfactual Recovery Advantage 只把 `anchor success` 的 pair 视为可解释的恢复证据。
anchor 失败的 pair 不能证明 perturbation 造成失败，因此从 CRA 中排除。

当前实现是 hard gate：

- 只保留 anchor 成功 pair 的 perturbed outcome；
- 至少需要 2 个 eligible pair 才能构造 leave-one-out centered advantage；
- group 内 advantage 和为 0；
- 如果 eligible outcome 全成或全败，CRA 仍为 0。

这避免了错误因果归因，但也造成明显的信号稀疏问题。

### 2.3 NPC

Nominal Performance Constraint 为每个 task 维护：

- warm-start anchor reference；
- anchor success EMA；
- dual multiplier `mu_task`；
- allowed nominal drop。

总体优势是：

```text
A_total = A_CRA + mu_task * A_anchor
```

关键安全机制：

- 每 task 先用 untouched warm-start policy 校准；
- 所有 task 校准完成前启用 global barrier，不更新任何参数；
- calibration 或全零 advantage 时禁止 optimizer step，避免 AdamW weight decay
  偷偷改变模型；
- constraint state 当前只保存在进程内，所以训练 profile 强制 fresh output，
  不支持可信 resume。

`lambda_recovery=0` 只表示旧 fixed recovery bonus 已禁用，不表示新 CRA 没有意义。

### 2.4 A4/A5

- A4：CRA-only。
- A5：CRA + NPC。

A4 为保证与 A5 拥有相同校准等待期，仍经过相同 calibration/global barrier，但使用：

```text
nominal_allowed_drop = 1.0
nominal_initial_multiplier = 0.0
```

这样 constraint target 恒为 0，`mu` 数学上始终为 0，只留下 CRA。

## 3. 环境与关键路径

### 3.1 Python/代码环境

不要使用错误的 conda 环境。正确环境是：

```text
/home/imc/anaconda3/envs/ript_vla_openvla_oft
```

关键路径：

```text
项目：
/home/imc/yzy/RA_LOOP

RIPT-VLA：
/home/imc/code/ript-vla

LIBERO official：
/home/imc/code/LIBERO-official

训练数据：
/home/imc/data/ra-loop/libero-datasets

OpenVLA-OFT runtime mirror：
/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke

scale header：
/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
```

warm-start 是旧 overnight pilot 的 step 5：

```text
outputs/ra_loop_spatial_overnight_pilot/libero_spatial/LIBERO_SPATIAL/openvla/
RA-LOOP_spatial_robot_init_overnight_pilot/
one_task_21step_k8_h220_fixed_l2_0p1_recovery_lr1e5/run_000/
openvla_lora_step_000005
```

其关键哈希：

```text
adapter_model.safetensors
bbcc2d58b5466772de2e0ea919119d86d8cbf624754377a48efb96cc07040511

openvla_headers.pt
6f5652ad49c8710fef00d045158875083db9e4edf4cb9652f8ec54e2bfcffa68
```

### 3.2 测试环境变量

直接调用 pytest 时必须隔离用户 site-packages，并给 Numba 可写缓存目录。正确形式：

```bash
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/imc/yzy/RA_LOOP:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla \
  NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache \
  MPLCONFIGDIR=/tmp/ra_loop_mpl_cache \
  MUJOCO_GL=osmesa \
  PYOPENGL_PLATFORM=osmesa \
  /home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python \
  -m pytest -q tests/test_ript_recovery.py
```

如果不设 `PYTHONNOUSERSITE=1`，会误加载 `~/.local` 中不匹配的 `peft/transformers`。
如果不设 `NUMBA_CACHE_DIR`，robosuite 可能尝试在只读安装目录创建 cache 并报错。

历史验证：

- counterfactual 核心合入后完整 CPU tests：`76 passed`；
- A4 profile 添加后相关测试：`17 passed`；
- A4 Hydra/factory CPU-only preflight：passed。

## 4. 用户明确要求与实验工作习惯

新 AI 必须遵守以下偏好。

### 4.1 逐步推进

- 一次只完成一个明确阶段；
- 每个大步骤结束后停下来，向用户解释做了什么、结果意味着什么；
- 不要试图一次把准备、训练、评测和扩展全部做完；
- 启动长实验前说明目的、规模、GPU、预计时间和验收标准；
- 实验跑完后先分析结果，再决定下一步。

### 4.2 GPU 安全

- **GPU 0 不用于训练或评测。**
- 已获准使用 GPU 2--7；训练通常使用 GPU 6/7，六卡独立评测使用 GPU 2--7。
- 每次启动前先只读检查：

```bash
nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

- 推荐拒绝阈值：显存 > 1024 MiB、利用率 > 10%、温度 > 75°C。
- 一个 OpenVLA 7B 进程通常占用约 16--20 GB，不要在同一卡叠加两个模型。
- 训练当前是单卡；多卡最适合并行独立 seed/checkpoint evaluation，不要误以为
  把同一单卡训练直接放到更多卡一定会线性加速。
- GPU 利用率在 MuJoCo/环境 rollout 阶段短暂为 0% 可以正常；应结合 CPU、日志
  mtime 和进程状态判断，不能只看瞬时 GPU utilization 就断言卡死。

### 4.3 文件安全

- 不删除或覆盖已有 output/log/checkpoint；
- 所有训练和评测使用新的、明确日期标记的输出目录；
- launcher 应在已有 output 时拒绝启动；
- 不运行 `rm -rf`、`git reset --hard`、`git checkout --`；
- 当前工作树有重要未提交改动，**不得清理或回滚**；
- 截至 2026-07-25 17:15，磁盘剩余约 166 GB，根分区使用率 91%，扩大实验前
  必须重新检查空间；
- runtime mirror 加载时会修改 `config.json` 并创建 timestamp backup。多个模型
  并行加载此前成功过，但仍应避免不必要的并发写入；
- commit/push 只有在用户明确要求时执行。不要擅自推送。

### 4.4 tmux 习惯

所有超过几分钟的 GPU 操作都放入共享 tmux。原因：

- 用户可以从另一个控制台直接观察；
- 对话中断不会终止训练；
- 不需要 AI 持续 `working` 或高频轮询；
- pane 保留退出信息，便于事后检查。

推荐模式：

```bash
tmux new-session -d -s <descriptive_session> -n gpu6 \
  "bash -lc 'set -o pipefail; cd /home/imc/yzy/RA_LOOP; \
  bash <launcher> --run 6 2>&1 | tee <log>; \
  code=\${PIPESTATUS[0]}; \
  echo [RA_LOOP_EXIT] \${code} | tee -a <log>; \
  exec bash -i'"
```

必须：

- session 名称清楚表达实验；
- 每个 worker/pane 绑定明确 GPU；
- 用 `tee` 保存完整 console log；
- 写明确 exit marker；
- 最后 `exec bash -i`，任务结束后 pane 不消失；
- 不设置会杀掉长训练的短 timeout；
- 给正常进度条；
- 启动后只做一次启动检查，此后用户要求时再查看，避免持续轮询。

常用只读检查：

```bash
tmux list-sessions
tmux list-windows -t <session>
tmux capture-pane -p -t <session>:<window> -S -120
rg "Finished training|CHECKPOINT_DONE|EXIT|Traceback|RuntimeError" <log>
```

注意：tmux session 仍存在不代表任务仍在运行。本文列出的 counterfactual sessions
目前多数只是保留已结束 shell，必须看 exit marker。

用户查看：

```bash
tmux attach -t <session>
```

## 5. 已完成的基线和旧目标结论

### 5.1 OpenVLA-OFT Spatial 基线

完整 LIBERO-Plus Spatial Robot-init：

```text
135 / 350 = 38.5714%
```

见：

- `docs/BASELINE_OFT_SPATIAL_ROBOT350_20260720.md`

### 5.2 当前 4-task warm-start paired baseline

协议：

```text
4 tasks
每 task 6 anchor + 6 fixed-L2
总计 48 episodes
```

结果：

```text
anchor:      24/24
fixed-L2:    21/24
total:       45/48
```

三条 baseline 扰动失败：

```text
top_drawer fixed_l2 init 0
top_drawer fixed_l2 init 4
stove      fixed_l2 init 2
```

### 5.3 旧 fixed-reward/mode-stratified 目标

旧 lambda sweep 和全任务候选没有稳定超过 baseline。常见现象是修复一条失败、
同时新增另一条失败，净结果不提升。

关键文档：

- `docs/RA_LOOP_STRATIFIED_LAMBDA_CHECKPOINT_EVAL_RESULT_20260722.md`
- `docs/RA_LOOP_COUNTERFACTUAL_TRAINING_READY_20260724.md`

因此不要继续盲目扫旧 `lambda_recovery`。

## 6. Counterfactual smoke 结果

### 6.1 第一份 2-step smoke

tmux：

```text
ra_loop_counterfactual_smoke
```

结果：

- calibration 阶段正确禁止参数更新；
- active step 参数更新成功；
- 但只有 1 个 CRA eligible pair，实际主要是 NPC anchor signal；
- exit 0，GPU 正常释放。

### 6.2 第二份 3-step CRA activation smoke

tmux：

```text
ra_loop_counterfactual_cra_smoke
```

日志：

```text
logs/ra_loop_counterfactual_cra_smoke_20260724/gpu7.log
```

结果：

- 3/3 steps，37:06，exit 0；
- 第三步有 2 个 eligible pair；
- eligible perturbed outcome 一成一败，因此 CRA 确实非零；
- `parameter_update_applied=1`；
- model gradient norm `1.7202`；
- header gradient norm（pre-clip）`56.4375`；
- 证明 CRA 在真实 OpenVLA-OFT 训练链路中可生效。

这只证明链路和信号存在，不证明泛化效果。

## 7. 4-task 50-step A4/A5 长训练

### 7.1 A5：CRA + NPC

tmux：

```text
ra_loop_counterfactual_gate
```

日志：

```text
logs/ra_loop_counterfactual_gate_20260724/gpu6.log
```

输出：

```text
outputs/ra_loop_spatial_counterfactual_gate
```

结果：

```text
50/50 steps
耗时 8:41:25
exit 0
global calibration 完成后的 active steps: 35
parameter updates: 31
active online anchor rate: 0.6357
active online perturbed rate: 0.5643
last-10 anchor/perturbed: 0.675 / 0.600
```

约束行为：

```text
next_to_plate:       final mu 0.8865
between/ramekin:     final mu 1.0392
stove:               final mu 0.9571
top_drawer:          final mu 1.0667
```

drawer 与 between 的 violation 使 multiplier 上升，NPC 机制行为符合设计。

### 7.2 A4：CRA-only matched control

tmux：

```text
ra_loop_counterfactual_cra_only_gate
```

日志：

```text
logs/ra_loop_counterfactual_cra_only_gate_20260724/gpu7.log
```

输出：

```text
outputs/ra_loop_spatial_counterfactual_cra_only_gate
```

结果：

```text
50/50 steps
耗时 8:30:42
exit 0
active steps: 35
真正产生非零 CRA 更新的 steps: 17
active online anchor rate: 0.6929
active online perturbed rate: 0.6286
last-10 anchor/perturbed: 0.600 / 0.650
四个 task 的 mu 始终为 0
```

`17/35` 是重要诊断：hard-pair CRA 在每批只有 4 对时信号很稀疏。

### 7.3 训练解释注意事项

- A4/A5 前 33 步 task order 相同；
- 从 step 34 开始有 10 个 step 的 task order 不同，原因是优化路径消耗 RNG 后影响
  后续 dataloader shuffle；
- 两组数据预算、seed 和 task 集匹配，但不能声称 50 步 task 顺序逐步完全相同；
- online rate 是 on-policy 训练数据，不能作为最终效果比较。

### 7.4 checkpoint 保存陷阱

两份 50-step 训练只留下：

```text
step 10 / 20 / 30 / 40
```

没有 step 50。当前 trainer/save interval 语义不会自动保存最终 step。

step 10 发生在 global calibration barrier 内，A4/A5 的 adapter/header 与 warm-start
逐文件哈希完全一致。因此独立评测跳过了 step 10。

未来长训练必须先修正或验证 final checkpoint 保存，不要再次默认“50 步就会有
step-50 目录”。

## 8. 最新独立评测结果

### 8.1 运行

tmux：

```text
ra_loop_counterfactual_checkpoint_eval
```

日志与结果：

```text
logs/ra_loop_counterfactual_checkpoint_eval_20260725
```

launcher：

```text
eval/launch_ra_loop_counterfactual_checkpoint_eval.sh
eval/launch_ra_loop_counterfactual_checkpoint_eval_tmux.sh
```

规模：

```text
A4 step 20/30/40
A5 step 20/30/40
每 checkpoint 4 tasks × 12 episodes = 48
总计 288 episodes
24/24 summaries
288/288 rows
6/6 exit 0
GPU 2--7 已释放
```

### 8.2 汇总

| 模型 | Anchor | Fixed-L2 | 总计 | 相对 baseline |
|---|---:|---:|---:|---|
| warm-start baseline | 24/24 | 21/24 | 45/48 | — |
| A4 CRA-only step 20 | 24/24 | 20/24 | 44/48 | 新增 1 个 perturb failure |
| A4 CRA-only step 30 | 24/24 | 21/24 | 45/48 | 48 条逐样本完全相同 |
| A4 CRA-only step 40 | 24/24 | 21/24 | 45/48 | 48 条逐样本完全相同 |
| A5 CRA+NPC step 20 | 24/24 | 21/24 | 45/48 | 48 条逐样本完全相同 |
| A5 CRA+NPC step 30 | 23/24 | 21/24 | 44/48 | 新增 1 个 anchor failure |
| A5 CRA+NPC step 40 | 24/24 | 21/24 | 45/48 | 48 条逐样本完全相同 |

新增失败：

```text
A4 step20:
next_to_plate fixed_l2 init 1

A5 step30:
stove anchor init 1
```

没有任何 checkpoint 修复 baseline 的三条 fixed-L2 失败。

### 8.3 当前科学结论

严格结论是：

1. 实现、训练、校准、约束和 checkpoint evaluation 均正常；
2. 当前 4-task/50-step/hard-pair CRA 没有产生可检测的独立恢复提升；
3. NPC 会动态响应 nominal risk，但没有带来净成功率提升；
4. 当前 Gate 2 正向效果标准未通过；
5. 不允许写“方法已证明有效”。

同时，48 episodes 很窄：

```text
4 tasks × (6 anchor + 6 perturbed) = 48
真正衡量 recovery 的 perturbed denominator 只有 24
1 条 episode = 4.17 percentage points
baseline perturbed 已经 87.5%，存在明显 ceiling effect
```

所以该结果足以阻止盲目扩大训练，但还不足以断言方法在更宽分布上绝对无效。

## 9. 下一步建议与决策树

### 9.1 第一优先：扩大评测，不立即重训

只评测三个模型：

```text
warm-start baseline
A4 CRA-only step 40
A5 CRA+NPC step 40
```

建议首轮宽评测：

```text
10 个 LIBERO-Spatial tasks
4 个训练内 tasks 与 6 个未训练 tasks 分开报告
每 task 20 个新 init states
避开当前 init 0--5，建议使用连续 held-out 区间，例如 6--25
使用训练未见的 perturbation seed
anchor + fixed-L2 0.1
3 models × 10 tasks × 20 pairs × 2 modes = 1200 episodes
```

当前 evaluator 固定使用 `range(num_pairs)`，所以开始宽评测前应先添加并测试
`--init-start`（或显式 init index manifest），避免误称 held-out。

宽评测必须：

- 使用完全相同的 init indices 和 perturbation seeds；
- 报告 seen/unseen tasks；
- 报告 paired wins、paired losses、净变化和置信区间；
- 预先冻结选择规则，不能看结果后挑 checkpoint；
- 不重复评测 A4/A5 的所有 checkpoint，先只测 step 40。

建议 Gate：

```text
perturbed success 提升约 >= 5 percentage points
anchor drop <= 2 percentage points
paired gains 不能只是等量交换失败
```

### 9.2 宽评测后的分支

如果 A4/A5 step 40 出现可信增益：

1. 补 3 个训练 seeds；
2. 再扩到 10-task 训练；
3. task 数增加时，训练覆盖和总 steps 必须一起增加；
4. 建议 10 tasks、更多 init/demos、约 100--150 steps；
5. 再加入 held-out strength/direction；
6. A4/A5 成立后才考虑 Recovery-Frontier Curriculum。

如果宽评测仍无增益：

1. 不要继续把相同 hard-pair 训练简单拉长；
2. 优先解决 `17/35` 非零 CRA 更新的稀疏性；
3. 候选方向包括：
   - soft confidence/competence weighting，替代一次 anchor failure 的硬切断；
   - 增加每批 pair 数；
   - 跨 batch recovery replay/baseline；
   - failure-focused sampling，但必须保持 episode budget 对照；
   - 明确记录 CRA nonzero count，而不只记录 group eligible count；
4. 修改后先做 CPU truth-table 和小 smoke，再进入长训练。

### 9.3 为什么现在不直接增加训练任务数

“提高任务数量”和“扩大评测样本”不是同一件事。

当前应先扩大评测，因为 24 个 perturbed 样本太少。若把训练从 4 tasks 直接改成
10 tasks、但仍只训练 50 steps，每个 task 获得的 exposure 反而更少，calibration
也会消耗更大比例预算。因此训练 task、每 task init coverage 和总 steps 必须联动。

## 10. 当前 Git/文件状态

截至本文写入前，工作树已有以下重要未提交改动：

```text
 M eval/ra_loop_checkpoint_pair_eval.py
 M train/ra_loop_spatial_learning_probe.sh
?? eval/launch_ra_loop_counterfactual_checkpoint_eval.sh
?? eval/launch_ra_loop_counterfactual_checkpoint_eval_tmux.sh
?? train/ra_loop_counterfactual_cra_only_gate.sh
```

本文自身加入后还会新增：

```text
?? docs/RA_LOOP_AI_HANDOFF_20260725.md
```

这些改动属于已成功完成的 A4 和独立评测，不是临时垃圾。新 AI 不得回滚。

最近相关提交：

```text
d8ddbca Add CRA activation smoke
242fba8 Prepare counterfactual constrained training
1ae2502 Add nominal performance dual constraint
285bba7 Add pair-conditioned recovery advantages
d55df53 Add counterfactual recovery metrics
0c06804 Redirect RA-LOOP toward counterfactual recovery
6593009 Record full-task candidate evaluation launch
```

当前 `main` 相对 `origin/main` 领先 6 commits。本轮 A4/evaluator/handoff 仍未提交、
未推送。只有用户明确要求后才 commit/push。

## 11. 新 AI 开始工作的建议顺序

1. 完整阅读本文；
2. 阅读：
   - `docs/RA_LOOP_ICLR_METHOD_REDIRECT_20260723.md`
   - `docs/RA_LOOP_COUNTERFACTUAL_TRAINING_READY_20260724.md`
3. 执行只读检查：

```bash
git status --short
df -h /home/imc/yzy/RA_LOOP
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

4. 不要重新运行已完成的 48-episode evaluation；
5. 向用户复述当前结论：链路通过、独立增益未证明、48 episodes 太窄；
6. 下一步先设计并预检 1200-episode held-out evaluation；
7. 在用户确认后才启动；
8. 仍按“一步完成后停下来解释”的节奏工作。

## 12. 禁止误读

新 AI 特别注意：

- `lambda_recovery=0` 不代表 CRA 关闭；
- `mean_perturbed_advantage=0` 可能只是 centered estimator 均值为零，不能单独判断
  CRA 是否为零；
- `cra_groups_with_update=1` 只表示 eligible 数足够，不保证 outcome mixed；
- A4 中 `parameter_update_applied=1` 才能直接说明有非零 CRA，因为 `mu=0`；
- A5 的 parameter update 可能仅来自 NPC；
- online success 不是 independent evaluation；
- step 10 是未更新 warm-start，不是训练后收益；
- 没有 step-50 checkpoint；
- 48-episode 结果没有提升，但样本量和 ceiling 都不足以给出论文级否定结论；
- tmux session 存在不代表 GPU 任务仍在运行；
- 不要使用 GPU 0；
- 不要在固定 50-step 预算下简单增加 task 数并声称扩大覆盖。
