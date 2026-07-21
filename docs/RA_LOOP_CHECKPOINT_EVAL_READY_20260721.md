# RA-LOOP checkpoint paired-eval 就绪记录 — 20260721

> 完成时间：2026-07-21 09:18:43 CST（UTC+08:00）  
> 当前边界：五候选已于 2026-07-21 09:24--09:38 CST 在共享 tmux 中启动；
> 100 条独立 paired evaluation 已全部完成。结果见
> `docs/RA_LOOP_CHECKPOINT_PAIR_EVAL_RESULT_20260721.md`。

## 正式运行状态（2026-07-21 09:39:17 CST）

- tmux session：`ra_loop_checkpoint_pair_eval`
- window/GPU：`step0`/1、`step5`/2、`step10`/3、`step15`/4、`step20`/5
- 五张卡启动前均约 18 MiB、0% utilization、41--43°C。
- 启动后显存约 15.8--16.7 GiB，检查温度 44--55°C，未触发安全闸。
- step 5/10/15/20 均已确认 base model、对应 LoRA、action/scale headers
  加载成功并进入 `starting episode`；step 5 检查时已完成 2/20。
- step 0 于 09:24 首先单独启动并持续正常占用 GPU；其标准输出因启动时未设置
  Python unbuffered 而延迟显示。后续四窗口已设置 `PYTHONUNBUFFERED=1`，日志实时更新。
- 运行日志：`logs/ra_loop_checkpoint_pair_eval_20260721/console_step{0,5,10,15,20}.log`
- 最终结果仍只在每个 checkpoint 完成 20/20 后原子生成，不将中间状态当作完成结果。

观察命令：

```bash
tmux attach -t ra_loop_checkpoint_pair_eval
```

切换窗口用 `Ctrl-b` 后按 `n`/`p`，退出观察并保持运行用 `Ctrl-b` 后按 `d`。

## 评测目标

对未训练 step 0 以及 pilot step 5/10/15/20 做完全相同的独立 paired evaluation，
避免用训练期在线成功率选 checkpoint。

每个候选固定：

- 同一 Spatial 任务
- 基础 init indices 0--9
- 10 条 anchor
- 同一 10 条 base state 上施加 fixed-L2 0.1 rad
- perturb seeds 20260720--20260729
- deterministic action mean（关闭 Laplace sampling）
- horizon=220，单 in-process 环境
- 共 20 episodes/checkpoint

总计 5×20=100 条 episode。GPU 1--5 已并行启动；依据训练 rollout 实测，
从最后一个窗口启动起预计约 20--30 分钟，保守按 25--35 分钟观察。

## 加载契约

- step 0：suite-specific Spatial base model、base action/proprio head、base scale header；
  policy 的新 LoRA 分支按 PEFT 默认零增量初始化。
- step 5/10/15/20：相同 base model，加载对应 `adapter_model.safetensors`，随后
  从同目录 `openvla_headers.pt` 加载训练后的 action/scale header。
- checkpoint 目录只读，不生成 runtime copy，不修改权重。

## 实现与安全

- evaluator：`eval/ra_loop_checkpoint_pair_eval.py`
- launcher：`eval/launch_ra_loop_checkpoint_pair_eval.sh`
- launcher 无参数/`--plan` 只做 CPU 路径和计划验证；GPU 必须显式
  `--run <STEP> <GPU_ID>`。
- 每个 step 的输出目录必须不存在；拒绝覆盖或混写。
- GPU 路径拒绝显存 >1 GiB、利用率 >10% 或温度 >75°C 的卡。
- evaluator 要求 launcher 只暴露一张 GPU。
- 结果先写 `results.jsonl.incomplete`，完整 20 条后原子改名为 `results.jsonl`；
  crash 时不会伪装成完整结果。
- summary 记录 adapter/header SHA-256、anchor/perturbed successes 和配置。

## 验证

- Python `py_compile`：passed
- shell syntax：passed
- 五套 CPU-only plan：passed
- 五条 GPU 命令静态打印：passed
- 项目回归：30/30 passed
- 输出根目录启动前不存在
- 本步骤 CUDA 隐藏，未加载模型/环境、未查询或使用 GPU

## 下一步

100/100 覆盖及原子输出已验收。当前结果在 step 5--20 全部饱和为 20/20；继续训练前
先扩大独立任务/初态覆盖，以打破 checkpoint 间并列。
