# RA-LOOP mode-stratified checkpoint 独立评测 — 20260722

## 设计

- checkpoint：stratified 四任务训练 step 5/10/15/20/25/30
- GPU：1/2/3/4/5/6，一 checkpoint 一卡
- 每 checkpoint：四任务 ×（6 anchor + 6 fixed-L2 0.1）= 48 episodes
- 总计：288 episodes
- deterministic action mean，horizon=220，单环境
- init、perturb seed 与训练前 baseline、旧 mixed-objective 评测精确匹配
- 预计 45--60 分钟

主要判据为 anchor 24/24 保持、fixed-L2 超过 21/24，以及相对 baseline 的逐 pair
gain/loss。中间 task 结果不作为有效结论，必须等 288/288 原子结果全部完成。

## 安全与路径

- shared tmux：`ra_loop_stratified_checkpoint_eval`
- 输出：`logs/ra_loop_stratified_checkpoint_eval_20260722/step<step>/task<0..3>/`
- 控制台：`logs/ra_loop_stratified_checkpoint_eval_20260722/console_step<step>.log`
- launcher 拒绝已有 session/output，且每份 step 启动前检查 GPU 显存、利用率和温度
- evaluator 默认只生成 CPU plan；`--execute` 只由 guarded launcher 添加
- 每个 task 只有完整 12 条后才原子生成 `results.jsonl`

完成后验收 24 summaries、288 rows、无 `.incomplete`、六个 exit 0 与 GPU 释放，再做
paired checkpoint 选择。
