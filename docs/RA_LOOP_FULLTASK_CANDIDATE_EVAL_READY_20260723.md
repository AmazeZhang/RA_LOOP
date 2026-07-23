# RA-LOOP Spatial 全任务候选独立评测 — 20260723

> 状态：2026-07-23 22:05 CST 已启动。六个首轮 worker 均已加载对应
> checkpoint 并进入 episode。

tmux 会话为 `ra_loop_fulltask_candidate_eval`，窗口 `gpu2` 到 `gpu7`；每张卡仅在
当前候选 exit 0 后串行进入下一候选。

固定评测集为 10 tasks ×（6 anchor + 6 fixed-L2）= 每候选 120 episodes。候选包括
pilot step-5 warm-start baseline，以及 lambda=0/0.5 × seed10000/20000 四组训练的
step30/60/90，共 13 个候选、1560 deterministic paired episodes。

GPU 2--7 六个 worker 各串行 2--3 份候选，最大并发始终为 6；预计约 5--6 小时。
任一候选非零退出时，该卡不继续后续候选。四个旧评测任务沿用原来的 init/perturb seed，
六个新任务使用相同 fixed-L2 0.1 口径。结果只用于选择是否补评邻近 checkpoint，不把
训练在线成功率作为模型选择依据。
