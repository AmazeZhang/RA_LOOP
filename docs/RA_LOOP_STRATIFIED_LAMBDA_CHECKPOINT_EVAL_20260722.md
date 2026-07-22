# RA-LOOP stratified lambda checkpoint 独立评测 — 20260722

> 状态：2026-07-22 22:21 CST 已完成。12/12 exit 0、48 summaries、576/576 rows、
> 无 incomplete，GPU 2--7 已释放。结果见
> `docs/RA_LOOP_STRATIFIED_LAMBDA_CHECKPOINT_EVAL_RESULT_20260722.md`。

tmux 会话：`ra_loop_stratified_lambda_checkpoint_eval`。六个窗口分别绑定 GPU 2--7，
首份成功结束后会在同一卡上自动启动第二份。

## 评测矩阵

- 两个来源：lambda=0、lambda=0.25；
- 每个来源：step 5/10/15/20/25/30，共 12 份 checkpoint；
- 每份：4 tasks ×（6 anchor + 6 fixed-L2）= 48 episodes；
- 总计：576 deterministic paired episodes；
- baseline 与 lambda=0.5 已有完全同口径结果，不重复运行。

## 六卡两轮调度

GPU 2--7 各自串行执行两份 checkpoint，因此任意时刻最多六个模型并发。每张卡各跑
一个 lambda=0 和一个 lambda=0.25，首轮顺序交叉，避免某个权重固定绑定某张卡或
固定处在第一轮。预计总耗时 90--120 分钟。

启动器默认只进行 CPU-only 路径/参数验证；显式 `--run` 前统一检查六张卡，任何一张
显存、利用率或温度超阈值便在创建 tmux 和输出目录前整体拒绝。每份 checkpoint 使用
独立目录和日志，第一份非零退出时该 GPU 不进入第二份。
