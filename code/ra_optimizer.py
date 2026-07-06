"""
ra_optimizer.py
=================================================================
RA-LOOP 的核心算法实现。

设计原则
-----------------------------------------------------------------
1. 侵入面最小: 不 fork RIPT-VLA 仓库, 而是通过 monkey-patch 或
   `Hydra _target_` 反射方式替换掉两个类:
     - RolloutGenerator          → RAPerturbedRolloutGenerator
     - RLOptimizerOpenVLAOFT     → RAOptimizer
   同时提供一个 stub RobustnessAwareReward, 保持
   config/reward_function/robustness_aware.yaml 的 _target_ 可实例化
   （真正的 R_consistency / R_recovery 计算在 RAOptimizer 里完成）。

2. 与官方 LOOP 行为保持严格向后兼容:
     - lambda_consistency = 0 且 lambda_recovery = 0 且 perturbation.enabled = false
       时, RAOptimizer 的输出与 RLOptimizerOpenVLAOFT 数值一致（除 dtype 抖动外）。
   这一点在 unit test 里通过 `torch.allclose(advantage, expected)` 断言。

3. 数值稳定性:
     - R_consistency 使用 -||a - a_mean||^2 而不是 cos_sim,
       因为在训练早期 a 全零或全同, cos_sim 会 NaN。
     - λ_c, λ_r 由 warmup 线性 ramp-up, 避免污染 baseline。
     - Perturbation warmup 亦线性 ramp。

4. 单机-多机通信: 完全沿用 RIPT-VLA 原有的 dist.all_reduce/all_gather 逻辑,
   不引入新的通信 op。

引用（2026-07-06 线上核对）
-----------------------------------------------------------------
- LOOP 数学: Chen et al., "Interactive Post-Training for Vision-Language-Action Models",
  arXiv:2505.17016, §3.2
- OpenVLA-OFT bidirectional attention: Kim et al., 2025-02,
  https://openvla-oft.github.io
- LIBERO-Plus 7 维扰动定义: sylvestf/LIBERO-plus README (main branch, 2026-07-06)
=================================================================
"""

from __future__ import annotations

import time
import copy
import gc
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

# 从 RIPT-VLA 仓库导入依赖。这里的 import path 假设:
#   PYTHONPATH 里包含 repos/ript-vla/, 即 RA-LOOP setup/step02_repos.sh 的默认结果。
from ript.algos.rl_optimizers.rl_optimizer_openvla_oft import RLOptimizerOpenVLAOFT
from ript.algos.rl_optimizers.rollout_generator import (
    RolloutGenerator,
    compute_hash_from_state,
)
from ript.algos.reward_functions.libero import BaseRewardFunction
from ript.env_runner.openvla_oft_libero_runner import (
    get_vla_action_batch,
    laplace_log_prob,
)

ACTION_DIM = 7  # 与 RIPT-VLA 一致: (x, y, z, rx, ry, rz, gripper)


# =================================================================
# 1. Stub reward: 只负责成功位, 用于兼容 Hydra 的 reward_function 组
# =================================================================
class RobustnessAwareReward(BaseRewardFunction):
    """
    Stub reward function.
    真正的 R_consistency / R_recovery 计算发生在 RAOptimizer.optimize() 内,
    因为它们需要"同 init_hash 下 K 条 rollout 的联合信息", 单条 rollout
    的 compute_reward 接口无法承担。

    这里只输出 R_success (与 SuccessReward 等价), 并把 lambda / metric 存起来,
    让 RAOptimizer 在初始化时可以读回。
    """

    def __init__(
        self,
        lambda_consistency: float = 0.5,
        lambda_recovery: float = 0.5,
        consistency_metric: str = "action_l2",
        consistency_horizon: int = 30,
        recovery_bonus: float = 1.0,
    ):
        super().__init__()
        # 记下参数, RAOptimizer 会从 model.cfg 或 reward_function 对象里读回来
        self.lambda_consistency = lambda_consistency
        self.lambda_recovery = lambda_recovery
        self.consistency_metric = consistency_metric
        self.consistency_horizon = consistency_horizon
        self.recovery_bonus = recovery_bonus

    def compute_reward(
        self,
        rollout_idx: int,
        rollout_episode: Dict[str, Any],
        ground_truth_batch: Dict[str, Any],
    ) -> float:
        # 仅返回 success bit, 保持与 SuccessReward 的接口一致
        return 1.0 if rollout_episode.get("success", False) else 0.0


# =================================================================
# 2. Perturbation utilities
# =================================================================
def _apply_perturbation_to_init(
    init_state: np.ndarray,
    perturb_type: str,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    对 LIBERO 环境的 init_state 向量应用扰动。

    LIBERO init_state 排布（robosuite 底层, 见 LIBERO-plus README）:
      [ robot_joint_pos (7) , robot_gripper (2) , obj_pose (N*7) ]
    这里我们只做"训练侧可负担"的四类扰动:
      - robot_init: 对前 7 维 (关节角) 加 gaussian 噪声, std=strength (rad)
      - camera    : 相机外参不写在 init_state 里, 通过 rollout 时的环境属性调整,
                    此函数对相机扰动返回原状态, 由 env_runner 端另行处理。
      - light     : 同上, 由 env_runner 端调 sim 的光源强度。
      - layout    : 对 obj_pose 的 xy 坐标 (第 9-10 维、第 16-17 维 …) 加噪声。

    Args:
        init_state: shape (D,), numpy float32/64
        perturb_type: 4 类之一
        strength: 扰动幅度 (rad / m / 无量纲)
        rng: numpy Generator, 保证多机可复现

    Returns:
        新的 init_state, shape (D,)
    """
    out = init_state.copy()

    if perturb_type == "robot_init":
        # 前 7 维为机械臂 7 关节 q; 若 D<7 则直接返回
        if out.shape[0] >= 7:
            out[:7] = out[:7] + rng.normal(0.0, strength, size=7).astype(out.dtype)
    elif perturb_type == "layout":
        # obj pose 从第 9 维开始, 每 7 个一段 (x, y, z, qx, qy, qz, qw)
        obj_start = 9
        if out.shape[0] > obj_start:
            num_objs = (out.shape[0] - obj_start) // 7
            for k in range(num_objs):
                idx_x = obj_start + 7 * k + 0
                idx_y = obj_start + 7 * k + 1
                if idx_y < out.shape[0]:
                    out[idx_x] += rng.normal(0.0, strength)
                    out[idx_y] += rng.normal(0.0, strength)
    else:
        # camera / light 由 env_runner 端处理, 这里保持不变
        pass

    return out


class RAPerturbedRolloutGenerator(RolloutGenerator):
    """
    在原 RolloutGenerator 之上, 把每次 rloo_batch 内的 K 条 rollout
    切成 num_perturb_groups 组, 每组一个扰动 seed。

    - 若 config.perturbation.enabled = False, 行为退化为完全等价于父类。
    - K 必须能被 num_perturb_groups 整除。
    - 记录 (init_hash, group_id) 到 episode['perturb_group'], 供 RAOptimizer 分组用。
    """

    def __init__(self, *args, perturbation_cfg: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # perturbation_cfg 来自 algo.perturbation 那一段
        self.perturbation_cfg = perturbation_cfg or {"enabled": False}
        self.global_step = 0  # 由 RAOptimizer 每次 optimize 调用后 +=1

        if self.perturbation_cfg.get("enabled", False):
            assert (
                self.rloo_batch_size
                % self.perturbation_cfg["num_perturb_groups"]
                == 0
            ), (
                f"rloo_batch_size={self.rloo_batch_size} 必须能被 "
                f"num_perturb_groups={self.perturbation_cfg['num_perturb_groups']} 整除"
            )

    def _current_strength(self, base: float) -> float:
        """warmup 线性 ramp: 前 warmup_steps 内从 0 → base"""
        warmup = self.perturbation_cfg.get("warmup_steps", 0)
        if warmup <= 0:
            return base
        return base * min(1.0, self.global_step / warmup)

    def _build_perturbed_init_batch(
        self, base_init: np.ndarray, rank: int
    ) -> Tuple[np.ndarray, List[int]]:
        """
        输入: shape (D,) 的原 init_state
        输出:
          env_init_states: shape (rloo_batch_size, D)
          perturb_group_ids: 长度 rloo_batch_size 的整型列表, 表明每条属于哪一组
        """
        K = self.rloo_batch_size
        if not self.perturbation_cfg.get("enabled", False):
            return (
                np.tile(base_init, (K, 1)),
                [0] * K,
            )

        G = self.perturbation_cfg["num_perturb_groups"]
        per_group = K // G
        # 用 (rank, global_step, group_idx) 组合成 seed, 保证跨进程可复现且不同 rank 扰动不同
        types = self.perturbation_cfg["types"]

        out = np.zeros((K, base_init.shape[0]), dtype=base_init.dtype)
        gids: List[int] = []
        for g in range(G):
            seed = (rank * 10007 + self.global_step * 131 + g * 17) & 0x7FFFFFFF
            rng = np.random.default_rng(seed)
            ptype = types[g % len(types)]
            if ptype == "robot_init":
                strength = self._current_strength(
                    self.perturbation_cfg["robot_init_std"]
                )
            elif ptype == "layout":
                strength = self._current_strength(
                    self.perturbation_cfg["layout_std"]
                )
            else:
                # camera / light 走 env_runner 那侧, 此处只标记 group
                strength = 0.0
            perturbed = _apply_perturbation_to_init(base_init, ptype, strength, rng)
            for j in range(per_group):
                out[g * per_group + j] = perturbed
                gids.append(g)
        return out, gids

    def generate_rollouts(self, model, batch, data_iterator, dataloader):
        """
        重写父类的 generate_rollouts, 只在"如何构造 env_init_states"和
        "在 episode 上打 perturb_group 标签"两处改动, 其余 100% 沿用父类逻辑。

        为避免重复维护数百行, 我们直接调用父类实现 —— 但那样没法插入扰动。
        因此这里 copy 父类的主循环, 只改必要行。若父类 API 改动, 需要重新对齐。
        """
        # ---- 与父类一致的初始化 ----
        from ript.algos.rl_optimizers.file_counter import reset_global_counter

        reset_global_counter(self.file_counter)

        all_successes: List[bool] = []
        all_scores: List[float] = []
        all_episodes: List[Dict[str, Any]] = []
        all_task_ids: List[int] = []

        demo_batch_size = batch["task_id"].shape[0]
        valid_samples = 0

        current_batch = batch
        batch_index = 0
        samples_checked = 0
        early_stop = False
        rank = dist.get_rank() if dist.is_initialized() else 0

        while valid_samples < demo_batch_size and not early_stop:
            samples_checked += 1

            if batch_index >= current_batch["task_id"].shape[0]:
                try:
                    current_batch = next(data_iterator)
                    batch_index = 0
                except StopIteration:
                    data_iterator = iter(dataloader)
                    current_batch = next(data_iterator)
                    batch_index = 0

            sample_task_id = current_batch["task_id"][batch_index].item()
            task_idx = sample_task_id
            task_name = self.task_names_to_use[task_idx]
            created_env = self.created_envs[task_idx]

            # ---- 关键改动: 构造扰动后的 env_init_states ----
            if self.use_val_init:
                # val_init 分支我们不做扰动, 因为 val_init 本身就已经是随机
                _, env_id, env_num = created_env
                all_env_init_states = self.env_runner.benchmark.get_task_init_states(
                    env_id
                )
                all_env_num = len(all_env_init_states)
                if self.mix_val_init_in_rloo:
                    select_idx = np.random.randint(
                        0, all_env_num, size=self.rloo_batch_size
                    )
                    env_init_states = all_env_init_states[select_idx]
                else:
                    select_idx = np.random.randint(0, all_env_num)
                    env_init_state = all_env_init_states[select_idx]
                    env_init_states = np.tile(
                        env_init_state, (self.rloo_batch_size, 1)
                    )
                gids = [0] * self.rloo_batch_size
                random_init = False
            else:
                sample_states = current_batch["init_state"]
                init_state = sample_states["states"][batch_index, 0][
                    sample_states["pad_mask"][batch_index]
                ]
                base_init = init_state.cpu().numpy()
                env_init_states, gids = self._build_perturbed_init_batch(
                    base_init, rank
                )
                random_init = False

            init_hash = compute_hash_from_state(current_batch["init_state"], batch_index)

            if self.enable_rollout_stats_tracking and init_hash in self.rollout_stats:
                new_rollout_successes = self.rollout_stats[init_hash][
                    -self.rloo_batch_size :
                ]
                if all(s == 1 for s in new_rollout_successes):
                    batch_index += 1
                    self.rollout_skip_cnt[init_hash] += 1
                    if (
                        self.rollout_skip_cnt[init_hash]
                        > self.rollout_skip_threshold
                    ):
                        del self.rollout_stats[init_hash]
                    continue
            else:
                self.rollout_stats[init_hash] = []
                self.rollout_skip_cnt[init_hash] = 0

            # ---- 执行 rollout ----
            rollout_env = self.env_runner.run_policy_in_env(
                task_name,
                model,
                env_init_states,
                render=False,
                created_env=created_env,
                random_init=random_init,
            )
            sample_successes: List[bool] = []
            sample_scores: List[float] = []
            sample_episodes: List[Dict[str, Any]] = []
            sample_task_ids: List[int] = []

            for k in range(self.rloo_batch_size):
                try:
                    success, total_reward, episode = next(rollout_env)
                except StopIteration:
                    break
                episode["init_hash"] = init_hash
                episode["perturb_group"] = gids[k]  # ← RA-LOOP 关键标签
                episode["is_perturbed"] = gids[k] > 0  # group 0 视为"未扰动 anchor"
                sample_successes.append(success)
                sample_scores.append(total_reward)
                sample_episodes.append(episode)
                sample_task_ids.append(task_idx)
                self.rollout_stats[init_hash].append(int(success))

            if (
                self.enable_dynamic_sampling
                and len(sample_successes) > 0
                and (
                    all(s == 0 for s in sample_successes)
                    or all(s == 1 for s in sample_successes)
                )
            ):
                pass  # 丢弃
            else:
                all_successes.extend(sample_successes)
                all_scores.extend(sample_scores)
                all_episodes.extend(sample_episodes)
                all_task_ids.extend(sample_task_ids)
                valid_samples += 1
                self.file_counter.update(1)

            current_global = self.file_counter.get()
            if current_global >= self.global_rollout_demo_threshold:
                early_stop = True
            batch_index += 1

        # ---- padding (与父类一致) ----
        target_rollouts = self.demo_batch_size * self.rloo_batch_size
        valid_mask = [True] * len(all_successes)
        if len(all_successes) < target_rollouts:
            num_pad = target_rollouts - len(all_successes)
            if len(all_successes) > 0:
                last_success = all_successes[-1]
                last_score = all_scores[-1]
                last_episode = all_episodes[-1]
                last_task_id = all_task_ids[-1]
            else:
                last_success = False
                last_score = 0.0
                last_episode = {
                    "context_tokens": [0],
                    "action_indices": [0],
                    "policy_inference_steps": [1],
                    "perturb_group": 0,
                    "is_perturbed": False,
                }
                last_task_id = 0
            for _ in range(num_pad):
                all_successes.append(last_success)
                all_scores.append(last_score)
                all_episodes.append(last_episode)
                all_task_ids.append(last_task_id)
                valid_mask.append(False)

        device = model.device
        for i, success in enumerate(all_successes):
            all_episodes[i]["success"] = success

        valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool).to(device)
        if dist.is_initialized():
            dist.barrier()

        # 每次 rollout 结束, 全局步数 +1 (供 warmup 用)
        self.global_step += 1

        return (
            all_episodes,
            torch.tensor(all_task_ids).to(device),
            valid_mask_t,
            samples_checked,
        )


# =================================================================
# 3. RAOptimizer: 在 RLOptimizerOpenVLAOFT 之上加 R_consistency / R_recovery
# =================================================================
class RAOptimizer(RLOptimizerOpenVLAOFT):
    """
    继承自 RIPT-VLA 官方 RLOptimizerOpenVLAOFT。

    改动只集中在 optimize() 计算 reward 那一段:
      - 原实现: rlhf_reward = R_success (一维)
      - 本实现: rlhf_reward = R_success + λ_c·R_consistency + λ_r·R_recovery
        其余 (advantage, PPO clip, gradient sync) 全部继承。

    这样做的好处:
      1. 逻辑集中: 消融 λ_c=0, λ_r=0 立刻退化为 vanilla LOOP。
      2. 不动 PPO 数值路径, 降低数值 bug 面。
    """

    def __init__(
        self,
        *args,
        lambda_consistency: float = 0.5,
        lambda_recovery: float = 0.5,
        consistency_metric: str = "action_l2",
        consistency_horizon: int = 30,
        recovery_bonus: float = 1.0,
        robustness_warmup_steps: int = 200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_consistency = lambda_consistency
        self.lambda_recovery = lambda_recovery
        self.consistency_metric = consistency_metric
        self.consistency_horizon = consistency_horizon
        self.recovery_bonus = recovery_bonus
        self.robustness_warmup_steps = robustness_warmup_steps
        self._opt_step = 0  # 每次 optimize() 结束 +1, 供 λ warmup 用

    # -------------------------------
    def _lambda_ramp(self, base: float) -> float:
        if self.robustness_warmup_steps <= 0:
            return base
        return base * min(1.0, self._opt_step / self.robustness_warmup_steps)

    # -------------------------------
    def _compute_consistency_reward(
        self, group_actions: List[np.ndarray]
    ) -> List[float]:
        """
        输入: group_actions[k] = shape (T_k, action_dim), 属于同一 init_hash+perturb_group
              的 K/G 条 rollout 的 action 序列
        输出: 与 group_actions 同长度的 float 列表, 每条 rollout 对应一个 R_consistency

        R_consistency_i = - mean_over_t ||a_i(t) - a_mean(t)||_2^2
        - 只取前 consistency_horizon 步（0 表示整条）
        - 若组内只有 1 条 rollout, 返回 0.0（无参考均值）
        - 若两条 rollout 长度不一致, 取最短长度对齐
        """
        n = len(group_actions)
        if n <= 1:
            return [0.0] * n

        # 对齐长度
        min_T = min(a.shape[0] for a in group_actions)
        if min_T == 0:
            return [0.0] * n
        if self.consistency_horizon > 0:
            min_T = min(min_T, self.consistency_horizon)

        stacked = np.stack([a[:min_T] for a in group_actions], axis=0)  # (n, T, D)
        mean_a = stacked.mean(axis=0, keepdims=True)                    # (1, T, D)

        if self.consistency_metric == "action_l2":
            diff2 = np.mean((stacked - mean_a) ** 2, axis=(1, 2))         # (n,)
            r = -diff2
        elif self.consistency_metric == "action_cos":
            # (n, T*D) cos_sim with mean
            flat = stacked.reshape(n, -1)
            m = mean_a.reshape(1, -1)
            denom = np.linalg.norm(flat, axis=1) * np.linalg.norm(m, axis=1) + 1e-8
            r = (flat * m).sum(axis=1) / denom
        else:
            raise ValueError(f"Unknown consistency_metric: {self.consistency_metric}")

        return [float(x) for x in r]

    # -------------------------------
    def _compute_augmented_rewards(
        self,
        all_episodes: List[Dict[str, Any]],
        base_scores: List[float],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        把 base R_success 列表和 all_episodes 组合出 augmented reward, 并返回统计量。
        分组维度: (init_hash, perturb_group)。若无 perturb_group（vanilla 兼容路径）,
        整个 init_hash 视为一组。
        """
        # 分组
        groups: Dict[Tuple[str, int], List[int]] = {}
        for i, ep in enumerate(all_episodes):
            key = (ep.get("init_hash", "N/A"), ep.get("perturb_group", 0))
            groups.setdefault(key, []).append(i)

        # 计算 consistency
        R_cons = [0.0] * len(all_episodes)
        for key, idxs in groups.items():
            # 取每条 rollout 的 normalized_actions (T, D)
            group_actions: List[np.ndarray] = []
            for i in idxs:
                acts = all_episodes[i].get("actions_normalized", None)
                if acts is None or len(acts) == 0:
                    group_actions.append(np.zeros((0, ACTION_DIM), dtype=np.float32))
                else:
                    a_np = np.asarray(acts, dtype=np.float32)
                    if a_np.ndim == 3:  # (T, 1, D) or (T, K, D)
                        a_np = a_np.reshape(a_np.shape[0], -1)[:, :ACTION_DIM]
                    group_actions.append(a_np)
            r_list = self._compute_consistency_reward(group_actions)
            for i, r in zip(idxs, r_list):
                R_cons[i] = r

        # Recovery: 只在 is_perturbed=True 时给正 bonus, 其他给 0
        R_rec = [
            self.recovery_bonus if (ep.get("is_perturbed", False) and ep.get("success", False)) else 0.0
            for ep in all_episodes
        ]

        # 组装
        lc = self._lambda_ramp(self.lambda_consistency)
        lr = self._lambda_ramp(self.lambda_recovery)

        total = [
            base_scores[i] + lc * R_cons[i] + lr * R_rec[i]
            for i in range(len(all_episodes))
        ]

        stats = {
            "mean_R_success":     float(np.mean(base_scores)) if base_scores else 0.0,
            "mean_R_consistency": float(np.mean(R_cons)) if R_cons else 0.0,
            "mean_R_recovery":    float(np.mean(R_rec)) if R_rec else 0.0,
            "lambda_c_effective": lc,
            "lambda_r_effective": lr,
        }
        return torch.tensor(total, dtype=torch.float32).to(device), stats

    # -------------------------------
    def optimize(self, model, batch, optimizers, data_iterator=None, dataloader=None):
        """
        与父类 optimize() 完全一致, 除了 rlhf_reward 的计算。
        为了不复制 200 行 PPO 代码, 我们采用 "monkey patch reward" 的方式:
          1) 先调 rollout_generator.generate_rollouts 得到 all_episodes
          2) 用 _compute_augmented_rewards 得到 augmented rlhf_reward
          3) 把 reward_function.compute_reward 临时替换成 "查表返回增广 reward"
          4) 调 super().optimize() 走 PPO
          5) 恢复 reward_function
        这样 PPO 数值路径完全复用父类, 避免维护漂移。
        """
        # ---- Step 0: 先跑一次 rollout, 拿到 episodes ----
        with torch.no_grad():
            all_episodes, all_task_ids, valid_mask, samples_checked = (
                self.rollout_generator.generate_rollouts(
                    model, batch, data_iterator, dataloader
                )
            )

        # ---- Step 1: 计算 base R_success ----
        base_scores = [
            self.reward_function.compute_reward(i, all_episodes[i], batch)
            for i in range(len(all_episodes))
        ]

        # ---- Step 2: 计算 augmented reward ----
        augmented, stats = self._compute_augmented_rewards(
            all_episodes, base_scores, model.device
        )

        # ---- Step 3: 用一个 stub reward_function 把 rollouts 结果打回 super() ----
        class _LookupReward(BaseRewardFunction):
            def __init__(self, table):
                super().__init__()
                self.table = table

            def compute_reward(self, rollout_idx, rollout_episode, ground_truth_batch):
                return float(self.table[rollout_idx].item())

        original_reward_fn = self.reward_function
        original_rollout_gen_generate = self.rollout_generator.generate_rollouts
        original_rollout_gen_reset = getattr(
            self.rollout_generator.file_counter, "get", lambda: 0
        )

        self.reward_function = _LookupReward(augmented)

        # 让 super().optimize 复用我们已经跑好的 rollout: 覆盖 generate_rollouts 一次即可
        _cached = (all_episodes, all_task_ids, valid_mask, samples_checked)

        def _fake_generate_rollouts(*args, **kwargs):
            return _cached

        self.rollout_generator.generate_rollouts = _fake_generate_rollouts

        try:
            metrics = super().optimize(
                model, batch, optimizers, data_iterator=data_iterator, dataloader=dataloader
            )
        finally:
            # 恢复
            self.reward_function = original_reward_fn
            self.rollout_generator.generate_rollouts = original_rollout_gen_generate

        # ---- Step 4: 把 RA-LOOP 特有指标写进 metrics, 供 wandb 记录 ----
        metrics.update(stats)
        metrics["ra_opt_step"] = float(self._opt_step)
        self._opt_step += 1

        return metrics


# =================================================================
# 4. Factory: 供 train_ript_openvla_oft.py 侧调用, 决定用不用 RA-LOOP
# =================================================================
def build_rl_optimizer_from_cfg(cfg, rollout_generator, reward_function):
    """
    Hydra 侧调用示例:
        rl_opt = build_rl_optimizer_from_cfg(cfg, rg, rf)
    只要 cfg.algo.use_ra_loop=True 就返回 RAOptimizer, 否则退回官方 RLOptimizerOpenVLAOFT。
    这样在 train script 里只需要:
        from code.ra_optimizer import build_rl_optimizer_from_cfg
        rl_opt = build_rl_optimizer_from_cfg(cfg, rg, rf)
    """
    common_kwargs = dict(
        rollout_generator=rollout_generator,
        reward_function=reward_function,
        ppo_clip_range=cfg.algo.ppo_clip_range,
        ppo_clip_high=cfg.algo.ppo_clip_high,
        num_ppo_epochs=cfg.algo.num_ppo_epochs,
        gradient_accumulation_steps=cfg.algo.gradient_accumulation_steps,
        grad_norm_clip_model=cfg.training.grad_clip,
        grad_norm_clip_header=cfg.training.grad_clip,
        log_prob_mode=cfg.algo.log_prob_mode,
    )

    if not cfg.algo.get("use_ra_loop", False):
        return RLOptimizerOpenVLAOFT(**common_kwargs)

    return RAOptimizer(
        **common_kwargs,
        lambda_consistency=cfg.algo.robustness.lambda_consistency,
        lambda_recovery=cfg.algo.robustness.lambda_recovery,
        consistency_metric=cfg.algo.robustness.consistency_metric,
        consistency_horizon=cfg.algo.robustness.consistency_horizon,
        recovery_bonus=cfg.algo.robustness.recovery_bonus,
        robustness_warmup_steps=cfg.algo.perturbation.get("warmup_steps", 0),
    )
