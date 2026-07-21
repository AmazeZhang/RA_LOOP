# OpenVLA-OFT union / LIBERO-Plus libero_10 Robot-init baseline

> **模型身份更正（2026-07-20）**：这里使用的四套件联合 checkpoint 在
> LIBERO-Plus 论文与官方仓库中记作 **OpenVLA-OFT_m（mix-SFT）**，不是官方
> Robot 31.9% 对应的分套件 OpenVLA-OFT。OFT_m 的官方 Robot 指标是 21.7%。
> 四套件完整 1550 条结果和证据见
> `docs/BASELINE_OFTM_ROBOT1550_2026-07-20.md`。

## 1. 结论

2026-07-18 完成官方 OpenVLA-OFT union checkpoint 在 LIBERO-Plus
`libero_10 / Robot Initial States` 完整切片上的评测：

```text
Success: 141 / 393
Rate:    35.877863%（报告为 35.9%）
```

严格汇总检查通过：覆盖 `393/393`、missing=0；7 个 shard 均完整，无重复、
无任务串片，manifest SHA、checkpoint、suite、seed 和 render backend 一致。

这是可信的 **OFT_m / libero_10 Robot-init 切片 baseline**，不是跨四个 suite
的完整 Robot-init 指标，也不是 LIBERO-Plus 全 10,030 任务指标。因此不能把
35.9% 直接等同或替代官方分套件 OpenVLA-OFT 的跨 suite Robot 31.9%。

## 2. 评测协议

| 项目 | 值 |
|---|---|
| Checkpoint | `/home/imc/models/ra-loop/openvla-oft-union` |
| Suite | `libero_10` |
| Category | `Robot Initial States` |
| Tasks / episodes | 393 / 393（每任务 1 episode） |
| Seed | 7 |
| Shards | 7 个确定性、不重叠 manifest |
| Physical GPUs | 1--7；GPU 0 未使用 |
| Render backend | OSMesa；未使用 EGL |
| Python | `/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python` |
| Python isolation | `PYTHONNOUSERSITE=1` |
| Launcher | `eval/launch_union_robot393_tmux.sh --resume` |
| Tmux session | `ra_union_robot393` |

模型只做推理；此次运行没有训练或修改 checkpoint。

## 3. 按难度等级

| Difficulty | Success / Tasks | Rate |
|---:|---:|---:|
| 1 | 54 / 64 | 84.4% |
| 2 | 57 / 77 | 74.0% |
| 3 | 21 / 100 | 21.0% |
| 4 | 6 / 83 | 7.2% |
| 5 | 3 / 69 | 4.3% |

主要现象是 difficulty 2 到 3 出现明显断层，difficulty 4/5 几乎失效。这比
单独看总成功率更能定位 Robot-init 鲁棒性缺口。

## 4. 按基础任务

| Base task | Success / Tasks | Rate |
|---|---:|---:|
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 36 / 44 | 81.8% |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 18 / 44 | 40.9% |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 19 / 42 | 45.2% |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 6 / 34 | 17.6% |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 9 / 39 | 23.1% |
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 12 / 41 | 29.3% |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 12 / 35 | 34.3% |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 6 / 38 | 15.8% |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 19 / 42 | 45.2% |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 4 / 34 | 11.8% |

不同基础任务差异很大：stove + 单 moka pot 为 81.8%，book/caddy 仅 11.8%。
后续不能只按整体 Robot-init rate 评估方法改进，还应报告任务级变化，避免提升由
少数容易任务主导。

## 5. 分片结果与文件校验

| Shard | GPU | Success / Tasks | Rate | tqdm elapsed | results.jsonl SHA-256 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 20 / 57 | 35.1% | 1:41:32 | `dd9fe6975d81496014185424cf4a178af7da8c541420a7af5ec2efdb6b83581b` |
| 1 | 2 | 21 / 56 | 37.5% | 1:36:07 | `77d3cd2c13cd282d33979c87d431c6ef1c6bfdcc36b569824de62379c9cd165f` |
| 2 | 3 | 17 / 56 | 30.4% | 1:44:39 | `fe59a313f7f1befbb489c51520a8888f7dcfb734e6ab20a6eda3279ceacb19cc` |
| 3 | 4 | 15 / 56 | 26.8% | 1:44:07 | `0ec082415581848df72fe620e4758ea48ded846639225f2142500f61d1c7a79b` |
| 4 | 5 | 22 / 56 | 39.3% | 1:35:57 | `5cb38a7fabee40c96ffb1eb157e883d9badeeb6a528cd79fefe28b0d2d0314b2` |
| 5 | 6 | 26 / 56 | 46.4% | 1:33:23 | `cab7c5df2a1a6bf233116ef77558b59bd5669f10a742ac86d66b055c5695da17` |
| 6 | 7 | 20 / 56 | 35.7% | 1:40:51 | `6bf7b8d37fa0921c53aced43fc58eddbee3b52538b5bf2988737e32f16f95d7c` |

按 shard 0 到 6 的原始字节依次拼接后：

```text
SHA-256: 7859ec3a2809e6d6d4145f76eca8d5d0fe8c31302c4e13883ce78fb099ad2ead
```

分片 rate 的差异来自任务组成不同，不能当作 GPU 间性能差异。

## 6. 产物与运行结束状态

结果根目录：

```text
/home/imc/yzy/RA_LOOP/logs/openvla_oft_union_libero10_robot_full393_7gpu_20260718
```

每个 shard 目录包含 `results.jsonl`、`run_metadata.json`、upstream 日志和视频。
7 个 evaluator 均 exit code 0；结束后无 evaluator 进程，GPU 1--7 回到空闲。
tmux 会话暂时保留为 7 个空 shell，便于人工检查。

此前 20 条 extreme pilot 的 45.0% 由每个基础任务的最低/最高难度样本构成，抽样
分布与完整 393 条不同；它只作为工程 smoke，不覆盖或替代本记录的正式结果。
