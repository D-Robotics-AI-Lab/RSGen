# mixed DIOR V3 最窄下游入口

仅消费 `MIGC_DIOR_MIXED_FAILED4_RECOVERED4_FULL_POOL_V3` 与独立
`MIGC_DIOR_MIXED_RECOVERY_FULL_POOL_ACCEPTANCE_V3.json` owner Main（MON 根）。
拒绝V1/V2、partial、缺旧encodedSHA复核或新流V3强化证明。四个 a002 hbindingfix P1
（l5xrn/5gb9f/gw4wj/9w7vk，全部 Completed）已由 v3 accept_recovery_stream 逐流 CPU 验收
（1543/1543/1543/1542），v3 collect_pair 全池 12340 + raw symlink view 完成。

保留原 DIOR GT/COCO combine_dior、K9264、LR .04、steps6176/8492、seed42，以及原 CPU
noCUDA/Gloo8 loader、freeze、exactK、训练/评测命令。8个 numerical/guard/loader 文件逐字节同
v1（cpu_gate_tests 强制）；仅 chunk_receipt_contract / prepare_pool(Main 绑定) / protocol
(EFS namespace + source pins) / cpu_gate_tests 四处 V2→V3 定点替换。旧 v1/v2 完全未改。

新数据/评测/checkpoint分别在 EFS datasets/evaluations/checkpoints 的
`RSGen/multibackbone_rot_no_ref_20261106_v1/downstream_dior_mixed_recovery_cpu_v3/MIGC/dior/ours/`。
后续实际8GPU训练仍按 train-eval 命令单独提交准入。
