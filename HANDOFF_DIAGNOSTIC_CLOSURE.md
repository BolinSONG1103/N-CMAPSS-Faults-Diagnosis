# 第3章诊断闭环重构交接

## 已完成

1. 冻结研究协议：`docs/第三章诊断闭环研究协议.md`；
2. 新增 unit 级三划分并让管线缓存绑定划分签名；
3. 修复 `experiment_20.m` 的分布内留出泄漏，新结果写入 `t2_dd_v2`，不覆盖旧结果；
4. 新增 `chapter3_diagnostic_lib.m`：健康阈值、持续判决、多标签隔离、一维 GMM 有序等级、物理重构拒识及指标；
5. 新增 `experiment_21_diagnostic_closure.m`：calibration 选择 lambda，ID/OOD 测试，阶段敏感性和五折留一故障族拒识；
6. 新增五组闭环结果图生成器、输出验证器和合成单元测试。
7. 新增约束消融与传感器噪声、固定偏置、逐通道缺失实验，并将其纳入第五组闭环结果图。
8. 新增五个 unit 划分的完整重建入口；每个划分拥有独立管线缓存和 HistGB 模型目录，避免重复实验相互覆盖。

## 科学边界

- 退化等级是训练数据定义的有序严重度，不是 OEM/适航安全阈值；
- 未知故障只通过 leave-one-fault-family-out 验证；
- 单调约束只适用于不可逆 run-to-failure 退化；维修复位必须另行分段；
- 当前为 9 维诊断字典，缺少 HPT flow；
- 旧版 E1“公平性通过”的表述失效，必须以 `t2_dd_v2` 的重跑结果替换。

## 用户本地下一次正式运行

```matlab
cd('D:/PythonProjects/N-CMAPSS-fault-diagnosis/matlab')
results = runtests('tests/test_chapter3_diagnostic_lib.m');
assertSuccess(results)
cd run
run_21_diagnostic_closure
run_22_robustness_ablation
run_build_diagnostic_figures
run_verify_diagnostic_closure
run_23_repeated_splits
```

不得在 smoke 结果上撰写论文数字。正式运行完成后，先检查：

- `outputs/protocol/unit_split_manifest.csv`；
- `outputs/t4_diagnostic_closure/lambda_calibration.csv`；
- `diagnostic_unit_metrics.csv` 与 `unit_bootstrap_summary.csv`；
- `stage_component_sensitivity.csv`；
- `unknown_family_summary.csv`；
- `verification.json`。

## 尚未完成且必须等待正式数据

1. 依据新 CSV 裁决检测、隔离、阶段和拒识是否成立；
2. 对失败项做不移动阈值的机制分析；
3. 运行五划分重建并检查 `t6_repeated_splits/repeated_split_summary.csv`；
4. 将通过验证的结果写回第三章重构稿，并最终替换旧图号和旧 manifest。
