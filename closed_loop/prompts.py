"""All LLM prompt templates for the closed-loop system."""

IDEA_GENERATION_PROMPT = """
你是一位经验丰富的 ML 研究者。你的任务是提出改进 CIFAR-10 图像分类的 idea。

## 当前任务
我们有一个 ResNet-18 baseline，在 CIFAR-10 上达到 {baseline_acc}% 的测试准确率。
训练配置：SGD(lr=0.1, momentum=0.9, weight_decay=5e-4), CosineAnnealingLR(T_max=200), 200 epochs。
目标：提出能超越 baseline 的改进方案。

## 历史实验结果
{history_summary}

## 历史经验教训
{experience_lessons}

## 经验约束与指导
{generation_guidance}

## 上一轮的改进建议
{previous_suggestions}

## 要求
请提出 {num_ideas} 个独立的改进 idea。每个 idea 必须包含：
1. **名称**：简短的名称（英文，用于文件命名）
2. **描述**：改什么、为什么改、预期效果（2-3句话）
3. **改动类型**：数据增强 / 正则化 / 学习率策略 / 架构修改 / 损失函数 / 其他
4. **预期影响**：预计提升多少 accuracy

请以 JSON 格式输出，格式如下：
```json
[
  {{
    "name": "cutmix_augmentation",
    "description": "添加 CutMix 数据增强...",
    "category": "数据增强",
    "expected_improvement": "+0.5%"
  }},
  ...
]
```

重要：
- 每个 idea 必须是对 baseline 代码的具体、可实现的修改，不能太抽象
- 严格遵守上面的"经验约束与指导"中的禁止模式，不要重复已确认失败的做法
- 优先考虑"可复用组件"中提到的有潜力的子组件，尝试新的组合方式
- idea 之间应该多样化，覆盖不同的改动类型
"""

CODE_IMPLEMENTATION_PROMPT = """
你是一位 PyTorch 专家。请根据以下 idea 修改 baseline 训练代码。

## Idea
名称：{idea_name}
描述：{idea_description}

## 原始 baseline 代码
```python
{baseline_code}
```

## 要求
1. 输出修改后的完整 Python 代码（不是 patch，是完整文件）
2. 保持原有的命令行参数接口（--epoch, --out_dir, --seed）
3. 保持 results.json 的输出格式不变
4. 只修改与 idea 相关的部分，不要做无关的改动
5. 确保代码可以直接运行，不需要安装额外的包
6. 如果需要导入新的模块，只用 PyTorch 标准库中的

请只输出 Python 代码，不要有任何额外的解释文字。代码用 ```python 和 ``` 包裹。
"""

SCREENING_PROMPT = """
你是一位有经验的 ML 研究者，正在分析 {num_ideas} 个候选 idea 的 5-epoch 预实验结果。
你的目标是选出 {num_survivors} 个最值得投入完整 200-epoch 训练的 idea。

## Baseline（无改动）5 epoch 结果
val_acc={baseline_val_acc}% | train_loss_slope={baseline_train_loss_slope:.4f} | val_loss_slope={baseline_val_loss_slope:.4f} | train_val_gap={baseline_gap:.2f}% | gap_trend={baseline_gap_trend:.2f}

## 候选 Idea 对比表
{candidates_table}

## 分析框架

请对每个 idea 按以下四个维度进行诊断：

1. **当前竞争力**：val_acc 相对 baseline 的 delta。正数=目前领先，负数=目前落后。
   但注意：5 epoch 的绝对 accuracy 不是最重要的，因为很多好方法是"慢热型"的。

2. **学习效率**：train_loss_slope 和 val_loss_slope。
   - 两者都陡=学得快且泛化好（最理想）
   - train 陡但 val 平=可能在记忆训练数据而非真正学习
   - 两者都平=方法可能没有产生实质性影响

3. **泛化潜力**：train_val_gap 和 gap_trend 是最关键的长期预测指标。
   - gap 小 + gap_trend 小或负=泛化性好，长期潜力大（即使当前 acc 低）
   - gap 大 + gap_trend 大且正=过拟合在加剧，长期可能退化（即使当前 acc 高）

4. **训练稳定性**：val_loss_var 和 best_epoch。
   - var 高=训练不稳定，结果不可靠
   - best_epoch 过早=可能已经开始退化

## 4 大象限抢救法则（REFINE vs DISCARD）

对于**未被选中**的 idea，请依据以下法则判断它是否还有"抢救价值"：

- **过拟合潜力股**：gap 在加大，train_loss 下降快，说明模型学到了但泛化不足 → **REFINE**（建议：加正则化、label smoothing、数据增强）
- **慢热稳定型**：gap 极小，var 极小，但当前 acc 偏低，说明方法稳定但学得慢 → **REFINE**（建议：提高学习率、增大模型宽度、延长训练）
- **狂躁巅峰型**：中间 epoch 出现过极高值，var 极大，说明有能力但不稳定 → **REFINE**（建议：加梯度裁剪、换优化器、降低学习率）
- **真·死胡同**：train_loss 和 val_loss 都不下降，所有特征平庸，没有学到东西 → **DISCARD**（写入禁区，未来不再尝试类似方向）

## 输出要求（JSON 格式）

对于每个 idea（无论选中还是淘汰），请提供结构化分析：

```json
{{
  "analysis": [
    {{
      "idea_name": "...",
      "diagnosis": "一句话综合判断",
      "verdict": "select" 或 "reject",
      "decision": "REFINE" 或 "DISCARD"，仅对 verdict=reject 的 idea 必填。verdict=select 时填 "N/A"。,
      "decision_reason": "结合8维特征说明为什么 REFINE 或 DISCARD（一句话）"
    }}
  ],
  "selected": ["idea_name_1", "idea_name_2"],
  "rejected_lessons": [
    {{
      "idea_name": "...",
      "lesson": "为什么被淘汰，对未来有什么启示",
      "reusable_components": "这个 idea 中有什么可回收的部分",
      "failure_type": "overfitting / no_effect / optimization_instability / underperformance",
      "mechanism_hypothesis": "失败的底层原因假设（一句话）",
      "avoid_pattern": "未来应该避免的具体模式（一句话）",
      "decision": "REFINE" 或 "DISCARD",
      "decision_reason": "结合8维特征说明为什么 REFINE 或 DISCARD（一句话）"
    }}
  ]
}}
```
"""

RESULT_ANALYSIS_PROMPT = """
你是一位 ML 研究者，正在分析本轮完整实验（200 epoch）的结果。

## Baseline 结果
最佳 test_acc: {baseline_best_acc}%

## 本轮实验结果
{results_summary}

## 历史最优结果
{best_so_far}

## 历史经验教训
{experience_lessons}

## 请完成以下分析：

1. **效果评估**：每个实验是否超越了 baseline？超越了多少？
2. **原因分析**：为什么有效/无效？从训练曲线中能看出什么？
3. **经验提炼**：本轮实验给出了什么可用于指导下一轮的经验教训？
4. **下一轮建议**：基于累积的所有经验，下一轮应该探索什么方向？
5. **REFINE / DISCARD 判定**：对每个未达预期的实验，判断是否还有打磨空间。

## 4 大象限抢救法则

对每个 effective=false 的实验，请依据以下法则判断它是否还有"抢救价值"：

- **过拟合潜力股**：训练准确率远高于测试准确率，模型学到了但泛化不足 → **REFINE**（建议：加正则化、label smoothing、数据增强）
- **慢热稳定型**：train-val gap 极小且稳定，但最终 acc 偏低，说明方法保守但潜力未释放 → **REFINE**（建议：提高学习率、增大模型宽度、延长训练）
- **狂躁巅峰型**：训练中 acc 曾达到过较高值但最终回落，说明能力有但不稳定 → **REFINE**（建议：加梯度裁剪、换优化器、降低学习率）
- **真·死胡同**：训练曲线平坦无改善，各项指标平庸 → **DISCARD**（写入禁区，未来不再尝试类似方向）

请以 JSON 格式输出。对每个实验，除了效果判断，还需输出结构化的失败/成功分析：

```json
{{
  "round_summary": "本轮总结（1-2句话）",
  "experiment_analyses": [
    {{
      "idea_name": "...",
      "final_acc": 0.0,
      "delta_vs_baseline": 0.0,
      "effective": true,
      "reason": "...",
      "failure_type": "overfitting / no_effect / optimization_instability / underperformance / successful_pattern",
      "mechanism_hypothesis": "为什么成功/失败的底层机制（一句话）",
      "avoid_pattern": "如果失败，未来应避免的模式（一句话）；如果成功，留空",
      "salvageable_parts": ["可保留复用的子组件1", "子组件2"],
      "decision": "REFINE" 或 "DISCARD" 或 "N/A"（成功时填 N/A）",
      "decision_reason": "结合训练曲线特征说明为什么 REFINE 或 DISCARD（一句话）"
    }}
  ],
  "lessons_learned": ["教训1", "教训2"],
  "next_round_suggestions": ["建议1", "建议2"]
}}
```
"""

IDEA_REFINEMENT_PROMPT = """
你是一位经验丰富的 ML 研究者。你正在对一个有潜力但未达预期的 idea 进行"打磨"。

## 原始 Idea
名称：{original_idea_name}
描述：{original_idea_description}
类型：{original_idea_category}

## 上一轮实验结果
{experiment_results}

## 抢救原因（为什么值得继续打磨）
{decision_reason}

## 失败诊断
失败类型：{failure_type}
机制假设：{mechanism_hypothesis}

## 任务要求
请**不要提出全新方向**，而是针对上述原始 idea 提出 {num_refinements} 个改良变体（v2.0）。
每个变体必须：
1. 保留原始 idea 的核心思想
2. 针对 decision_reason 中指出的具体问题进行修复
3. 给出不同的修复策略（例如：一个加正则，一个调超参，一个换组合方式）

请以 JSON 格式输出：
```json
[
  {{
    "name": "{original_idea_name}_v2a",
    "description": "在原始方案基础上...",
    "category": "...",
    "expected_improvement": "+0.3%",
    "refinement_strategy": "针对什么问题做了什么修复（一句话）"
  }},
  ...
]
```

重要：
- 名称必须以原始 idea 名称为前缀（如 xxx_v2a, xxx_v2b）
- 每个变体的修复策略必须不同
- 不要偏离原始方向，只做"手术式"的定向改良
"""

NO_MINI_EXP_SCREENING_PROMPT = """
你是一位有经验的 ML 研究者。以下是 {num_ideas} 个候选改进 idea，请直接根据你的专业判断选出 {num_survivors} 个最值得尝试的。

## Baseline
ResNet-18 on CIFAR-10, 95.51% test accuracy.
SGD(lr=0.1, momentum=0.9, wd=5e-4), CosineAnnealingLR(T_max=200), 200 epochs.

## 候选 Ideas
{ideas_list}

## 历史经验教训
{experience_lessons}

请以 JSON 格式输出：
```json
{{
  "selected": ["idea_name_1", "idea_name_2"],
  "reasoning": "选择理由"
}}
```
"""
