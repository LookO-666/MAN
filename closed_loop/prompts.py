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
- 避免重复已经尝试过的失败 idea（参考历史经验教训）
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

## 输出要求（JSON 格式）

```json
{{
  "analysis": [
    {{
      "idea_name": "...",
      "diagnosis": "一句话综合判断",
      "verdict": "select" 或 "reject"
    }}
  ],
  "selected": ["idea_name_1", "idea_name_2"],
  "rejected_lessons": [
    {{
      "idea_name": "...",
      "lesson": "为什么被淘汰，对未来有什么启示",
      "reusable_components": "这个 idea 中有什么可回收的部分"
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

请以 JSON 格式输出：
```json
{{
  "round_summary": "本轮总结（1-2句话）",
  "experiment_analyses": [
    {{
      "idea_name": "...",
      "final_acc": 0.0,
      "delta_vs_baseline": 0.0,
      "effective": true,
      "reason": "..."
    }}
  ],
  "lessons_learned": ["教训1", "教训2"],
  "next_round_suggestions": ["建议1", "建议2"]
}}
```
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
