# 回滚复用 + CTC 延长

## 固定协议

每 2 秒到达新音频，每次 encoder 读取从录音开头累计到当前的音频。所有序号从 1 开始：

| 触发 | 稳定文本前缀 P | 旧草稿 R | 组合策略 |
| --- | --- | --- | --- |
| 第 1 轮 | 空 | 空 | 无任何草稿，正常 target 解码 |
| 第 2 轮 | 空 | 第 1 轮完整内容输出 | 可用 CTC 延长 R |
| 第 3 轮及以后 | 上一轮输出回滚后的前缀 | 被回滚的尾部内容 token | 可用 CTC 延长 R |

对应官方流式参数 `unfixed_chunk_num=2`；默认 `unfixed_token_num=5`。如果输出比回滚长度短，尾部可以是整段输出。Unicode 边界修复可能回滚超过指定 K，因此以实际 prompt 前缀确定切分点。

CTC 输出 C 来自当前累计音频，预测完整内容序列。仅在 C 的开头与 `P + R` 的 token ID 完全一致时，令组合候选为 `R + C[len(P+R):]`；否则使用 R。第 1 轮无条件返回空候选。`rollback_ctc.build_hybrid_candidate()` 不读取 teacher 当前生成结果，不使用未来音频或 oracle 接受位置来决定拼接。

若 P 不能从上一轮实际 token ID 的开头精确切出，`old_prefix_match=false`，这一轮不复用、不追加；保留原始历史供诊断。文本重编码可能改变 token 分段，不能默默按字符数或相同 token 数切出旧草稿。当前门槛较保守；`old_prefix_reusable_rate` 低时应先对照线上 tokenizer/prompt 行为。

本版只实现延长，不修改旧草稿的错误 token。若旧草稿靠前发生错误，追加 token 仍会被首次拒绝挡住。修复候选、拒绝后的重提案及真实 verifier 接入是后续实验。

## 采集与训练

`collect_teacher.py` 调用官方普通贪心流式解码，记录 target 轨迹；在相同 prompt/音频、正确实现的贪心投机验证下，该轨迹可作为离线参考。它不运行真实投机引擎。

新版事件文件保留：

```text
audio, end_sample, text, target_token_ids
```

新版 teacher 文件保留：

```text
schema_version = 2
chunk_index, chunk_samples, end_sample, audio
unfixed_chunk_num, unfixed_token_num, language
prefix_token_ids        # 当前实际 prompt 中的内容前缀
generated_token_ids     # vLLM 实际生成 ID，去除末尾特殊 token
target_token_ids        # prefix_token_ids + generated_token_ids
previous_token_ids     # 上一轮完整内容 ID，第 1 轮为空
old_draft_token_ids     # 第 2 轮完整上一轮；第 3+ 轮为回滚尾部
old_prefix_match       # 是否可以从上一轮实际 ID 精确切出当前前缀
```

前缀 ID 从 `prompt_raw + prefix` 的 token 化结果中切出，并检查 base prompt 边界；vLLM 返回实际 prompt IDs 时再校验末尾内容 ID。生成内容始终保留原始 ID，不先 decode 再 encode。轨迹按录音检查连续性和上一轮历史，draft/teacher 必须覆盖完全相同的事件。

训练保持完整累计序列 CTC loss、冻结 encoder、仅训练线性 CTC 头。新事件直接使用 `target_token_ids`；`--target-token-source recorded` 要求每条 train/dev 记录都有该字段。`auto` 自动选择，混合新旧标签会报错。`text` 仅用于显式旧版/文本标签对照。不要把标签直接截成新增几个 token 再送入无文本条件的整段 CTC。

第 1 轮仍可用于 CTC 声学训练和原始输出诊断，但不提出投机草稿，也不贡献组合收益选模分数。`stream_draft.py` 目前采集每轮原始 CTC 输出及耗时，包括第 1 轮；此耗时不是已部署混合策略的完整运行时间。

## 指标和选模

`score_teacher.py --require-rollback --verify-budget 8` 在 `rollback_comparison` 输出三种策略：`old`、`ctc`、`hybrid`。这三种策略在第 1 轮都没有候选。顶层原有 CTC 指标用于单独诊断，不参与 hybrid 选模。

对每个相同音频/prompt 的事件，以候选与 teacher 本轮生成后缀的最长公共前缀作为可接受内容 token 数，按验证预算截断：

```text
A_K(D) = min(K, longest_common_prefix(D, teacher_suffix))
delta_K = A_K(hybrid) - A_K(old)
```

报告 1/2/4/8 和指定预算，分 `all`、`after_first`、`by_trigger.first/second/later`。空分组的 `events=0`，其零均值不代表有观测。主要字段：

- `mean_delta_accepted_at_8`：相对回滚基线额外可接受 token。
- `old_mean_accepted_at_8` / `hybrid_mean_accepted_at_8`：相同预算的两个接受量。
- `extension_proposed_rate` / `mean_appended_tokens`：提出追加候选的比例和长度。
- `old_fills_budget_rate_at_8`：旧草稿长度已占满预算的比例。
- `old_prefix_reusable_rate`：上一轮实际 ID 可被当前前缀精确切分的比例。

`--selection-metric hybrid --verify-budget 8` 优先最大化 `after_first.mean_delta_accepted_at_8`，相同时比较组合接受量，再比较较低 token 错误率。训练启动前校验 v2 历史、train/dev recorded 标签和 dev teacher 目标一致性；只含第 1 轮的 dev 数据不能用于 hybrid 选模。`auto` 使用 v2 时自动选择 hybrid，旧版 teacher 退回 CTC 单独指标，无 teacher 时按 token 错误率。实际选择与预算写入 run/checkpoint。

该指标仅模拟**本轮首次验证**，不包括 target bonus token、EOS、后续验证、候选长度动态调度或执行开销。旧草稿长度已经达到预算 K 时，追加部分不会在这次验证中产生增益；不代表更大预算或后续验证也没有收益。最终须与仅回滚复用比较真实端到端时间、输出一致性和设备开销。

## 迁移

省略 `--dev-manifest` 使用自动录音划分时，`--dev-teacher` 可以是全量清单对应的 teacher 文件；训练入口先筛选完整验证录音历史，再执行严格事件覆盖检查。明确指定独立 dev 清单时，teacher 必须与该清单完全对应。

音频包和人工整句清单无需修改。旧 events/teacher 没有完整实际 ID 历史，不能靠补几个字段转换成 v2；用同一模型和固定目录重新采集，使用新输出文件名。采集、训练和评测命令见 [README.md](README.md) 和 [DATASET.md](DATASET.md)。保留独立 test 集，不用它选模型。
