# CTC 完整前缀与回滚尾部局部对齐消融

## 目的与输入

在**同一个 checkpoint、同一组 dev 事件、同一验证预算**下，比较候选利用规则。无需新增训练；只要已有草稿 JSONL，评分仅需 Python 标准库，不加载模型或音频。

优先直接使用该训练目录的 `best_predictions.jsonl`：它是保存 `best_head.pt` 时对应的完整 dev 草稿。若使用其他 checkpoint，须使用它实际生成的 dev 草稿，不要混用。

```bash
python compare_alignment.py \
  --drafts data/ctc-runs/aishell-expanded-coverage10-001/best_predictions.jsonl \
  --teacher data/ctc-traces-expanded/dev_teacher.jsonl \
  --verify-budget 8 \
  --output data/ctc-runs/aishell-expanded-coverage10-001/dev_alignment.json \
  --events-out data/ctc-runs/aishell-expanded-coverage10-001/dev_alignment_events.jsonl
```

若没有对应的 dev 草稿，先用已有 checkpoint 推理一次（这不是训练）：

```bash
python stream_draft.py \
  --checkpoint data/ctc-runs/aishell-expanded-coverage10-001/best_head.pt \
  --manifest data/ctc-traces-expanded/dev_events.jsonl \
  --output data/ctc-runs/aishell-expanded-coverage10-001/dev_drafts.jsonl \
  --device npu
```

然后将比较命令的 `--drafts` 改成这个文件。teacher 必须与草稿使用相同绝对音频路径和 chunk 边界，完整覆盖从第 1 轮开始的历史；不接受只截取后几轮的轨迹。比较结果及逐事件输出须为新文件。

## 两种规则

记当前稳定前缀为 P、旧回滚草稿为 R、完整 CTC 输出为 C：

| 策略 | 追加条件 |
| --- | --- |
| `full_prefix` | C 从头精确匹配 P+R，取其后缀追加；与现有生产评分规则一致 |
| `rollback_tail_local` | 在整个 C 中搜索完整 R 的连续精确匹配；唯一命中时取匹配结束后的后缀追加 |

局部规则忽略 P 的匹配情况，但不改变 R 内部任何 token。第 2 轮的 R 仍是上一轮完整输出，第 3 轮起才是实际回滚尾部。两种策略都保留“第 1 轮无草稿、历史不能复用时无草稿”的协议。

局部锚点是**整段 R**，不是任意截取最后 1–2 个 token，也不进行模糊编辑距离匹配。R 为空时不做局部追加；R 出现多次时归为 `ambiguous_tail`，回退到原 R。末尾没有剩余 token 的匹配也计入出现次数，不能为获得追加而挑选另一处位置。唯一匹配可能是偶然重复，并不保证声学时间对齐正确；后续仍由 target 验证。

候选生成函数只接收 C、P、R 和预先固定的策略。当前 teacher 的 `generated_token_ids` 仅在两种候选确定后用于评分；不用于选择匹配位置、长度或候选。

## 看哪些字段

主要看 `policies.<策略>.after_first`，并核对 `by_trigger.second` 和 `by_trigger.later`。分母为该分组内全部事件，包含无法对齐的事件。

| 字段 | 含义 |
| --- | --- |
| `alignment_success_rate` | 对齐成功比例；包括对齐成功但没有后缀的事件 |
| `alignment_success_rate_eligible` | 分母改为第 2 轮起且历史前缀可复用的事件；无此类事件为 null |
| `extension_proposed_rate` | 实际构造出非空追加后缀的比例，尚未按预算截断 |
| `extension_within_budget_rate_at_8` | 预算 8 内确实容纳至少一个追加 token 的比例 |
| `mean_delta_accepted_at_8` | 相对旧回滚草稿，每事件新增的连续接受量 |
| `incremental_gain_rate_at_8` | 相对旧草稿获得正增量的事件比例 |
| `recordings_with_incremental_gain_at_8` | 有正增量事件的录音数，避免只看集中在单条录音上的收益 |
| `mean_recording_delta_accepted_at_8` | 先按每条录音平均，再对录音等权平均的增量 |
| `reason_counts` | 未追加原因及数量，见下表 |

`comparison.after_first.mean_local_minus_full_accepted_at_8` 是局部策略相对完整前缀策略的配对差值；两种策略各自的 `mean_delta_accepted_at_8` 则都是相对旧回滚草稿。局部策略遇到重复锚点会回退，因此也可能弱于完整前缀策略，`local_worse_events` 会保留这种结果。

### 旧回滚草稿全部被接受时

`when_old_fully_accepted` 专门筛选：第 2 轮起、历史可复用、R 非空，且**整个 R**与当前 teacher 生成序列的开头一致。

- `events`：符合上述条件的事件数；不要求成功追加。
- `extension_proposed_events`：其中有追加后缀的事件数。
- `budget_room_events_at_8`：其中旧草稿长度小于 8 的事件数。
- `mean_appended_accepted_tokens`：这些事件中平均接受的追加 token 数，不限制预算。
- `mean_appended_accepted_at_8`：这些事件中预算 8 内平均接受的追加 token 数。
- `mean_appended_accepted_given_extension_at_8`：进一步只在确实提出追加的事件中计算预算内平均接受量。

条件子集为空时均值为 null，不能把它当作 0 接受率。如果旧 R 中途被拒绝，后面的追加不能贡献本轮收益，即使后缀内容恰好正确。

### 对齐原因

- `first_no_draft`：第 1 轮按协议禁用。
- `old_prefix_not_reusable`：历史输出无法与当前稳定前缀建立规定的复用关系。
- `prefix_mismatch`：完整前缀策略未匹配 P+R。
- `empty_tail`：局部策略没有非空锚点。
- `tail_not_found`：C 中没有连续匹配 R。
- `ambiguous_tail`：C 中存在多个匹配位置。
- `aligned_no_extension`：匹配成功，但 C 到此结束。
- `extended`：产生非空追加；是否被接受需看评分字段。

逐事件 JSONL 包含 C、P、R、两个候选、匹配位置（`match_end` 为 C 中从 0 开始的右开边界）、匹配数量、原草稿接受量、追加长度和接受量，可直接检查错误样例。

## 判断实验结果

1. **局部对齐和追加都有提升，增量接受在多条 dev 录音上出现：** 值得进一步测在线延迟；继续优化 CTC 有了比整体 TER 更直接的依据。
2. **追加很多，但旧草稿完整接受且有预算余量时，新增接受仍接近 0：** 更支持“末尾候选不准确/局部匹配错位”的判断。结合逐事件样例区分后，再决定转向条件化 drafter。
3. **局部对齐仍很少：** 说明这个完整尾部精确锚点也难匹配，尚不能单靠该结果判断 CTC 的所有后缀都无用。
4. **旧草稿经常未完整接受：** 当前只追加策略的收益机会本来就有限，需要把旧草稿修正与后缀扩展分开讨论。

该报告是离线、单次验证、greedy 内容 token 指标，不包含 bonus/EOS，不模拟后续验证轮次，不修改现有训练选模或线上推理。不得把“对齐成功”当作允许直接提交文本；真正上线仍必须经过 target 验证。先固定 dev 上的策略，再用独立数据确认稳定性，不能据此单独宣称端到端加速。
