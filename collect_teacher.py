"""Collect exact vLLM decoder token suffixes at each Qwen3-ASR streaming step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_pilot import load_manifest
from stream_draft import full_chunk_ends, read_audio
from streaming_ctc import suffix_after_prefix


class _GenerateCapture:
    def __init__(self, model):
        self.model = model
        self.last = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, inputs, **kwargs):
        outputs = self.model.generate(inputs, **kwargs)
        if len(inputs) != 1 or len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("expected one streaming vLLM request and one output")
        self.last = (inputs[0]["prompt"], list(outputs[0].outputs[0].token_ids),
                     getattr(outputs[0], "prompt_token_ids", None))
        return outputs


def capture_stream_events(asr, audio, chunk_samples: int, audio_path: str,
                          language: str, unfixed_chunk_num: int,
                          unfixed_token_num: int) -> tuple[list[dict], list[dict]]:
    """Run the official streaming method while observing its actual generate call."""
    if unfixed_chunk_num != 2:
        raise ValueError("rollback pilot requires unfixed_chunk_num=2: full reuse at second chunk")
    ends = full_chunk_ends(len(audio), chunk_samples)
    if not ends:
        raise ValueError(f"{audio_path}: recording has no full streaming chunk")
    state = asr.init_streaming_state(
        language=language, chunk_size_sec=chunk_samples / 16000,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
    )
    original_model = asr.model
    capture = _GenerateCapture(original_model)
    asr.model = capture
    events, teachers = [], []
    previous_ids = []
    try:
        previous = 0
        for index, end in enumerate(ends, 1):
            capture.last = None
            state = asr.streaming_transcribe(audio[previous:end], state)
            previous = end
            if capture.last is None:
                raise RuntimeError(f"{audio_path}: no decoder output at {end} samples")
            prompt, generated_ids, actual_prompt_ids = capture.last
            if not prompt.startswith(state.prompt_raw):
                raise RuntimeError("streaming prompt no longer starts with prompt_raw")
            if not all(type(token) is int and token >= 0 for token in generated_ids):
                raise RuntimeError("vLLM returned invalid generated token IDs")
            special_ids = set(getattr(asr.processor.tokenizer, "all_special_ids", []))
            while generated_ids and generated_ids[-1] in special_ids:
                generated_ids.pop()
            tokenizer = asr.processor.tokenizer
            # Derive the prefix in its prompt context, not by re-encoding isolated text.
            base_ids = tokenizer.encode(state.prompt_raw, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if prompt_ids[:len(base_ids)] != base_ids:
                raise RuntimeError("prompt/content token boundary changed; cannot safely split prefix")
            prefix_ids = prompt_ids[len(base_ids):]
            if actual_prompt_ids is not None and prefix_ids:
                if list(actual_prompt_ids[-len(prefix_ids):]) != prefix_ids:
                    raise RuntimeError("actual vLLM prompt suffix differs from content prefix IDs")
            if index <= 2 and prefix_ids:
                raise RuntimeError("first two chunks must have empty stable prefix")
            old_ids = suffix_after_prefix(previous_ids, prefix_ids)
            target_ids = prefix_ids + generated_ids
            events.append({"audio": audio_path, "end_sample": end,
                           "text": state.text, "target_token_ids": target_ids})
            teachers.append({"audio": audio_path, "end_sample": end,
                             "schema_version": 2, "chunk_index": index,
                             "chunk_samples": chunk_samples,
                             "unfixed_chunk_num": unfixed_chunk_num,
                             "unfixed_token_num": unfixed_token_num,
                             "language": language,
                             "previous_token_ids": previous_ids,
                             "old_draft_token_ids": old_ids or [],
                             "old_prefix_match": old_ids is not None,
                             "target_token_ids": target_ids,
                             "prefix_token_ids": prefix_ids,
                             "generated_token_ids": generated_ids})
            previous_ids = target_ids
    finally:
        asr.model = original_model
    return events, teachers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3-ASR checkpoint")
    parser.add_argument("--manifest", required=True,
                        help="Whole-recording JSONL with audio and text")
    parser.add_argument("--events-out", required=True, help="New event manifest JSONL")
    parser.add_argument("--teacher-out", required=True, help="New token trace JSONL")
    parser.add_argument("--language", required=True,
                        help="Force text-only ASR output, e.g. Chinese")
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, choices=(2,), default=2,
                        help="This pilot uses no stable prefix in the first two chunks")
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if (args.chunk_seconds <= 0 or args.unfixed_chunk_num < 0
            or args.unfixed_token_num < 0 or args.tensor_parallel_size < 1):
        raise ValueError("chunk/tensor parallel sizes must be positive; rollback counts nonnegative")
    events_path = Path(args.events_out).resolve()
    teacher_path = Path(args.teacher_out).resolve()
    if events_path == teacher_path or events_path.exists() or teacher_path.exists():
        raise FileExistsError("event and teacher outputs must be separate new files")
    records = load_manifest(args.manifest, min_records=1)
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.LLM(model=args.model,
                            tensor_parallel_size=args.tensor_parallel_size)
    chunk_samples = int(round(args.chunk_seconds * 16000))
    events_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("x", encoding="utf-8") as events_file, \
            teacher_path.open("x", encoding="utf-8") as teacher_file:
        for record in records:
            audio = read_audio(record["audio"])
            events, teachers = capture_stream_events(
                asr, audio, chunk_samples, record["audio"], args.language,
                args.unfixed_chunk_num, args.unfixed_token_num)
            for event, teacher in zip(events, teachers):
                events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                teacher_file.write(json.dumps(teacher, ensure_ascii=False) + "\n")
            events_file.flush()
            teacher_file.flush()
            print(f"captured {len(events)} chunks: {record['audio']}", flush=True)


if __name__ == "__main__":
    main()
