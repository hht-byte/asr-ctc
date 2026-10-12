"""Collect exact vLLM decoder token suffixes at each Qwen3-ASR streaming step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_pilot import load_manifest
from stream_draft import full_chunk_ends, read_audio


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
        self.last = (inputs[0]["prompt"], list(outputs[0].outputs[0].token_ids))
        return outputs


def capture_stream_events(asr, audio, chunk_samples: int, audio_path: str,
                          language: str, unfixed_chunk_num: int,
                          unfixed_token_num: int) -> tuple[list[dict], list[dict]]:
    """Run the official streaming method while observing its actual generate call."""
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
    try:
        previous = 0
        for end in ends:
            capture.last = None
            state = asr.streaming_transcribe(audio[previous:end], state)
            previous = end
            if capture.last is None:
                raise RuntimeError(f"{audio_path}: no decoder output at {end} samples")
            prompt, generated_ids = capture.last
            if not prompt.startswith(state.prompt_raw):
                raise RuntimeError("streaming prompt no longer starts with prompt_raw")
            if not all(type(token) is int and token >= 0 for token in generated_ids):
                raise RuntimeError("vLLM returned invalid generated token IDs")
            special_ids = set(getattr(asr.processor.tokenizer, "all_special_ids", []))
            while generated_ids and generated_ids[-1] in special_ids:
                generated_ids.pop()
            prefix_text = prompt[len(state.prompt_raw):]
            prefix_ids = asr.processor.tokenizer.encode(
                prefix_text, add_special_tokens=False)
            events.append({"audio": audio_path, "end_sample": end,
                           "text": state.text})
            teachers.append({"audio": audio_path, "end_sample": end,
                             "prefix_token_ids": prefix_ids,
                             "generated_token_ids": generated_ids})
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
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
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
