#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "soundfile",
# ]
# ///
"""Render a crossfaded loop from an audio file.

Structure per output:
    intro + body + (crossfade + body) * (repeats - 1) + final_tail

- intro     = audio[0:loop_start]                         (pre-loop content, played once)
- body      = audio[loop_start:loop_end - fade_n]         (loop content excluding the tail region)
- crossfade = outgoing tail + incoming pre-head
    * outgoing = audio[loop_end - fade_n:loop_end], equal-power fade-out  (100% -> 0%)
    * incoming = audio[loop_start - fade_n:loop_start], equal-power fade-in x gain  (0% -> 100%)
- final_tail = outgoing tail with fade-out, no incoming (clean ending)

Pre-head means the crossfade ends exactly at loop_start, so any transient there
(a sound effect, a downbeat) hits at full volume rather than being buried under a fade.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_time(t: str) -> float:
    if ":" in t:
        parts = t.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(t)


def load_audio(path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def equal_power_curves(n: int):
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    return np.cos(t), np.sin(t)


def render_loop(
    audio: np.ndarray,
    sr: int,
    loop_start_s: float,
    loop_end_s: float,
    fade_out_ms: float,
    fade_in_ms: float,
    gain_db: float,
    repeats: int,
    include_intro: bool,
    fade_out_ending: bool = False,
) -> np.ndarray:
    loop_start = int(round(loop_start_s * sr))
    loop_end = int(round(loop_end_s * sr))
    fade_out_n = int(round(fade_out_ms / 1000.0 * sr))
    fade_in_n = int(round(fade_in_ms / 1000.0 * sr))
    fade_out_n = max(1, min(fade_out_n, loop_end - loop_start - 1))
    fade_in_n = max(1, min(fade_in_n, loop_start, fade_out_n))
    gain = 10.0 ** (gain_db / 20.0)

    # Outgoing tail fills the crossfade region; incoming pre-head aligns to the end of it.
    fade_out_curve = np.cos(np.linspace(0.0, np.pi / 2.0, fade_out_n, dtype=np.float32))[:, None]
    fade_in_curve = np.sin(np.linspace(0.0, np.pi / 2.0, fade_in_n, dtype=np.float32))[:, None]

    outgoing = audio[loop_end - fade_out_n:loop_end] * fade_out_curve
    incoming = audio[loop_start - fade_in_n:loop_start] * fade_in_curve * gain

    crossfade = outgoing.copy()
    crossfade[-fade_in_n:] += incoming

    body = audio[loop_start:loop_end - fade_out_n]
    ending = outgoing if fade_out_ending else audio[loop_end - fade_out_n:]

    parts = []
    if include_intro and loop_start > 0:
        parts.append(audio[:loop_start])
    parts.append(body)
    for _ in range(repeats - 1):
        parts.append(crossfade)
        parts.append(body)
    parts.append(ending)

    return np.concatenate(parts, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--start", required=True, help="loop start (seconds or mm:ss[.ms])")
    p.add_argument("--end", required=True, help="loop end (seconds or mm:ss[.ms])")
    p.add_argument("--fade-ms", type=float, default=None, help="symmetric crossfade; sets --fade-in-ms and --fade-out-ms together")
    p.add_argument("--fade-in-ms", type=float, default=None, help="incoming pre-head fade duration")
    p.add_argument("--fade-out-ms", type=float, default=None, help="outgoing tail fade duration")
    p.add_argument("--gain-db", type=float, default=1.5)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-intro", action="store_true")
    p.add_argument("--fade-out-ending", action="store_true", help="end with a fade-out instead of the original outro")
    p.add_argument("--sr", type=int, default=44100)
    args = p.parse_args()

    start_s = parse_time(args.start)
    end_s = parse_time(args.end)
    if end_s <= start_s:
        sys.exit(f"error: --end ({end_s}s) must be greater than --start ({start_s}s)")

    if args.fade_ms is not None:
        fade_out_ms = fade_in_ms = args.fade_ms
    else:
        fade_out_ms = args.fade_out_ms if args.fade_out_ms is not None else 1000.0
        fade_in_ms = args.fade_in_ms if args.fade_in_ms is not None else 1000.0

    print(f"loading: {args.input.name}")
    audio = load_audio(args.input, sr=args.sr)
    dur = len(audio) / args.sr
    print(f"  {dur:.2f}s @ {args.sr} Hz, stereo")
    if end_s > dur:
        sys.exit(f"error: --end ({end_s}s) exceeds track duration ({dur:.2f}s)")

    print(
        f"rendering loop {start_s:.3f}s -> {end_s:.3f}s  "
        f"fade_out={fade_out_ms}ms  fade_in={fade_in_ms}ms  "
        f"gain=+{args.gain_db}dB  repeats={args.repeats}"
    )
    out = render_loop(
        audio, args.sr, start_s, end_s,
        fade_out_ms, fade_in_ms, args.gain_db, args.repeats,
        include_intro=not args.no_intro,
        fade_out_ending=args.fade_out_ending,
    )

    if args.output is None:
        stem = args.input.stem
        out_path = args.input.parent / "Loops" / (
            f"{stem}_loop_{start_s:.1f}-{end_s:.1f}_x{args.repeats}.wav"
        )
    else:
        out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_path), out, args.sr, subtype="PCM_16")
    print(f"saved: {out_path}  ({len(out) / args.sr:.2f}s)")


if __name__ == "__main__":
    main()
