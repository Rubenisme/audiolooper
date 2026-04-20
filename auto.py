#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "librosa",
#   "soundfile",
# ]
# ///
"""Auto-detect loop parameters.

Pipeline:
  1. Load audio.
  2. Body detection via smoothed RMS envelope (trim low-energy intro/outro).
  3. BPM + beat tracking via librosa.
  4. Loop point search: chroma cosine similarity at beats, constrained to body.
  5. Fade durations snapped to musical bars (4/4 assumed).

Prints the suggested parameters. Pass --render to also render the loop.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate, find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from loop import load_audio, render_loop


HOP = 512


def _checkerboard_kernel(k: int) -> np.ndarray:
    size = 2 * k
    ker = np.zeros((size, size), dtype=np.float32)
    ker[:k, :k] = 1.0
    ker[k:, k:] = 1.0
    ker[:k, k:] = -1.0
    ker[k:, :k] = -1.0
    x, y = np.meshgrid(
        np.arange(size) - k + 0.5,
        np.arange(size) - k + 0.5,
    )
    sigma = k / 2.0
    gauss = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return (ker * gauss).astype(np.float32)


def foote_novelty(audio_mono: np.ndarray, sr: int, kernel_s: float = 1.0) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio_mono, sr=sr, hop_length=HOP, n_mfcc=13)
    norms = np.linalg.norm(mfcc, axis=0, keepdims=True)
    mfcc_norm = mfcc / np.maximum(norms, 1e-10)
    ssm = mfcc_norm.T @ mfcc_norm  # [n, n]

    k = max(4, int(kernel_s * sr / HOP))
    kernel = _checkerboard_kernel(k)
    n = ssm.shape[0]
    nov = np.zeros(n, dtype=np.float32)
    for t in range(k, n - k):
        nov[t] = float(np.sum(ssm[t - k:t + k, t - k:t + k] * kernel))
    nov -= nov.min()
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def novelty_peaks(nov: np.ndarray, sr: int, min_distance_s: float = 2.0, top_n: int = 15) -> np.ndarray:
    min_dist = int(min_distance_s * sr / HOP)
    peaks, props = find_peaks(nov, distance=min_dist, prominence=0.05)
    if len(peaks) == 0:
        return peaks
    prom = props["prominences"]
    order = np.argsort(prom)[::-1][:top_n]
    return np.sort(peaks[order])


def detect_body(
    audio_mono: np.ndarray,
    sr: int,
    smooth_s: float = 3.0,
    threshold_pct: float = 0.30,
) -> tuple[float, float]:
    rms = librosa.feature.rms(y=audio_mono, frame_length=2048, hop_length=HOP)[0]
    smooth_n = max(1, int(smooth_s * sr / HOP))
    kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
    rms_smooth = np.convolve(rms, kernel, mode="same")

    threshold = np.max(rms_smooth) * threshold_pct
    above = rms_smooth > threshold
    if not above.any():
        return 0.0, len(audio_mono) / sr

    first = int(np.argmax(above))
    last = len(above) - 1 - int(np.argmax(above[::-1]))
    return first * HOP / sr, last * HOP / sr


def detect_downbeats(beat_frames: np.ndarray, peak_frames: np.ndarray, meter: int = 4) -> np.ndarray:
    """Group beats into bars and pick the phase whose downbeats align closest to structural peaks."""
    if len(beat_frames) < meter:
        return beat_frames
    best_phase, best_score = 0, float("inf")
    for phase in range(meter):
        candidates = beat_frames[phase::meter]
        score = float(sum(int(np.min(np.abs(candidates - p))) for p in peak_frames)) if len(peak_frames) > 0 else 0.0
        if score < best_score:
            best_score, best_phase = score, phase
    return beat_frames[best_phase::meter]


def snap_fade_to_downbeat(
    anchor_s: float,
    target_s: float,
    downbeat_times: np.ndarray,
    min_s: float,
    max_s: float,
) -> float:
    """Return fade duration so that anchor_s - duration lands on the nearest downbeat."""
    start_target = anchor_s - target_s
    valid = downbeat_times[(downbeat_times >= anchor_s - max_s) & (downbeat_times <= anchor_s - min_s)]
    if len(valid) == 0:
        return target_s
    return anchor_s - float(valid[np.argmin(np.abs(valid - start_target))])


def find_loop_in_body(
    audio_mono: np.ndarray,
    sr: int,
    body_start_s: float,
    body_end_s: float,
    peak_frames: np.ndarray,
    min_loop_s: float = 20.0,
    context_frames: int = 40,
    peak_tolerance_s: float = 1.0,
    peak_bonus: float = 0.15,
) -> tuple[dict | None, float, np.ndarray, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr, hop_length=HOP)
    bpm = float(tempo if np.isscalar(tempo) else tempo[0])

    harmonic, _ = librosa.effects.hpss(audio_mono)
    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr, hop_length=HOP)
    chroma_max = chroma.shape[1]

    # candidate positions: union of beats and novelty peaks
    cand_frames = np.union1d(beat_frames, peak_frames).astype(int)
    cand_times = cand_frames * HOP / sr
    mask = (cand_times >= body_start_s) & (cand_times <= body_end_s)
    cand_frames = cand_frames[mask]
    cand_times = cand_times[mask]

    min_loop_frames = int(min_loop_s * sr / HOP)
    tol = int(peak_tolerance_s * sr / HOP)

    def near_peak(f: int) -> float:
        if len(peak_frames) == 0:
            return 0.0
        d = int(np.min(np.abs(peak_frames - f)))
        if d > tol:
            return 0.0
        return 1.0 - d / tol

    all_scored = []
    for i, a in enumerate(cand_frames):
        for j, b in enumerate(cand_frames):
            if b - a < min_loop_frames:
                continue

            va, vb = chroma[:, a], chroma[:, b]
            d = np.linalg.norm(va) * np.linalg.norm(vb)
            if d < 1e-10:
                continue
            point_sim = float(np.dot(va, vb) / d)

            la = min(context_frames, chroma_max - a)
            lb = min(context_frames, chroma_max - b)
            L = min(la, lb)
            if L < 4:
                continue
            ca = chroma[:, a:a + L]
            cb = chroma[:, b:b + L]
            dot = np.einsum("ij,ij->j", ca, cb)
            na = np.linalg.norm(ca, axis=0)
            nb = np.linalg.norm(cb, axis=0)
            seq_sim = float(np.mean(dot / np.maximum(na * nb, 1e-10)))

            sim = 0.4 * point_sim + 0.6 * seq_sim
            duration_s = (b - a) * HOP / sr
            bonus = peak_bonus * (near_peak(a) + near_peak(b))
            # extra bonus if start is near the FIRST novelty peak (body-start structural cue)
            first_peak_bonus = 0.0
            if len(peak_frames) > 0 and abs(a - int(peak_frames[0])) <= tol:
                first_peak_bonus = 0.02
            score = sim + bonus + first_peak_bonus + 0.0015 * duration_s

            all_scored.append({
                "start_s": float(cand_times[i]),
                "end_s": float(cand_times[j]),
                "score": score,
                "similarity": sim,
                "peak_bonus": bonus,
                "duration_s": duration_s,
            })

    all_scored.sort(key=lambda x: x["score"], reverse=True)
    return all_scored, bpm, cand_frames, beat_frames


def align_loop_endpoints(
    audio_mono: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    search_ms: float = 50.0,
) -> tuple[float, float]:
    """Shift end_s by up to ±search_ms to minimise waveform discontinuity at the loop boundary.

    Cross-correlates a window centred on loop_start with a search region around loop_end.
    The lag that maximises correlation is where the waveform phase at loop_end best matches
    the phase at loop_start, reducing the audible discontinuity inside the crossfade.
    """
    search_n = int(search_ms / 1000.0 * sr)
    win_n = search_n * 2

    s = int(round(start_s * sr))
    e = int(round(end_s * sr))

    if s < win_n or e - search_n - win_n < 0 or e + search_n + win_n > len(audio_mono):
        return start_s, end_s

    ref = audio_mono[s - win_n : s + win_n].astype(np.float64)
    region = audio_mono[e - search_n - win_n : e + search_n + win_n].astype(np.float64)

    xcorr = correlate(region, ref, mode="valid")
    best_lag = int(np.argmax(xcorr)) - search_n

    return start_s, end_s + best_lag / sr


def compute_adaptive_gain_db(
    audio_mono: np.ndarray,
    sr: int,
    loop_start_s: float,
    loop_end_s: float,
    fade_out_s: float,
    fade_in_s: float,
) -> float:
    """Return gain_db that makes the incoming pre-head RMS match the outgoing tail RMS."""
    loop_start = int(round(loop_start_s * sr))
    loop_end = int(round(loop_end_s * sr))
    fade_out_n = int(round(fade_out_s * sr))
    fade_in_n = int(round(fade_in_s * sr))

    outgoing = audio_mono[loop_end - fade_out_n : loop_end]
    incoming = audio_mono[loop_start - fade_in_n : loop_start]

    rms_out = float(np.sqrt(np.mean(outgoing ** 2)))
    rms_in = float(np.sqrt(np.mean(incoming ** 2)))

    if rms_in < 1e-10:
        return 0.0

    gain_db = 20.0 * float(np.log10(rms_out / rms_in))
    return float(np.clip(gain_db, -12.0, 12.0))


def snap_to_bars(target_s: float, bar_s: float, choices: list[int]) -> tuple[int, float]:
    cands = [(n, n * bar_s) for n in choices]
    return min(cands, key=lambda c: abs(c[1] - target_s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--min-loop-s", type=float, default=20.0)
    p.add_argument("--gain-db", type=float, default=1.5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--render", action="store_true")
    p.add_argument("--sr", type=int, default=44100)
    args = p.parse_args()

    print(f"loading: {args.input.name}")
    audio = load_audio(args.input, sr=args.sr)
    mono = audio.mean(axis=1)
    dur = len(mono) / args.sr
    print(f"  duration: {dur:.2f}s\n")

    print("[1] Foote novelty (MFCC, kernel=1.0s)")
    nov = foote_novelty(mono, args.sr, kernel_s=1.0)
    peaks = novelty_peaks(nov, args.sr, min_distance_s=2.0, top_n=20)
    peak_times = peaks * HOP / args.sr
    print(f"  {len(peaks)} peaks: {', '.join(f'{t:.2f}' for t in peak_times)}\n")

    print("[2] body detection")
    if len(peak_times) >= 2:
        body_start = float(peak_times[0])
        body_end = float(peak_times[-1])
        print(f"  body (first->last novelty peak): {body_start:.2f}s -> {body_end:.2f}s")
        print(f"  (RMS fallback would give: {', '.join(f'{v:.2f}' for v in detect_body(mono, args.sr))})")
    else:
        body_start, body_end = detect_body(mono, args.sr)
        print(f"  body (RMS fallback): {body_start:.2f}s -> {body_end:.2f}s")
    print()

    print("[3] BPM + loop search (beats + novelty peaks as candidates, HPSS chroma)")
    all_pairs, bpm, cand_frames, beat_frames = find_loop_in_body(
        mono, args.sr, body_start, body_end, peaks, min_loop_s=args.min_loop_s,
    )
    print(f"  bpm: {bpm:.1f}  ({len(cand_frames)} candidate positions in body)")
    if not all_pairs:
        print("  no loop candidates found")
        return
    print("  top 5 pairs:")
    print(f"    {'start':>7} {'end':>7} {'dur':>6} {'sim':>6} {'peak':>6} {'score':>7}")
    for p in all_pairs[:5]:
        print(
            f"    {p['start_s']:>7.2f} {p['end_s']:>7.2f} "
            f"{p['duration_s']:>6.2f} {p['similarity']:>6.3f} "
            f"{p['peak_bonus']:>6.3f} {p['score']:>7.4f}"
        )
    best = all_pairs[0]
    print()

    print("[4] downbeat-snapped fade durations")
    bar_s = 4 * 60 / bpm
    body_dur = best["end_s"] - best["start_s"]
    target_out_s = min(body_dur * 0.20, 16.0)
    target_in_s = min(body_dur * 0.10, 8.0)

    downbeat_frames = detect_downbeats(beat_frames, peaks)
    downbeat_times = downbeat_frames * HOP / args.sr
    phase_offset_s = float(downbeat_times[0]) if len(downbeat_times) > 0 else 0.0
    print(f"  bar length: {bar_s:.3f}s (at {bpm:.1f} bpm, 4/4)  downbeat phase: {phase_offset_s:.2f}s")

    fade_out_s = snap_fade_to_downbeat(best["end_s"], target_out_s, downbeat_times, bar_s * 3, bar_s * 18)
    fade_in_s = snap_fade_to_downbeat(best["start_s"], target_in_s, downbeat_times, bar_s * 1, bar_s * 10)
    fade_in_s = min(fade_in_s, fade_out_s)
    print(f"  fade_out: {fade_out_s:.2f}s  (target was {target_out_s:.2f}s, start snapped to downbeat)")
    print(f"  fade_in:  {fade_in_s:.2f}s  (target was {target_in_s:.2f}s, start snapped to downbeat)\n")

    print("[5] phase alignment")
    original_end_s = best["end_s"]
    best["start_s"], best["end_s"] = align_loop_endpoints(mono, args.sr, best["start_s"], best["end_s"])
    shift_ms = (best["end_s"] - original_end_s) * 1000
    print(f"  end shifted by {shift_ms:+.1f}ms  ({original_end_s:.3f}s -> {best['end_s']:.3f}s)\n")

    print("[6] adaptive loudness")
    gain_db = compute_adaptive_gain_db(mono, args.sr, best["start_s"], best["end_s"], fade_out_s, fade_in_s)
    print(f"  incoming pre-head will be {gain_db:+.2f}dB relative to outgoing tail\n")

    print("suggested loop.py command:")
    print(
        f'  uv run loop.py "{args.input.name}" \\\n'
        f'    --start {best["start_s"]:.2f} --end {best["end_s"]:.2f} \\\n'
        f'    --fade-out-ms {fade_out_s*1000:.0f} --fade-in-ms {fade_in_s*1000:.0f} \\\n'
        f'    --gain-db {gain_db:.2f} --repeats {args.repeats}'
    )

    if args.render:
        print("\nrendering...")
        out = render_loop(
            audio, args.sr,
            best["start_s"], best["end_s"],
            fade_out_s * 1000, fade_in_s * 1000,
            gain_db, args.repeats,
            include_intro=True,
        )
        out_path = args.input.parent / "Loops" / f"{args.input.stem}_auto.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, args.sr, subtype="PCM_16")
        print(f"  saved: {out_path}  ({len(out)/args.sr:.2f}s)")


if __name__ == "__main__":
    main()
