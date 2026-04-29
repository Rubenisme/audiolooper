# audiolooper

Two small Python scripts that turn a song into a smooth long-form loop:

- **`loop.py`** — deterministic renderer. You give it loop points and fade parameters; it writes a WAV with `intro + body + (crossfade + body) × (repeats−1) + outro`.
- **`auto.py`** — analysis front-end. Detects where the body starts/ends, finds candidate loop points, and picks fade lengths that fall on musical bar boundaries. Prints a ready-to-run `loop.py` command.

Both scripts are standalone and designed to work together.

## Quick start

```bash
# fully automatic — analyses the track and prints a suggested command
/path/to/librosa-venv/python auto.py "song.m4a"

# deterministic render with explicit parameters
uv run loop.py "song.m4a" \
    --start 19.16 --end 77.60 \
    --fade-out-ms 10588 --fade-in-ms 7059 \
    --gain-db 1.5 --repeats 4
```

Output WAVs land in `Loops/` next to the input file.

## How `loop.py` works

### Anatomy of a render

```
intro  |  body  | crossfade | body | crossfade | body | … | final_tail
```

- `intro` = `audio[0 : loop_start]` — plays once, untouched.
- `body` = `audio[loop_start : loop_end − fade_out_n]` — the loop content minus the tail region.
- `crossfade` = outgoing tail faded out + incoming pre-head faded in.
- `outro` = by default, `audio[loop_end − fade_out_n :]` played straight through — the tail region continues into the song's original ending. Pass `--fade-out-ending` to fade the tail to silence instead.

### Pre-head crossfade

The incoming material in the crossfade is `audio[loop_start − fade_in_n : loop_start]` — the seconds **before** `loop_start`, not after. This matters: the crossfade ends exactly *at* `loop_start`, so any transient there (a sound effect, a downbeat) lands at full volume instead of being buried under a fade-in. Post-head fades are rarely what you want for music with distinct section boundaries.

### Asymmetric fades

`--fade-out-ms` and `--fade-in-ms` are independent. Longer outgoing lets a dominant element (e.g. a drum pattern) decay naturally before the new material arrives; a shorter incoming keeps the entry punchy. Both curves are equal-power (sin/cos) so perceived loudness stays roughly constant at the midpoint.

### Gain

`--gain-db` boosts the incoming side of the crossfade only. It never affects the intro, the first body, or the final tail — so the initial listen-through is unchanged and the lift only kicks in on the loop-around.

## How `auto.py` works

Six stages, each explainable in one line:

### 1. Foote novelty (MFCC, 1-second checkerboard kernel)

Compute a self-similarity matrix on MFCC frames. Slide a checkerboard kernel along the diagonal — the output spikes wherever the spectral character changes. These peaks are structural boundaries: song sections, sound-effect entries, drops.

### 2. Body detection

The first strong novelty peak is treated as the body start; the last strong one as the body end. Anything before/after is intro/outro and is excluded from loop-point search. A smoothed-RMS fallback exists for material with no clear structural peaks.

### 3. Loop-point search

Candidate positions = union of detected beats (`librosa.beat.beat_track`) and novelty peaks, restricted to the body region.

For every candidate pair `(a, b)` with `b − a ≥ min_loop_s`:

```
score = 0.4 × chroma_cosine_similarity(frame_a, frame_b)
      + 0.6 × chroma_cosine_similarity(sequence[a:a+40], sequence[b:b+40])
      + 0.15 × (near_peak(a) + near_peak(b))
      + 0.02 × (1 if a is at the first novelty peak else 0)
      + 0.0015 × duration_s
```

The extra "first-peak" bonus encodes a human-like prior: *the body begins at the first structural boundary*. The duration weight rewards loops that use most of the body rather than clipping short to chase a marginally better chroma match.

The top pair is the suggestion; the top-5 is printed for inspection.

### 4. Downbeat-snapped fades

BPM and beat positions come from `librosa.beat.beat_track`. Beats are grouped into bars of 4; all four phase offsets are tried and the one whose downbeats land closest to structural novelty peaks is selected. Fade targets (20% of loop duration out, 10% in) are then snapped to the nearest *actual* detected downbeat rather than a multiple of the estimated bar length, so fade start positions land on real bar-1 positions rather than average-BPM estimates.

### 5. Phase alignment

A window centred on `loop_start` is cross-correlated with a ±50 ms search region around `loop_end`. The lag that maximises correlation is the sample offset where the waveform phase at `loop_end` best matches `loop_start`, reducing the audible discontinuity inside the crossfade blend.

### 6. Adaptive loudness

The RMS of the outgoing tail (`audio[loop_end − fade_out_n : loop_end]`) and the incoming pre-head (`audio[loop_start − fade_in_n : loop_start]`) are measured. Their ratio becomes `--gain-db`, so the loop-around neither pops loud nor drops quiet regardless of how the two sections differ in level.

## Why this pipeline vs. PyMusicLooper

PyMusicLooper's default loop-point search ranks candidates by chroma similarity at detected beats. It produces good matches but has no notion of "intro vs. body" — its top candidate for a track with a long intro will often be two mid-body moments that happen to sound alike. It also has no crossfade renderer; its export is either a hard splice or `LOOPSTART`/`LOOPLENGTH` tags.

`auto.py` keeps the chroma-matching core idea but layers a structural-segmentation stage on top, so loop endpoints correspond to *musical boundaries* rather than just *spectral echoes*. `loop.py` handles the rendering side that PyMusicLooper doesn't cover.

## Dependencies

- `loop.py`: `numpy`, `soundfile`, and `ffmpeg` on `$PATH` (for m4a/mp3 decoding). Runs via `uv run` using the inline script metadata.
- `auto.py`: `numpy`, `scipy`, `soundfile`, `librosa`. Because `librosa` pulls in `numba`/`llvmlite` (which currently fails to build on some macOS/Python combinations), the easiest path is to run with the Python interpreter from a venv that already has librosa installed, e.g. PyMusicLooper's venv.

## Output

All renders and previews go to `Loops/` next to the input file. Previews (short slices for A/B comparison) can be extracted with ffmpeg after rendering, e.g.

```bash
ffmpeg -ss 60 -i Loops/v4_auto.wav -t 20 -c copy Loops/preview_v4_auto.wav
```

## Tuning knobs

| flag | script | effect |
| --- | --- | --- |
| `--fade-out-ms` | loop | outgoing tail length — longer lets dominant elements decay |
| `--fade-in-ms` | loop | incoming pre-head length — typically shorter than fade-out |
| `--fade-ms` | loop | shorthand that sets both to the same value |
| `--gain-db` | loop | boost on the incoming side of the crossfade |
| `--repeats` | loop | number of body repetitions |
| `--fade-out-ending` | loop | fade the tail to silence instead of playing the original outro |
| `--min-loop-s` | auto | minimum acceptable loop length |
| `--render` | auto | after analysis, render the suggested loop directly |
