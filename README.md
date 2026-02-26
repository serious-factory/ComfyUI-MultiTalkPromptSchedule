# ComfyUI-MultiTalkPromptSchedule

Frame-based prompt scheduling for [InfiniteTalk/MultiTalk](https://github.com/kijai/ComfyUI-WanVideoWrapper) in ComfyUI.

Control **what happens when** in your multi-speaker videos by assigning prompts to specific frame ranges — enabling scene direction (who speaks, who listens, gaze direction) synchronized with audio.

## The problem

InfiniteTalk generates video in sliding windows with a single prompt for the entire duration. The native `|` separator maps one prompt per window, but you can't control timing precisely.

## The solution

**MultiTalk Prompt Schedule** lets you define prompts with exact frame durations:

```
120: Person 1 speaks to the camera, person 2 listens attentively
120: Person 2 responds with a smile, person 1 nods
120: Both look at the camera and smile
```

Each line = `<frames>: <prompt>`. At 24fps, 120 frames = 5 seconds.

The node handles encoding, and the prompt selection follows audio position — so prompts stay in sync even when window sizes don't align perfectly with your frame ranges.

## Features

- **Frame-precise scheduling** — prompts are selected based on audio position, not window iteration count
- **Per-scene negative prompts** — use `|||` to override the negative prompt for specific scenes:
  ```
  120: Person 1 speaks ||| blurred eyes, static pose
  120: Both smile
  ```
- **Comments** — lines starting with `#` are ignored
- **NAG compatible** — works with `WanVideoApplyNAG` (prompt_schedule is preserved through NAG)
- **Zero-patch install** — monkey-patches WanVideoWrapper at runtime, no files modified

## Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/serious-factory/ComfyUI-MultiTalkPromptSchedule
```

**Requires** [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (installed first).

## Workflow

```
                                    ┌─────────────────────────┐
T5 Encoder ──────────────────────>  │  MultiTalk Prompt       │
                                    │  Schedule               │
                                    │                         │
                                    │  120: P1 speaks...      │──> text_embeds ──> WanVideoApplyNAG ──> WanVideoSampler
                                    │  120: P2 responds...    │
                                    │  120: Both smile...     │──> negative_text_embeds
                                    │                         │
                                    │  negative_prompt: ...   │
                                    └─────────────────────────┘
```

Replaces `WanVideoTextEncodeSingle` in your InfiniteTalk workflow. Everything else stays the same.

## How it works

1. **At ComfyUI startup**: the node monkey-patches `multitalk_loop.py` to add frame-based prompt selection logic (source-level patch, skipped if already present)
2. **At generation time**: the node encodes all prompts with T5, builds a frame schedule `[(0,120), (120,240), (240,360)]`, and passes it in `text_embeds`
3. **During sampling**: each sliding window checks `audio_start_idx` against the schedule to pick the right prompt

If ComfyUI-WanVideoWrapper updates and changes the prompt selection code, the patch will log a warning and fall back gracefully — your other workflows won't break.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `t5` | T5 text encoder (from `WanVideo T5 Text Encoder Loader`) |
| `schedule` | Prompt schedule text (see format above) |
| `negative_prompt` | Default negative prompt for scenes without `\|\|\|` |
| `force_offload` | Offload T5 after encoding (default: true) |
| `device` | Encoding device: gpu or cpu |

## Known limitations

- InfiniteTalk follows reference image + audio more strongly than text prompts. Subtle prompt changes (e.g. gaze direction) may have limited visual impact. Stronger changes (scene description, actions) work better.
- `nag_scale` can be increased (15-20) to push prompt adherence, at the cost of visual quality.

## License

MIT
