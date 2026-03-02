# ComfyUI-MultiTalkPromptSchedule

Frame-based prompt scheduling for [InfiniteTalk/MultiTalk](https://github.com/kijai/ComfyUI-WanVideoWrapper) in ComfyUI.

Control **what happens when** in your multi-speaker videos by assigning prompts to specific frame ranges — enabling scene direction (who speaks, who listens, gaze direction) synchronized with audio.

## The problem

InfiniteTalk generates video in sliding windows with a single prompt for the entire duration. The native `|` separator maps one prompt per window, but you can't control timing precisely.

## The solution

**MultiTalk Prompt Schedule** lets you define prompts with exact frame durations:

```
120: Person 1 speaks to the camera, person 2 listens
120: Person 2 responds with a smile, person 1 nods
120: Both look at the camera and smile
```

Each line = `<frames>: <prompt>`. At 24fps, 120 frames = 5 seconds.

The node handles encoding, and the prompt selection follows audio position — so prompts stay in sync even when window sizes don't align perfectly with your frame ranges.

## Schedule format

### Basic — positive prompt only

Each scene gets the global `negative_prompt` field as its NAG negative:

```
120: Person 1 speaks to the camera, person 2 listens
120: Person 2 responds, person 1 nods
120: Both look at the camera and smile
```

### Per-scene NAG negatives

Use `|||` to set a custom NAG negative for a specific scene. This is the **recommended approach for gaze control** — NAG pushes against the wrong gaze direction:

```
120: Person 1 speaks to person 2 ||| looking at camera, staring at viewer, wandering eyes
120: Person 2 speaks to camera ||| looking sideways, looking at other person, turned away
120: Both smile at camera ||| looking sideways, turned away
```

Per-scene negatives feed **NAG (Normalized Attention Guidance)**, which operates inside cross-attention layers. This is the only negative guidance active at `cfg=1` (standard InfiniteTalk setting). Regular `negative_prompt_embeds` are ignored at `cfg=1`.

### Mixed — some scenes with custom negatives, others use global

Scenes **without** `|||` fall back to the global `negative_prompt` field:

```
120: Person 1 speaks to person 2 ||| looking at camera, wandering eyes
120: Person 2 responds
120: Both smile ||| looking sideways, turned away
```

Here scene 2 uses the global negative prompt, scenes 1 and 3 use their own.

### Comments

Lines starting with `#` are ignored:

```
# Scene 1: introduction (5s)
120: Person 1 speaks to the camera
# Scene 2: response (5s)
120: Person 2 responds
```

## Features

- **Frame-precise scheduling** — prompts are selected based on audio position, not window iteration count
- **Per-scene NAG negatives** — use `|||` to override the NAG negative for specific scenes (gaze control)
- **Global negative fallback** — scenes without `|||` use the node's `negative_prompt` field
- **Comments** — lines starting with `#` are ignored
- **NAG compatible** — per-scene negatives swap `nag_prompt_embeds` in the sampling loop
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
                                    │  120: P1 speaks...      │──> text_embeds ──────> WanVideoApplyNAG ──> WanVideoSampler
                                    │  120: P2 responds...    │
                                    │  120: Both smile...     │──> negative_text_embeds ──> WanVideoApplyNAG (nag_text_embeds)
                                    │                         │
                                    │  negative_prompt: ...   │
                                    └─────────────────────────┘
```

Connect **both outputs** to the NAG node:
- `text_embeds` (output 0) → `original_text_embeds`
- `negative_text_embeds` (output 1) → `nag_text_embeds`

No separate `WanVideoTextEncodeSingle` needed for the negative prompt.

## How it works

1. **At ComfyUI startup**: the node monkey-patches `multitalk_loop.py` to add frame-based prompt selection logic (source-level patch, skipped if already present)
2. **At generation time**: the node encodes all prompts + all unique negatives with T5, builds a frame schedule `[(0,120), (120,240), (240,360)]`, and passes everything in `text_embeds`
3. **During sampling**: each sliding window checks `audio_start_idx` against the schedule to pick the right prompt **and** swap `nag_prompt_embeds` for per-scene NAG negatives

If ComfyUI-WanVideoWrapper updates and changes the prompt selection code, the patch will log a warning and fall back gracefully — your other workflows won't break.

## How NAG per-scene works

At `cfg=1` (standard InfiniteTalk), the only negative guidance comes from **NAG** (Normalized Attention Guidance). NAG operates inside cross-attention layers, running each layer twice (positive + negative context) and combining the results.

When a scene has a `|||` negative:
1. The node encodes it with T5 and stores it in `text_embeds["all_nag_embeds"]`
2. The monkey-patch in `multitalk_loop` swaps `text_embeds["nag_prompt_embeds"]` for that scene
3. The sampler reads the swapped embedding and passes it to the transformer's cross-attention

This means you can push against specific artifacts per-scene. For gaze control:
- Scene where characters look at each other: `||| looking at camera, staring at viewer`
- Scene where characters look at camera: `||| looking sideways, turned away`

## Parameters

| Parameter | Description |
|-----------|-------------|
| `t5` | T5 text encoder (from `WanVideo T5 Text Encoder Loader`) |
| `schedule` | Prompt schedule text (see format above) |
| `negative_prompt` | Default NAG negative for scenes without `\|\|\|` |
| `force_offload` | Offload T5 after encoding (default: true) |
| `device` | Encoding device: gpu or cpu |

## Known limitations

- InfiniteTalk follows reference image + audio more strongly than text prompts. Subtle prompt changes (e.g. gaze direction) may have limited visual impact. NAG negatives help push against unwanted directions.
- `nag_scale` can be increased (15-20) to push prompt adherence, at the cost of visual quality.
- At `cfg=1`, the `negative_prompt_embeds` output is technically unused by the sampler. The negative guidance comes exclusively from NAG via `nag_prompt_embeds`.

## License

MIT
