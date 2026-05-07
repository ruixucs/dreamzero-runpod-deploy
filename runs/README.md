# Runs — Evidence From a Successful Inference Session

These artifacts come from a real inference run on RunPod with NVIDIA H200 (143 GB) on 2026-05-07. Committing them here so future readers can verify the deployment recipe actually works and compare their own first-run output to ours.

## Files

| File | What It Is | Size |
|---|---|---|
| `server.log.tail` | Last 500 non-spam lines of `logs/server-*.log` (filtered to remove the repeating `Transformer Engine is not available` warnings). Captures multiple complete client connections, the warm-up sequence, and steady-state inference timings. | ~40 KB |
| `sample_outputs/000019_05_07_00_59_30_n8.mp4` | Generated future-frame predictions saved by the server when a client called the `reset` endpoint. `n8` means 8 prediction chunks (8×24=192 frames) were produced before reset. | ~630 KB |
| `sample_outputs/000020_05_07_01_15_54_n5.mp4` | Same format, n=5 chunks. Two samples to compare with. | ~550 KB |

## What to Look For in `server.log.tail`

A normal "client connected → 5 inferences → reset → connection closed" sequence looks like this (excerpts):

```
INFO:root:Connection from ('127.0.0.1', 48012) opened
INFO:websockets.server:connection open

# First infer (1-frame initial observation)
last_state (1, 1, 7) unnormalized_action[action_key] (1, 24, 7)
Inference Time: Total 124.150 seconds, ...

# Subsequent infers (4-frame chunks)
Inference Time: Total 3.512 seconds, ...
Inference Time: Total 2.767 seconds, Transform: 0.009 s, Model: 2.757 s, Untransform: 0.001 s

# Reset (saves the MP4)
INFO:__main__:Saved video on reset to: ./checkpoints/.../DreamZero-DROID/000020_..._n5.mp4
INFO:root:Connection from ('127.0.0.1', 48012) closed
```

Three things to verify in your own logs:

1. **First inference is slow (~120 s).** This is torch.compile + first VAE pass; do not panic.
2. **Steady-state hits ~2.7-3.5 s** on H100/H200. If your numbers are worse, you might have forgotten `--enable-dit-cache`.
3. **`unnormalized_action.shape == (1, 24, 7)`** plus the gripper concat → wire shape `(24, 8)`. If you see different shapes the server is misconfigured.

## What's in the MP4s

The server reconstructs and saves what the model **predicted the future would look like**, not what actually happened (since this was a smoke test with stub video frames, not a real robot). The `nN` suffix in the filename indicates `N` action chunks were inferred, so `n8` ≈ 1.6 s × 8 = ~13 s of predicted "future" per inference cumulative.

Looking at these MP4s side by side with your own real-robot videos can help you debug observation-format issues (e.g., if you sent BGR instead of RGB, the predicted videos will look pink/cyan).
