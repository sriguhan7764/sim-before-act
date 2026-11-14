# Troubleshooting and recovery

| Symptom | Likely cause / action |
|---|---|
| `camera ... NOT connected: ... device unset` | Config value left null on purpose; fill from `inspect`. |
| RealSense not listed | Use a USB3 port/cable; close other processes using the camera; `pyrealsense2` must see it (`inspect`). |
| `robot port is unset` / cannot open port | Set `robot.follower.port`; if another process holds the port, use the tap (docs/COEXISTENCE.md). Never run two bus pollers. |
| `read_ok: false` records | Bus read errors (CRC/timeouts). Kept and flagged; frequent => cable/power/baud. |
| `unmatched_beyond_tolerance` many | Telemetry slower than tolerance or stalled; check telemetry Hz, widen `sync.max_state_tolerance_ms` only if physically justified. |
| `complete_with_drops` / `aborted_writer_overflow` | Writer too slow: lower resolution/fps, faster disk, raise `queue_max_items` (memory!), or set `overflow_policy: abort`. Drops are counted in `episode.json` and `events.jsonl`. |
| `aborted_low_disk` | Free space under `recording.min_free_disk_gb`. |
| `readiness gates failed` (research) | Message lists the failed gates; fix, or use `mode: diagnostic` (episodes are flagged). |
| Export `BLOCKED` | The message lists missing required signals (e.g. no robot calibration file => no joint angles). `--allow-partial` writes a manifest with `complete:false`. |
| Large reprojection error | Wrong marker id/orientation/dictionary, wrong size (white margin included), markers not flat. |
| Preview shows no window | Needs `$DISPLAY`; use `--headless` / `--duration`. |

## Interrupted recordings
A crash/power loss leaves `<id>.inprogress/` (index lines are only written after their image files exist; metadata is
flushed every `flush_interval_s`). Recover:

```bash
python -m collector recover --episode data/episodes/<id>.inprogress
```
Torn trailing lines and index entries with missing files are discarded (originals kept as `*.corrupt`), tables and
manifest are regenerated, and the episode is published as `status: interrupted_recovered`, `outcome: aborted`,
`quality_flags.recovered/partial: true`. It is **never** presented as complete and lightweight exports are blocked unless
`--allow-partial`. Data that was still in the queue at the crash is lost and cannot be reconstructed.

## Integrity
`validate --episode <dir>` re-hashes every file against `manifest.json`, checks monotonic timestamps, frame gaps, uint16
depth decodability and association health. Finalised masters are read-only (0444). Never edit them: use
`annotate` (versioned sidecar `<id>.annotations/vNNN.json`) or new export versions.
