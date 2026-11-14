#!/usr/bin/env bash
# LeRobot-based teleop / recording for this setup. Uses ONLY project-local calibration copies (never ~/.cache files).
#   ./run.sh teleop                      # leader -> follower, no cameras
#   ./run.sh record <task text> [N eps]  # teleop + RealSense (RGB+depth) + wrist cam -> LeRobot dataset
# Both arms must be in a similar pose before starting (the follower moves to the leader pose on connect).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH=/home/apic3/lerobot_v6_2/lerobot/src
PY=/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python
FOL=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030118-if00
LEA=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084050-if00
WRIST=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
RS=844212071704
ROBOT=(--robot.type=so101_follower --robot.port=$FOL --robot.id=g_c_follower_calibration
       --robot.calibration_dir=$HERE/calibration/follower --robot.max_relative_target=10)
TELEOP=(--teleop.type=so101_leader --teleop.port=$LEA --teleop.id=g_c_leader_calibration
        --teleop.calibration_dir=$HERE/calibration/leader)
CAMS="{ front: {type: intelrealsense, serial_number_or_name: \"$RS\", width: 640, height: 480, fps: 30, use_depth: true}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30}}"
case "${1:-}" in
  teleop)
    exec $PY -m lerobot.scripts.lerobot_teleoperate "${ROBOT[@]}" "${TELEOP[@]}" --fps=30 --teleop_time_s="${2:-30}" ;;
  record)
    TASK="${2:?task text}"; N="${3:-1}"; NAME="g_c_$(date +%Y%m%d_%H%M%S)"
    exec $PY -m lerobot.scripts.lerobot_record "${ROBOT[@]}" "--robot.cameras=$CAMS" "${TELEOP[@]}" \
      --dataset.repo_id=local/$NAME --dataset.root=$HERE/datasets/$NAME --dataset.push_to_hub=false \
      --dataset.single_task="$TASK" --dataset.num_episodes="$N" --dataset.fps=30 \
      --dataset.episode_time_s=20 --dataset.reset_time_s=5 --display_data=false --play_sounds=false ;;
  *) echo "usage: $0 teleop [seconds] | record \"task\" [episodes]"; exit 2 ;;
esac
