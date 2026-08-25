#!/usr/bin/env bash
# Mount a data HDD for continuous camera recordings.
#
# Target layout after mount:
#   /mnt/video_hdd/video_record/{counter_camera,entry_camera}/…
#
# Usage:
#   sudo bash scripts/mount_recording_hdd.sh
#   sudo bash scripts/mount_recording_hdd.sh /dev/sda1
#
# Notes:
#   - sda/sdb on this host are currently BitLocker (Windows). Unlock first
#     (dislocker + recovery key) or replace with an unlocked ext4/ntfs volume.
#   - App prefers RECORDING_HDD_MOUNT when it is a live mount; else Desktop.

set -euo pipefail

MOUNT_POINT="${RECORDING_HDD_MOUNT:-/mnt/video_hdd}"
DEVICE="${1:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo bash $0 ${DEVICE}"
  exit 1
fi

mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
  echo "Already mounted: $MOUNT_POINT"
  df -h "$MOUNT_POINT"
  mkdir -p "$MOUNT_POINT/video_record"
  exit 0
fi

if [[ -z "$DEVICE" ]]; then
  echo "Block devices:"
  lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT,ROTA,MODEL
  echo
  echo "Pick a partition, e.g.:"
  echo "  sudo bash $0 /dev/sda1"
  echo
  # Auto-pick first unmounted non-system disk partition with a linux/ntfs fs
  for cand in /dev/sd[ab]*[0-9] /dev/nvme*n*p*[0-9]; do
    [[ -b "$cand" ]] || continue
    mp=$(lsblk -no MOUNTPOINT "$cand" 2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$mp" ]] && continue
    fs=$(lsblk -no FSTYPE "$cand" 2>/dev/null | head -1 | tr -d ' ')
    case "$fs" in
      ext4|xfs|ntfs|ntfs3|exfat|btrfs)
        DEVICE="$cand"
        echo "Auto-selected $DEVICE (fstype=$fs)"
        break
        ;;
      BitLocker|crypto_LUKS|"")
        echo "Skip $cand (fstype=${fs:-unknown} — unlock BitLocker/LUKS first)"
        ;;
    esac
  done
fi

if [[ -z "$DEVICE" || ! -b "$DEVICE" ]]; then
  echo "ERROR: No usable device. Unlock BitLocker or pass DEVICE explicitly."
  exit 2
fi

FS=$(lsblk -no FSTYPE "$DEVICE" | head -1 | tr -d ' ')
echo "Mounting $DEVICE (fstype=$FS) → $MOUNT_POINT"

case "$FS" in
  ntfs|ntfs3)
    mount -t ntfs3 -o uid=1000,gid=1000,umask=002 "$DEVICE" "$MOUNT_POINT" \
      || mount -t ntfs-3g -o uid=1000,gid=1000,umask=002 "$DEVICE" "$MOUNT_POINT"
    ;;
  ext4|xfs|btrfs|exfat)
    mount "$DEVICE" "$MOUNT_POINT"
    chown -R retaileye:retaileye "$MOUNT_POINT" 2>/dev/null || true
    ;;
  BitLocker)
    echo "ERROR: $DEVICE is BitLocker-encrypted."
    echo "Unlock with dislocker, then remount the cleartext loop, e.g.:"
    echo "  sudo apt install dislocker"
    echo "  sudo mkdir -p /mnt/bitlocker_raw $MOUNT_POINT"
    echo "  sudo dislocker -V $DEVICE -u -- /mnt/bitlocker_raw"
    echo "  sudo mount -o loop /mnt/bitlocker_raw/dislocker-file $MOUNT_POINT"
    exit 3
    ;;
  *)
    echo "Trying generic mount for fstype='$FS'…"
    mount "$DEVICE" "$MOUNT_POINT"
    ;;
esac

mkdir -p "$MOUNT_POINT/video_record/counter_camera" \
         "$MOUNT_POINT/video_record/entry_camera"
chown -R retaileye:retaileye "$MOUNT_POINT/video_record" 2>/dev/null || true

echo "OK:"
df -h "$MOUNT_POINT"
ls -la "$MOUNT_POINT/video_record"
echo
echo "Set in .env (optional if defaults used):"
echo "  RECORDING_HDD_MOUNT=$MOUNT_POINT"
echo "  RECORDING_ROOT=$MOUNT_POINT/video_record"
echo "  ENABLE_CAMERA_RECORDING=true"
