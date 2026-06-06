from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ClipMetadata:
    source_video: Path
    clip_id: str
    subset: str


def sanitize_clip_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    return sanitized or "clip"


def build_frame_image_name(clip_id: str, index: int) -> str:
    return f"{clip_id}_{index:06d}.jpg"


def infer_clip_metadata(
    video_path: str | Path,
    raw_root: str | Path,
    default_subset: str = "unspecified",
) -> ClipMetadata:
    source_video = Path(video_path)
    root = Path(raw_root)
    try:
        relative = source_video.relative_to(root)
    except ValueError:
        subset = default_subset
    else:
        subset = relative.parts[0] if len(relative.parts) > 1 else default_subset
    clip_id = sanitize_clip_id(source_video.stem)
    return ClipMetadata(source_video=source_video, clip_id=clip_id, subset=subset)


def list_video_files(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root)
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
