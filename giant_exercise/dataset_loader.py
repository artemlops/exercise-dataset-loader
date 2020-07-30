import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


logger = logging.getLogger(__name__)

__all__ = [
    "DepthFrameMeta",
    "RgbFrameMeta",
    "ObservationMeta",
    "read_depth_frames_meta",
    "read_rgb_frames_meta",
    "read_observations_meta",
]


@dataclass
class DepthFrameMeta:
    id: int
    ms: int

    @property
    def file_name(self) -> str:
        return f"frame-{self.id:06}.png"


@dataclass
class RgbFrameMeta:
    id: int
    ms: int


@dataclass
class ObservationMeta:
    id: int
    ms: int


def _read_lines(file: Path) -> Sequence[str]:
    lines = [line.split(";")[0].strip() for line in file.read_text().splitlines()]
    lines = [line for line in lines if line]
    return lines


def _parse_ms_id_line(line: str, error_prefix: str) -> Tuple[int, int]:
    e = error_prefix
    spl = line.split()
    if len(spl) != 2:
        raise ValueError(f"{e}: expect 2 elements, got {len(spl)}: {spl}")
    ms_str, frame_id_str = spl

    try:
        ms = int(ms_str)
        if ms < 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"{e}: 1st element must be positive int, got: {ms_str}")

    try:
        id = int(frame_id_str)
        if id < 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"{e}: 2nd element must be positive int, got: {frame_id_str}")

    return ms, id


def read_depth_frames_meta(file: Path) -> Sequence[DepthFrameMeta]:
    err = f"Invalid depth frames meta file `{file}`"
    frames = []
    for i, line in enumerate(_read_lines(file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        frames.append(DepthFrameMeta(id, ms))
    return frames


def read_rgb_frames_meta(file: Path) -> Sequence[RgbFrameMeta]:
    err = f"Invalid rgb frames meta file `{file}`"
    frames = []
    for i, line in enumerate(_read_lines(file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        frames.append(RgbFrameMeta(id, ms))
    return frames


def read_observations_meta(file: Path) -> Sequence[ObservationMeta]:
    err = f"Invalid observations meta file `{file}`"
    observations = []
    for i, line in enumerate(_read_lines(file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        observations.append(ObservationMeta(id, ms))
    return observations


def load_observation(file: Path) -> Sequence[str]:
    err = f"Invalid observation file `{file}`"
    lines = _read_lines(file)
    if not lines:
        raise ValueError(f"{err}: empty file")
    if len(lines) > 1:
        raise ValueError(f"{err}: must be exactly 1 line, found: {len(lines)}")
    return lines[0].split()
