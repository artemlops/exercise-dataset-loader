import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import IterableDataset

from giant_exercise.utils import zip_closest


logger = logging.getLogger(__name__)

VIDEO_FILE_NAME = "video.mp4"
DEPTH_META_REL_PATH = Path("depth/per_frame_timestamps.txt")
RGB_META_REL_PATH = Path("rgb/per_frame_timestamps.txt")
OBSERVATION_META_REL_PATH = Path("touch/per_observation_timestamps.txt")

RGB_FRAME_CACHE_SIZE = 5
DEPTH_FRAME_CACHE_SIZE = 5
OBSERVATION_FRAME_CACHE_SIZE = 5


@dataclass(frozen=True)
class RgbFrameMeta:
    id: int
    ms: int
    video_path: Path


@dataclass(frozen=True)
class DepthFrameMeta:
    id: int
    ms: int
    base_dir: Path

    @property
    def _file_name(self) -> str:
        return f"frame-{self.id:06}.png"

    @property
    def file_path(self) -> Path:
        return self.base_dir / self._file_name


@dataclass(frozen=True)
class ObservationMeta:
    id: int
    ms: int
    base_dir: Path

    @property
    def _file_name(self) -> str:
        return f"observation-{self.id:06}.txt"

    @property
    def file_path(self) -> Path:
        return self.base_dir / self._file_name


class Video:
    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "Video":
        assert not self._cap
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise ValueError(f"Could open video file {self._path}")
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        assert self._cap and self._cap.isOpened()
        self._cap.release()
        self._cap = None

    def get_fps(self) -> int:
        assert self._cap and self._cap.isOpened()
        return self._cap.get(cv2.CAP_PROP_FPS)

    def get_frame_size(self) -> Tuple[int, int]:
        assert self._cap and self._cap.isOpened()
        w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(w), int(h)

    def seek_read_frame(self, ms: int) -> Optional[np.ndarray]:
        assert self._cap and self._cap.isOpened()
        self._cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ret, frame = self._cap.read()
        if not ret:  # EOF
            return None
        return frame

    def create_video_writer(
        self, out_path: Union[str, Path], fmt: str = "mp4v"
    ) -> cv2.VideoWriter:
        assert self._cap and self._cap.isOpened()
        fourcc = cv2.VideoWriter_fourcc(*fmt)
        return cv2.VideoWriter(
            str(out_path), fourcc, self.get_fps(), self.get_frame_size()
        )


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
        raise ValueError(f"{e}: 1st element must be a positive int, got: {ms_str}")

    try:
        id = int(frame_id_str)
        if id < 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"{e}: 2nd element must be a positive int, got: {frame_id_str}"
        )

    return ms, id


def read_rgb_frames_meta(meta_file: Path) -> Mapping[int, RgbFrameMeta]:
    err = f"Invalid rgb frames meta file `{meta_file}`"
    meta = {}
    video_path = meta_file.parent / VIDEO_FILE_NAME
    for i, line in enumerate(_read_lines(meta_file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        meta[ms] = RgbFrameMeta(id=id, ms=ms, video_path=video_path)
    return meta


def read_depth_frames_meta(meta_file: Path) -> Mapping[int, DepthFrameMeta]:
    err = f"Invalid depth frames meta file `{meta_file}`"
    base_dir = meta_file.parent
    meta = {}
    for i, line in enumerate(_read_lines(meta_file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        meta[ms] = DepthFrameMeta(id=id, ms=ms, base_dir=base_dir)
    return meta


def read_observations_meta(meta_file: Path) -> Mapping[int, ObservationMeta]:
    err = f"Invalid observations meta file `{meta_file}`"
    base_dir = meta_file.parent
    meta = {}
    for i, line in enumerate(_read_lines(meta_file), 1):
        ms, id = _parse_ms_id_line(line, f"{err}: line {i}")
        meta[ms] = ObservationMeta(id=id, ms=ms, base_dir=base_dir)
    return meta


@lru_cache(maxsize=RGB_FRAME_CACHE_SIZE)
def load_rgb_frame(frame: RgbFrameMeta) -> np.ndarray:
    # TODO: add tests
    path = frame.video_path
    err = f"Could not load rgb frame file `{path}`"
    try:
        with Video(path) as video:
            return video.seek_read_frame(frame.ms)
    except (ValueError, OSError) as e:
        raise ValueError(f"{err}: {e}") from e


@lru_cache(maxsize=DEPTH_FRAME_CACHE_SIZE)
def load_depth_frame(frame: DepthFrameMeta) -> np.ndarray:
    # TODO: add tests
    path = frame.file_path
    err = f"Could not load depth frame file `{path}`"
    try:
        img = Image.open(path)
        return np.array(img.getdata())
    except (ValueError, OSError) as e:
        raise ValueError(f"{err}: {e}") from e


@lru_cache(maxsize=OBSERVATION_FRAME_CACHE_SIZE)
def load_observation(obs: Optional[ObservationMeta] = None) -> Sequence[int]:
    # NOTE: tested in 'TestFunctionLoadObservation'
    if obs is None:
        return []

    path = obs.file_path
    err = f"Could not load observation file `{path}`"
    try:
        lines = _read_lines(path)
    except (ValueError, OSError) as e:
        raise ValueError(f"{err}: {e}") from e
    if not lines:
        raise ValueError(f"{err}: no observations found in file")
    if len(lines) > 1:
        raise ValueError(f"{err}: must be exactly 1 line, found: {len(lines)}")
    try:
        return [int(num) for num in lines[0].split()]
    except ValueError as e:
        raise ValueError(f"{err}: {e}")


class DataItem(NamedTuple):
    # Properties required in task
    touch_timestamp_i: int  # ms
    touch_i: Sequence[int]  # observation
    rgb_j: np.ndarray  # rgb frame data
    depth_k: np.ndarray  # depth frame data
    # Additional properties for debugging
    rgb_timestamp_j: int
    depth_timestamp_k: int


class GiantDataset(IterableDataset):  # type: ignore
    def __init__(self, root: Union[str, Path], linearize: bool = False):
        super().__init__()
        self._rgb_mapping = read_rgb_frames_meta(root / RGB_META_REL_PATH)
        self._depth_mapping = read_depth_frames_meta(root / DEPTH_META_REL_PATH)
        self._obs_mapping = read_observations_meta(root / OBSERVATION_META_REL_PATH)

        self._linearize = linearize
        self._step: Optional[int] = None  # step between frames in ms
        if self._linearize:
            first_rgb_frame = next(iter(self._rgb_mapping.values()))
            with Video(first_rgb_frame.video_path) as video:
                fps = video.get_fps()
            self._step = int(1_000 / fps)

    @property
    def step(self) -> Optional[int]:
        return self._step

    def __iter__(self) -> Iterator[DataItem]:
        # TODO: Support multiple workers
        obs_keys = tuple(self._obs_mapping.keys())

        iter_rgb = zip_closest(
            obs_keys,
            tuple(self._rgb_mapping.keys()),
            linearize=self._linearize,
            step=self._step,
        )
        iter_depth = zip_closest(
            obs_keys,
            tuple(self._depth_mapping.keys()),
            linearize=self._linearize,
            step=self._step,
        )

        for ((ts_i_1, rgb_j), (ts_i_2, depth_k)) in zip(iter_rgb, iter_depth):
            assert ts_i_1 == ts_i_2, ("error in 'zip_closest'", ts_i_1, ts_i_2)
            logger.debug(
                f"Loading touch ms {ts_i_1}, rgb ms {rgb_j}, depth ms {depth_k}"
            )
            item = DataItem(
                touch_timestamp_i=ts_i_1,
                rgb_timestamp_j=rgb_j,
                depth_timestamp_k=depth_k,
                touch_i=load_observation(self._obs_mapping.get(ts_i_1)),
                rgb_j=load_rgb_frame(self._rgb_mapping[rgb_j]),
                depth_k=load_depth_frame(self._depth_mapping[depth_k]),
            )
            yield item
