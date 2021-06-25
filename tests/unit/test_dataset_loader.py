from pathlib import Path
from textwrap import dedent

import cv2
import numpy as np
import pytest
from torch import tensor
from torch.utils.data import DataLoader

from dataset_loader.dataset_loader import (
    VIDEO_FILE_NAME,
    DepthFrameMeta,
    MyDataset,
    ObservationMeta,
    RgbFrameMeta,
    Video,
    load_observation,
    read_depth_frames_meta,
    read_observations_meta,
    read_rgb_frames_meta,
)


def test_read_depth_frames_meta_too_many_numbers(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000
        000002000 000001 0000002
        """
        )
    )
    with pytest.raises(ValueError, match="expect 2 elements, got 3"):
        read_depth_frames_meta(path)


def test_read_depth_frames_meta_negative_ms(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000
        -000002000 000001
        """
        )
    )
    with pytest.raises(ValueError, match="1st element must be a positive int"):
        read_depth_frames_meta(path)


def test_read_depth_frames_meta_non_int_ms(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000
        abcd 000001
        """
        )
    )
    with pytest.raises(ValueError, match="1st element must be a positive int"):
        read_depth_frames_meta(path)


def test_read_depth_frames_meta_negative_frame_id(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000
        000002000 -000001
        """
        )
    )
    with pytest.raises(ValueError, match="2nd element must be a positive int"):
        read_depth_frames_meta(path)


def test_read_depth_frames_meta_non_int_frame_id(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000
        000002000 abcd
        """
        )
    )
    with pytest.raises(ValueError, match="2nd element must be a positive int"):
        read_depth_frames_meta(path)


def test_read_depth_frames_meta_ok(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND DEPTH_FRAME_ID
        000001000 000000  ; this is a comment
        000001666 000001
        000002333 000002
        000003000 000003

        """
        )
    )
    assert read_depth_frames_meta(path) == {
        1000: DepthFrameMeta(ms=1000, id=0, base_dir=tmp_path),
        1666: DepthFrameMeta(ms=1666, id=1, base_dir=tmp_path),
        2333: DepthFrameMeta(ms=2333, id=2, base_dir=tmp_path),
        3000: DepthFrameMeta(ms=3000, id=3, base_dir=tmp_path),
    }


# TODO: test_read_depth_frames_meta_invalid_frame_elements_number
# TODO: test_read_depth_frames_meta_invalid_frame_non_int_ms
# TODO: test_read_depth_frames_meta_invalid_frame_non_int_frame_id
# TODO: test_read_depth_frames_meta_invalid_frame_negative_ms
# TODO: test_read_depth_frames_meta_invalid_frame_negative_frame_id


def test_read_rgb_frames_meta_ok(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
    video_path = tmp_path / VIDEO_FILE_NAME
    path.write_text(
        dedent(
            """\
        ; MILLISECOND RGB_FRAME_ID
        000001000 000000  ; this is a comment
        000001666 000001
        000002333 000002
        000003000 000003

        """
        )
    )
    assert read_rgb_frames_meta(path) == {
        1000: RgbFrameMeta(ms=1000, id=0, video_path=video_path),
        1666: RgbFrameMeta(ms=1666, id=1, video_path=video_path),
        2333: RgbFrameMeta(ms=2333, id=2, video_path=video_path),
        3000: RgbFrameMeta(ms=3000, id=3, video_path=video_path),
    }


# TODO: test_read_rgb_frames_meta_invalid_frame_elements_number
# TODO: test_read_rgb_frames_meta_invalid_frame_non_int_ms
# TODO: test_read_rgb_frames_meta_invalid_frame_non_int_frame_id
# TODO: test_read_rgb_frames_meta_invalid_frame_negative_ms
# TODO: test_read_rgb_frames_meta_invalid_frame_negative_frame_id


def test_read_observations_meta(tmp_path: Path) -> None:
    path = tmp_path / "per_observation_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; MILLISECOND RGB_FRAME_ID
        000001000 000000  ; this is a comment
        000001666 000001
        000002333 000002
        000003000 000003

        """
        )
    )
    assert read_observations_meta(path) == {
        1000: ObservationMeta(ms=1000, id=0, base_dir=tmp_path),
        1666: ObservationMeta(ms=1666, id=1, base_dir=tmp_path),
        2333: ObservationMeta(ms=2333, id=2, base_dir=tmp_path),
        3000: ObservationMeta(ms=3000, id=3, base_dir=tmp_path),
    }


# TODO: test_read_observations_meta_invalid_frame_elements_number
# TODO: test_read_observations_meta_invalid_frame_non_int_ms
# TODO: test_read_observations_meta_invalid_frame_non_int_frame_id
# TODO: test_read_observations_meta_invalid_frame_negative_ms
# TODO: test_read_observations_meta_invalid_frame_negative_frame_id


class TestVideo:
    @pytest.fixture
    def video_path(self, dataset_path: Path) -> Path:
        return dataset_path / "rgb" / "video.mp4"

    def test_open_not_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "not-exists.mp4"
        with pytest.raises(ValueError, match="Could open video file"):
            with Video(path):
                pass

    def test_fps_not_opened(self, video_path: Path) -> None:
        video = Video(video_path)
        with pytest.raises(AssertionError):
            video.get_fps()

    def test_frame_size_not_opened(self, video_path: Path) -> None:
        video = Video(video_path)
        with pytest.raises(AssertionError):
            video.get_frame_size()

    def test_seek_read_frame_not_opened(self, video_path: Path) -> None:
        video = Video(video_path)
        with pytest.raises(AssertionError):
            video.seek_read_frame(0)

    def test_create_video_writer_not_opened(self, video_path: Path) -> None:
        video = Video(video_path)
        with pytest.raises(AssertionError):
            video.create_video_writer("/tmp/out.mp4")

    def test_fps(self, video_path: Path) -> None:
        with Video(video_path) as video:
            assert video.get_fps() == 30

    def test_frame_size(self, video_path: Path) -> None:
        with Video(video_path) as video:
            assert video.get_frame_size() == (256, 224)

    def test_seek_read_frame(self, video_path: Path) -> None:
        with Video(video_path) as video:
            frame = video.seek_read_frame(100)
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (224, 256, 3)

    def test_seek_read_frame_not_found(self, video_path: Path) -> None:
        with Video(video_path) as video:
            frame = video.seek_read_frame(100_500 * 1_000_000)
            assert frame is None

    def test_create_video_writer(self, video_path: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "out.mp4"
        with Video(video_path) as video:
            writer = video.create_video_writer(out_path)
            assert isinstance(writer, cv2.VideoWriter)


class TestFunctionLoadObservation:
    def test_load_observation_none(self) -> None:
        assert load_observation(None) == []

    def test_load_observation_not_found(self, tmp_path: Path) -> None:
        obs = ObservationMeta(id=123, ms=100, base_dir=tmp_path)
        with pytest.raises(ValueError, match="No such file or directory"):
            load_observation(obs)

    def test_load_observation_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "observation-000123.txt"
        path.write_text(
            dedent(
                """\
            ; this is a comment

            """
            )
        )
        obs = ObservationMeta(id=123, ms=100, base_dir=tmp_path)
        with pytest.raises(ValueError, match="no observations found in file"):
            load_observation(obs)

    def test_load_observation_too_many_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "observation-000123.txt"
        path.write_text(
            dedent(
                """\
            ; this is a comment
            1 2 3
            4 5 6
            """
            )
        )
        obs = ObservationMeta(id=123, ms=100, base_dir=tmp_path)
        with pytest.raises(ValueError, match="must be exactly 1 line"):
            load_observation(obs)

    def test_load_observation_invalid_observation_format(self, tmp_path: Path) -> None:
        path = tmp_path / "observation-000123.txt"
        path.write_text(
            dedent(
                """\
            ; this is a comment
            v1 v2 v3
            """
            )
        )
        obs = ObservationMeta(id=123, ms=100, base_dir=tmp_path)
        with pytest.raises(ValueError, match="invalid literal for int"):
            load_observation(obs)

    def test_load_observation(self, tmp_path: Path) -> None:
        path = tmp_path / "observation-000123.txt"
        path.write_text(
            dedent(
                """\
            ; this is a comment
            1 2 3 4 5

            """
            )
        )
        obs = ObservationMeta(id=123, ms=100, base_dir=tmp_path)
        assert load_observation(obs) == [1, 2, 3, 4, 5]


class TestMyDataset:
    def test_non_linearize_check_timestamps(self, dataset_path: Path) -> None:
        ds = MyDataset(dataset_path)
        assert ds.step is None

        data = list(DataLoader(ds))
        data_timestamps = [
            (item.touch_timestamp_i, item.rgb_timestamp_j, item.depth_timestamp_k)
            for item in data
        ]
        assert data_timestamps == [
            (tensor([33]), tensor([66]), tensor([1166])),
            (tensor([1066]), tensor([1100]), tensor([1166])),
            (tensor([2100]), tensor([2100]), tensor([2333])),
            (tensor([2833]), tensor([2800]), tensor([2833])),
            (tensor([4000]), tensor([4000]), tensor([4166])),
            (tensor([5000]), tensor([5000]), tensor([5166])),
            (tensor([6033]), tensor([6000]), tensor([5833])),
            (tensor([6366]), tensor([6400]), tensor([5833])),
            (tensor([6500]), tensor([6500]), tensor([5833])),
            (tensor([6600]), tensor([6600]), tensor([5833])),
        ]

    def test_linearize_check_timestamps(self, dataset_path: Path) -> None:
        ds = MyDataset(dataset_path, linearize=True)
        assert ds.step == 33, "computed from the video's FPS"

        data = list(DataLoader(ds))
        data_timestamps = [
            (item.touch_timestamp_i, item.rgb_timestamp_j, item.depth_timestamp_k)
            for item in data
        ]
        assert data_timestamps[:15] == [
            ((tensor([33]), tensor([66]), tensor([1166]))),
            ((tensor([66]), tensor([66]), tensor([1166]))),
            ((tensor([99]), tensor([100]), tensor([1166]))),
            ((tensor([132]), tensor([100]), tensor([1166]))),
            ((tensor([165]), tensor([200]), tensor([1166]))),
            ((tensor([198]), tensor([200]), tensor([1166]))),
            ((tensor([231]), tensor([200]), tensor([1166]))),
            ((tensor([264]), tensor([300]), tensor([1166]))),
            ((tensor([297]), tensor([300]), tensor([1166]))),
            ((tensor([330]), tensor([300]), tensor([1166]))),
            ((tensor([363]), tensor([400]), tensor([1166]))),
            ((tensor([396]), tensor([400]), tensor([1166]))),
            ((tensor([429]), tensor([400]), tensor([1166]))),
            ((tensor([462]), tensor([500]), tensor([1166]))),
            ((tensor([495]), tensor([500]), tensor([1166]))),
        ]
