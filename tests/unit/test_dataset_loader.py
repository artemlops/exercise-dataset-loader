from pathlib import Path
from textwrap import dedent

from giant_exercise.dataset_loader import (
    DepthFrameMeta,
    ObservationMeta,
    RgbFrameMeta,
    load_observation,
    read_depth_frames_meta,
    read_observations_meta,
    read_rgb_frames_meta,
)


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
    assert read_depth_frames_meta(path) == [
        DepthFrameMeta(ms=1000, id=0),
        DepthFrameMeta(ms=1666, id=1),
        DepthFrameMeta(ms=2333, id=2),
        DepthFrameMeta(ms=3000, id=3),
    ]


# TODO: test_read_depth_frames_meta_invalid_frame_elements_number
# TODO: test_read_depth_frames_meta_invalid_frame_non_int_ms
# TODO: test_read_depth_frames_meta_invalid_frame_non_int_frame_id
# TODO: test_read_depth_frames_meta_invalid_frame_negative_ms
# TODO: test_read_depth_frames_meta_invalid_frame_negative_frame_id


def test_read_rgb_frames_meta_ok(tmp_path: Path) -> None:
    path = tmp_path / "per_frame_timestamps.txt"
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
    assert read_rgb_frames_meta(path) == [
        RgbFrameMeta(ms=1000, id=0),
        RgbFrameMeta(ms=1666, id=1),
        RgbFrameMeta(ms=2333, id=2),
        RgbFrameMeta(ms=3000, id=3),
    ]


# TODO: test_read_rgb_frames_meta_invalid_frame_elements_number
# TODO: test_read_rgb_frames_meta_invalid_frame_non_int_ms
# TODO: test_read_rgb_frames_meta_invalid_frame_non_int_frame_id
# TODO: test_read_rgb_frames_meta_invalid_frame_negative_ms
# TODO: test_read_rgb_frames_meta_invalid_frame_negative_frame_id


def test_read_observations_meta_ok(tmp_path: Path) -> None:
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
    assert read_observations_meta(path) == [
        ObservationMeta(ms=1000, id=0),
        ObservationMeta(ms=1666, id=1),
        ObservationMeta(ms=2333, id=2),
        ObservationMeta(ms=3000, id=3),
    ]


# TODO: test_read_observations_meta_invalid_frame_elements_number
# TODO: test_read_observations_meta_invalid_frame_non_int_ms
# TODO: test_read_observations_meta_invalid_frame_non_int_frame_id
# TODO: test_read_observations_meta_invalid_frame_negative_ms
# TODO: test_read_observations_meta_invalid_frame_negative_frame_id


def test_load_observation(tmp_path: Path) -> None:
    path = tmp_path / "per_observation_timestamps.txt"
    path.write_text(
        dedent(
            """\
        ; this is a comment
        v_1 v_2 v_3 v_4

        """
        )
    )
    assert load_observation(path) == ["v_1", "v_2", "v_3", "v_4"]


# TODO: test_load_observation_invalid_empty
# TODO: test_load_observation_invalid_empty_with_comments
# TODO: test_load_observation_invalid_multiple_lines
