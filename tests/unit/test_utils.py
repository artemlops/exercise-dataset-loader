import pytest

from dataset_loader.utils import zip_closest


@pytest.mark.parametrize("linearize", [True, False])
def test_zip_closest_empty_main(linearize: bool) -> None:
    main = ()
    secondary = (1, 2, 3)
    assert list(zip_closest(main, secondary, linearize=linearize)) == []


@pytest.mark.parametrize("linearize", [True, False])
def test_zip_closest_empty_main_empty_secondary(linearize: bool) -> None:
    main = ()
    secondary = ()
    assert list(zip_closest(main, secondary, linearize=linearize)) == []


@pytest.mark.parametrize("linearize", [True, False])
def test_zip_closest_nonempty_main_empty_secondary(linearize: bool) -> None:
    main = (1, 2, 3)
    secondary = ()
    with pytest.raises(
        ValueError, match="Got empty secondary sequence with non-empty main sequence"
    ):
        list(zip_closest(main, secondary, linearize=linearize))


def test_zip_closest_linearize_requires_step() -> None:
    main = (1, 2, 3)
    secondary = (1, 2, 3)
    with pytest.raises(ValueError, match="Option 'linearize' requires 'step' defined"):
        list(zip_closest(main, secondary, linearize=True, step=None))


def test_zip_closest_non_linearize_skip_middle_secondaries() -> None:
    main = (1, 2, 7)
    secondary = (1, 5, 6)
    combined = list(zip_closest(main, secondary))
    assert combined == [
        (1, 1),
        (2, 1),
        (7, 6),  # 7 is closer to 6 than to 5
    ]


def test_zip_closest_non_linearize_skip_middle_secondaries_exact_match() -> None:
    main = (1, 2, 7)
    secondary = (1, 5, 7)
    combined = list(zip_closest(main, secondary))
    assert combined == [
        (1, 1),
        (2, 1),
        (7, 7),  # 7 is closer to 7 than to 5
    ]


@pytest.mark.parametrize("linearize", [True, False])
def test_zip_closest_skip_first_secondary(linearize: bool) -> None:
    main = (1, 2)
    secondary = (0, 1, 2)
    combined = list(zip_closest(main, secondary, linearize=linearize))
    assert combined == [
        (1, 1),  # skipping 0
        (2, 2),
    ]


@pytest.mark.parametrize("linearize", [True, False])
def test_zip_closest_skip_last_secondary(linearize: bool) -> None:
    main = (1, 2)
    secondary = (1, 2, 10)
    assert list(zip_closest(main, secondary, linearize=linearize)) == [
        (1, 1),
        (2, 2),  # skipping 10
    ]


def test_zip_closest_non_linearize_complex_join_two() -> None:
    main = (1, 2, 4, 8, 10, 20, 50, 100_000)
    s1 = (1, 2, 3, 4, 10, 11, 30, 50, 100)
    s2 = (0, 3, 15, 20, 40, 10_000)
    combined = list(
        zip(
            zip_closest(main, s1, linearize=False),
            zip_closest(main, s2, linearize=False),
        )
    )
    assert combined == [
        ((1, 1), (1, 0)),
        ((2, 2), (2, 3)),
        ((4, 4), (4, 3)),
        ((8, 10), (8, 3)),
        ((10, 10), (10, 15)),
        ((20, 11), (20, 20)),
        ((50, 50), (50, 40)),
        ((100_000, 100), (100_000, 10_000)),
    ]


def test_zip_closest_non_linearize_complex_join_three() -> None:
    m = (1, 2, 4, 8, 9)
    s1 = (1, 2, 3, 4, 9)
    s2 = (1, 2, 5, 7, 8)
    s3 = (0, 3, 5, 9, 10, 15)
    combined = list(
        zip(
            zip_closest(m, s1, linearize=False),
            zip_closest(m, s2, linearize=False),
            zip_closest(m, s3, linearize=False),
        )
    )
    assert combined == [
        ((1, 1), (1, 1), (1, 0)),
        ((2, 2), (2, 2), (2, 3)),
        ((4, 4), (4, 5), (4, 5)),
        ((8, 9), (8, 8), (8, 9)),
        ((9, 9), (9, 8), (9, 9)),
    ]


def test_zip_closest_linearize_non_default_step_1() -> None:
    main = (1, 3, 10, 21, 40)
    secondary = (1, 4, 6, 9, 10, 15, 45)
    combined = list(zip_closest(main, secondary, linearize=True, step=5))
    assert combined == [
        (1, 1),
        (3, 4),  # +5
        (8, 9),
        (10, 10),  # +5
        (15, 15),  # +5
        (20, 15),
        (21, 15),  # +5
        (26, 15),  # +5
        (31, 45),  # +5
        (36, 45),  # +5
        (40, 45),
    ]


def test_zip_closest_linearize_non_default_step_2() -> None:
    main = (1, 2, 7, 8)
    secondary = (1, 4, 6, 9, 10)
    combined = list(zip_closest(main, secondary, linearize=True, step=2))
    assert combined == [
        (1, 1),
        (2, 1),  # +2
        (4, 4),  # +2
        (6, 6),
        (7, 6),
        (8, 9),  # skipping 10
    ]


def test_zip_closest_linearize_complex() -> None:
    main = (1, 2, 7, 8)
    secondary = (1, 4, 6, 9, 10)
    combined = list(zip_closest(main, secondary, linearize=True))
    assert combined == [
        (1, 1),
        (2, 1),
        (3, 4),
        (4, 4),
        (5, 6),
        (6, 6),
        (7, 6),
        (8, 9),  # skipping 10
    ]


def test_zip_closest_linearize_complex_join_two() -> None:
    m = (1, 2, 5, 7)
    s1 = (1, 3, 5)
    s2 = (1, 2, 4, 7, 8)
    combined = list(
        zip(zip_closest(m, s1, linearize=True), zip_closest(m, s2, linearize=True))
    )
    assert combined == [
        ((1, 1), (1, 1)),
        ((2, 3), (2, 2)),
        ((3, 3), (3, 4)),
        ((4, 5), (4, 4)),
        ((5, 5), (5, 4)),
        ((6, 5), (6, 7)),
        ((7, 5), (7, 7)),
    ]


def test_zip_closest_linearize_complex_join_three() -> None:
    m = (1, 2, 5, 7)
    s1 = (1, 3, 5)
    s2 = (1, 2, 4, 7, 8)
    s3 = (2, 4, 5, 8)
    combined = list(
        zip(
            zip_closest(m, s1, linearize=True),
            zip_closest(m, s2, linearize=True),
            zip_closest(m, s3, linearize=True),
        )
    )
    assert combined == [
        ((1, 1), (1, 1), (1, 2)),
        ((2, 3), (2, 2), (2, 2)),
        ((3, 3), (3, 4), (3, 4)),
        ((4, 5), (4, 4), (4, 4)),
        ((5, 5), (5, 4), (5, 5)),
        ((6, 5), (6, 7), (6, 5)),
        ((7, 5), (7, 7), (7, 8)),
    ]
