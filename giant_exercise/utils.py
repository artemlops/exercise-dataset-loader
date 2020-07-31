from typing import Iterator, Sequence, Tuple, Union


def zip_closest(
    main: Sequence[int],
    secondary: Sequence[int],
    *,
    linearize: bool = False,
    step: int = 1,
) -> Iterator[Tuple[int, int]]:
    """
    Iterates over 'main' sequence while taking the closest element
    from 'secondary' sequence.
    >>> a = [1, 2, 7, 8, 9, 15, 20]
    >>> b = [0, 1, 4, 6, 9, 10]
    >>> list(zip_closest(a, b))
    [(1, 1), (2, 1), (7, 6), (8, 9), (9, 9), (15, 10), (20, 10)]
    """
    m_iterator = iter(main)
    s_iterator = iter(secondary)

    # pointers for 'secondary' sequence:
    s = None
    s_next = next(s_iterator, None)

    # pointers for 'main' sequence:
    m_prev = None
    m = None
    m_next = next(m_iterator, None)

    # deltas between 's' and 'm' used to find the closest 's' to 'm'
    delta: Union[int, float] = float("inf")
    delta_next: Union[int, float]

    while True:
        if m_next is None:
            # 'm' is the latest element of 'main'
            break
        if not linearize or m_prev is None or m >= m_next - step:
            # update 'm' from the 'main' sequence
            m = m_next
            m_next = next(m_iterator, None)
        else:
            # when 'linearize=True', we fill the gaps between elements
            # of 'm' with new elements with step 'step'
            m = m_prev + step

        assert m is not None, "can never be null"
        if s_next is not None:
            if s is not None:
                delta = abs(m - s)
            delta_next = abs(m - s_next)

            while 0 < delta_next <= delta:
                s, delta = s_next, delta_next
                s_next = next(s_iterator, None)
                if s_next is None:
                    delta_next = float("inf")
                    break
                delta_next = abs(m - s_next)

            s = s if delta < delta_next else s_next

        # else: end of list, will take the latest
        elif s is None:
            raise ValueError(
                "Got empty secondary sequence with non-empty main sequence"
            )

        assert s is not None, "can never be null"
        yield (m, s)

        m_prev = m
