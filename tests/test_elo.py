from benchmark.arena.elo import update_elo, normalize_rating


def test_update_elo_win():
    a, b = update_elo(1000, 1000, 1.0)
    assert a > 1000
    assert b < 1000


def test_normalize_rating():
    assert normalize_rating(800) == 0
    assert normalize_rating(1400) == 100
