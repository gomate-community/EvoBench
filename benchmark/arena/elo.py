from dataclasses import dataclass


@dataclass
class EloConfig:
    k_factor: float = 32.0
    initial_rating: float = 1000.0


def expected_score(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(ra: float, rb: float, result_a: float, cfg: EloConfig = EloConfig()) -> tuple[float, float]:
    ea = expected_score(ra, rb)
    eb = expected_score(rb, ra)
    return ra + cfg.k_factor * (result_a - ea), rb + cfg.k_factor * ((1 - result_a) - eb)


def normalize_rating(rating: float, low_anchor: float = 800, high_anchor: float = 1400) -> float:
    return max(0.0, min(100.0, 100 * (rating - low_anchor) / (high_anchor - low_anchor)))
