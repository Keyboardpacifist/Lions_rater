"""GAS Score — proprietary player grading engine.

The GAS (Game-Adjusted Skill) Score is the canonical Lions Rater
benchmark for every player at every position. One number per player
per season, on a 0-100 scale, on a single consistent meaning across
all positions.

Default view convention: a player's headline GAS Score is their MOST
RECENT season's score (last year of college or NFL). Career history is
accessible as a toggle but is not the default — the default question
"how good is this guy?" should be answered by his last full season,
not a 7-year career average that smooths out everything that's
changed about him.

Design principles (from the strategic plan):
  • One number per (player, season). Plus per-bundle sub-grades.
  • Every weight comes from a transparent rationale; no magic numbers.
  • Stable across populations: a 90 GAS = +1.7σ in the position
    population for that season. A 50 GAS = league average.
  • Sample-size aware. Players with thin samples get shrunk toward
    the position median + flagged with confidence labels.

Public API
----------
    z_to_grade(z, ceiling=99, floor=1) -> 0-100 grade
    grade_to_z(g) -> z (inverse)
    confidence_for_n(n) -> "HIGH" | "MEDIUM" | "LOW"
    bundle_grade(stat_grades, weights) -> 0-100 weighted average
    composite_grade(bundle_grades, bundle_weights) -> 0-100
    shrunk_z(player_z, n, prior_z=0.0, tau=4) -> shrunk z toward prior
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt


# ── z → grade mapping ──────────────────────────────────────────────
# Using the standard normal CDF so a z of +1.7 ≈ 95th percentile = 95
# and a z of 0 = 50th percentile = 50. Linear in percentile space (not
# raw z) to match how fans intuitively think about player rankings.

def _normal_cdf(z: float) -> float:
    """Standard normal CDF. Returns probability in [0, 1]."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _normal_inverse(p: float) -> float:
    """Inverse standard normal — Beasley-Springer-Moro approximation.
    Good enough for our purposes (grading)."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    # Constants for Beasley-Springer
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = (-2 * (p > 0)) ** 0.5 if p > 0 else 0
        q = sqrt(-2 * (1e-300 if p == 0 else __import__("math").log(p)))
        return ((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
                / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    if p > p_high:
        q = sqrt(-2 * __import__("math").log(1 - p))
        return -((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
                 / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    q = p - 0.5
    r = q * q
    return ((((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
            / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1))


def z_to_grade(z: float, ceiling: float = 99.0,
                  floor: float = 1.0) -> float:
    """Map a z-score to a 0-100 grade via the standard normal CDF.

    z=0 → 50 (league average)
    z=+1.0 → ~84 (above average)
    z=+1.7 → ~95 (elite)
    z=+2.5 → ~99 (historic)
    z=-1.0 → ~16 (well below avg)

    Capped at [floor, ceiling] to avoid a single +4σ outlier game from
    pinning a player at exactly 100.
    """
    if z is None:
        return 50.0
    try:
        if z != z:   # NaN check
            return 50.0
    except Exception:
        return 50.0
    grade = 100.0 * _normal_cdf(float(z))
    return max(floor, min(ceiling, grade))


def grade_to_z(grade: float) -> float:
    """Inverse of z_to_grade. Useful for computing composite z's
    from grade scores (going back through the pipeline)."""
    p = max(0.001, min(0.999, float(grade) / 100.0))
    return _normal_inverse(p)


# ── Sample-size handling ───────────────────────────────────────────

def confidence_for_n(n: int,
                       high: int = 14, medium: int = 8) -> str:
    """For a season grade: HIGH = ≥14 games, MEDIUM = 8-13, LOW < 8.
    Tunable per position (kickers and punters use different thresholds)."""
    if n >= high:
        return "HIGH"
    if n >= medium:
        return "MEDIUM"
    return "LOW"


def shrunk_z(player_z: float, n: int,
              prior_z: float = 0.0, tau: float = 4.0) -> float:
    """Beta-Binomial-style shrinkage of a player's z toward a prior z.

    `tau` is the effective game-count of the prior. With tau=4, a player
    with n=4 games gets a 50/50 blend with the prior; n=20 games gets
    20/24 = 83% weight on their own z.

    Conservative default: shrink toward 0 (league average) so thin
    samples regress to the mean.
    """
    if n <= 0:
        return prior_z
    if player_z is None or player_z != player_z:
        return prior_z
    return (n * float(player_z) + tau * float(prior_z)) / (n + tau)


# ── Bundle / composite aggregators ─────────────────────────────────

@dataclass
class BundleSpec:
    """One bundle definition: which stat-grade columns feed it,
    and what relative weights to use within the bundle."""
    name: str
    stats: dict[str, float] = field(default_factory=dict)
        # {stat_z_col: weight_within_bundle}


@dataclass
class PositionGradeSpec:
    """The full grading recipe for a position. Bundles + their relative
    weights in the composite. Weights don't need to sum to 1 — they're
    normalized at compute time."""
    position: str
    bundles: dict[str, BundleSpec]
    bundle_weights: dict[str, float]
    name_for_grade: str = "GAS Score"

    def normalized_bundle_weights(self) -> dict[str, float]:
        total = sum(self.bundle_weights.values())
        if total <= 0:
            return self.bundle_weights
        return {k: v / total for k, v in self.bundle_weights.items()}


def bundle_grade(stat_grades: dict[str, float],
                   weights: dict[str, float]) -> float:
    """Weighted average of a bundle's stat grades.
    `stat_grades` is {stat_col: grade}, `weights` is {stat_col: w}.
    Stat columns missing from `stat_grades` get a fallback grade of 50.
    """
    total_w = 0.0
    weighted_sum = 0.0
    for stat, w in weights.items():
        g = stat_grades.get(stat, 50.0)
        if g is None or g != g:
            g = 50.0
        weighted_sum += w * g
        total_w += w
    if total_w == 0:
        return 50.0
    return weighted_sum / total_w


def composite_grade(bundle_grades: dict[str, float],
                      bundle_weights: dict[str, float]) -> float:
    """Composite from per-bundle grades, weighted by bundle importance."""
    total_w = 0.0
    weighted_sum = 0.0
    for bundle, w in bundle_weights.items():
        g = bundle_grades.get(bundle)
        if g is None or g != g:
            g = 50.0
        weighted_sum += w * g
        total_w += w
    if total_w == 0:
        return 50.0
    return weighted_sum / total_w


def grade_label(g: float) -> str:
    """Human-friendly label for a 0-100 grade. Aligned to PFF-style
    color-band conventions."""
    if g >= 90:
        return "Elite"
    if g >= 80:
        return "High-end starter"
    if g >= 70:
        return "Above average"
    if g >= 55:
        return "Solid starter"
    if g >= 45:
        return "Average / replaceable"
    if g >= 35:
        return "Below average"
    return "Poor"
