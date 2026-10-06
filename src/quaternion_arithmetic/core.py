"""Four-component real quaternions represented as (w, x, y, z) tuples.

Layout choice: scalar-first (w, x, y, z). This matches the convention used in
most rotation libraries (e.g. scipy, Eigen) and makes the rotation matrix
derivation in `rotate` the standard textbook form. Scalar-last (x, y, z, w)
would require swapping indices throughout; we do not support it.

All arithmetic is done in plain Python floats. There is no vectorization —
the library is small and explicit on purpose.
"""

from __future__ import annotations

import math
from typing import Tuple

# A quaternion is a 4-tuple of real numbers in (w, x, y, z) order.
Quat = Tuple[float, float, float, float]


def hamilton(a: Quat, b: Quat) -> Quat:
    """Return the Hamilton product a * b.

    Quaternion multiplication is non-commutative: in general hamilton(a, b)
    != hamilton(b, a). The formula below is the direct expansion of
    (a0 + a1 i + a2 j + a3 k)(b0 + b1 i + b2 j + b3 k) using i^2 = j^2 =
    k^2 = ijk = -1.
    """
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def conjugate(q: Quat) -> Quat:
    """Return the conjugate (w, -x, -y, -z).

    For a unit quaternion the conjugate equals the inverse; this property is
    what makes `rotate` reversible without an explicit division.
    """
    w, x, y, z = q
    return (w, -x, -y, -z)


def norm(q: Quat) -> float:
    """Return the Euclidean norm sqrt(w^2 + x^2 + y^2 + z^2).

    math.hypot is used rather than a hand-rolled sqrt(sum of squares) because
    hypot avoids overflow/underflow for very large or very small components.
    """
    w, x, y, z = q
    return math.hypot(w, x, y, z)


def _normalize(q: Quat) -> Quat:
    n = norm(q)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    w, x, y, z = q
    return (w / n, x / n, y / n, z / n)


def rotate(q: Quat, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Rotate a 3-vector v by the rotation encoded in unit quaternion q.

    Interpretation fixed by this library:
      * q is normalized internally; the caller does not have to.
      * The rotation is the standard p' = q p q* with p = (0, vx, vy, vz).
        This is active rotation of the vector by the angle/axis stored in q,
        using the right-hand rule: a quaternion (cos(theta/2),
        sin(theta/2)*axis) rotates vectors counter-clockwise when looking
        back along the axis toward the origin.
      * Only the vector part of the result is returned.

    The zero quaternion is rejected because it carries no axis and cannot be
    normalized. A quaternion extremely close to zero but not exactly zero
    will still be normalized; we do not try to detect numeric underflow
    beyond what math.hypot already handles.
    """
    qn = _normalize(q)
    p: Quat = (0.0, v[0], v[1], v[2])
    r = hamilton(hamilton(qn, p), conjugate(qn))
    # The scalar part of r is zero up to floating-point error; we drop it and
    # return only the vector part.
    return (r[1], r[2], r[3])


class Quaternion:
    """A thin object wrapper around the tuple functions.

    Exists for callers who prefer method chaining over free functions. The
    underlying tuple is always stored as plain floats; constructing from ints
    is fine and they are converted automatically by Python's arithmetic.
    """

    __slots__ = ("_q",)

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self._q = (float(w), float(x), float(y), float(z))

    @property
    def components(self) -> Quat:
        return self._q

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(*hamilton(self._q, other._q))

    def conjugate(self) -> "Quaternion":
        return Quaternion(*conjugate(self._q))

    def norm(self) -> float:
        return norm(self._q)

    def rotate(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return rotate(self._q, v)

    def __repr__(self) -> str:
        w, x, y, z = self._q
        return f"Quaternion(w={w!r}, x={x!r}, y={y!r}, z={z!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self._q == other._q
