# quaternion_arithmetic

A small, dependency-free Python library for arithmetic on four-component real
quaternions represented as `(w, x, y, z)` tuples: Hamilton product,
conjugate, norm, and rotation of 3-vectors.

## Usage

```python
from quaternion_arithmetic import hamilton, conjugate, norm, rotate, Quaternion
import math

half = math.sqrt(2.0) / 2.0
q = (half, 0.0, 0.0, half)  # 90-degree rotation about the z axis

# Rotate a vector. q is normalized internally.
print(rotate(q, (1.0, 0.0, 0.0)))  # (0.0, 1.0, 0.0)

# Basic arithmetic.
print(hamilton((1, 0, 0, 0), q))     # q (multiplicative identity)
print(conjugate(q))                   # (half, 0.0, 0.0, -half)
print(norm((0.0, 3.0, 4.0, 0.0)))    # 5.0

# Object-oriented form, if you prefer it.
qobj = Quaternion(half, 0.0, 0.0, half)
print(qobj.rotate((1.0, 0.0, 0.0)))  # (0.0, 1.0, 0.0)
```

Run the tests with:

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Why this exists

The point is to have quaternion rotation arithmetic with zero dependencies
and a fixed, documented convention. Every game-engine or linear-algebra
library bundles its own quaternion code; this one is for projects that need
the four operations above and nothing else, and need to be able to read the
whole implementation in one screen.

The trade-off is explicitness over performance. There is no vectorization,
no SIMD, no batch API. Inputs and outputs are plain tuples so they compose
with the rest of the standard library without conversion.

## Conventions

Quaternion layout is **scalar-first**: `(w, x, y, z)`. Scalar-last
`(x, y, z, w)` is not supported.

Rotation uses the active form `p' = q p q*` with `p = (0, vx, vy, vz)`, and
follows the right-hand rule: a quaternion
`(cos(theta/2), sin(theta/2) * axis)` rotates vectors counter-clockwise
when viewed looking back along the axis toward the origin. Only the vector
part of the result is returned.

## Awkward edges

- `rotate` normalizes its input, so scaling the quaternion does not change
  the rotation. This means passing the zero quaternion `(0, 0, 0, 0)` raises
  `ValueError` rather than returning a zero vector.
- Floating-point error means the scalar part of a rotated vector is not
  exactly zero; it is discarded. Length is preserved up to round-off.
- `norm` uses `math.hypot`, so very large components (around 1e300) do not
  overflow the way a naive `sqrt(w*w + x*x + y*y + z*z)` would.
