import math
import unittest

from quaternion_arithmetic import hamilton, conjugate, norm, rotate, Quaternion


class TestHamilton(unittest.TestCase):

    def test_multiplicative_identity(self):
        q = (2.0, -3.0, 1.0, 4.0)
        ident = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(hamilton(q, ident), q)
        self.assertEqual(hamilton(ident, q), q)

    def test_non_commutative(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        self.assertNotEqual(hamilton(a, b), hamilton(b, a))

    def test_associative(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        c = (9.0, 10.0, 11.0, 12.0)
        self.assertEqual(
            hamilton(hamilton(a, b), c),
            hamilton(a, hamilton(b, c)),
        )

    def test_known_product(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        # Computed by hand from the defining formula.
        expected = (-60.0, 12.0, 30.0, 24.0)
        self.assertEqual(hamilton(a, b), expected)

    def test_ijk_relations(self):
        one = (1.0, 0.0, 0.0, 0.0)
        i = (0.0, 1.0, 0.0, 0.0)
        j = (0.0, 0.0, 1.0, 0.0)
        k = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(hamilton(i, i), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(j, j), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(k, k), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(i, j), k)
        self.assertEqual(hamilton(j, k), i)
        self.assertEqual(hamilton(k, i), j)
        # Anti-commutativity of distinct basis elements.
        self.assertEqual(hamilton(j, i), (-k[0], -k[1], -k[2], -k[3]))


class TestConjugate(unittest.TestCase):

    def test_signs_flipped(self):
        q = (3.0, -1.0, 2.0, -4.0)
        self.assertEqual(conjugate(q), (3.0, 1.0, -2.0, 4.0))

    def test_double_conjugate(self):
        q = (1.5, -2.5, 3.5, -4.5)
        self.assertEqual(conjugate(conjugate(q)), q)

    def test_real_part_unchanged(self):
        q = (7.0, 0.0, 0.0, 0.0)
        self.assertEqual(conjugate(q), q)


class TestNorm(unittest.TestCase):

    def test_unit_quaternion(self):
        self.assertAlmostEqual(norm((1.0, 0.0, 0.0, 0.0)), 1.0)

    def test_pythagorean(self):
        # 3-4-5-... style check with an obvious norm.
        self.assertAlmostEqual(norm((0.0, 3.0, 4.0, 0.0)), 5.0)
        self.assertAlmostEqual(norm((0.0, 0.0, 0.0, 12.0)), 12.0)

    def test_zero(self):
        self.assertEqual(norm((0.0, 0.0, 0.0, 0.0)), 0.0)

    def test_does_not_overflow_on_large_components(self):
        # Components large enough that naive sum-of-squares would overflow a
        # double. math.hypot must keep this finite.
        big = 1e300
        n = norm((big, big, big, big))
        self.assertTrue(math.isfinite(n))
        self.assertAlmostEqual(n, 2.0 * big)


class TestRotate(unittest.TestCase):

    def _almost_equal_vec(self, a, b, tol=1e-9):
        for pa, pb in zip(a, b):
            self.assertAlmostEqual(pa, pb, delta=tol)

    def test_identity_leaves_vector_unchanged(self):
        self._almost_equal_vec(
            rotate((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            (1.0, 2.0, 3.0),
        )

    def test_180_about_y(self):
        # q = (0, 0, 1, 0) is a 180-degree rotation about the y axis.
        # A point on +x should map to -x.
        out = rotate((0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        self._almost_equal_vec(out, (-1.0, 0.0, 0.0))

    def test_90_about_z(self):
        # q = (cos(pi/4), 0, 0, sin(pi/4)) rotates 90 degrees CCW about z.
        half = math.sqrt(2.0) / 2.0
        out = rotate((half, 0.0, 0.0, half), (1.0, 0.0, 0.0))
        self._almost_equal_vec(out, (0.0, 1.0, 0.0))

    def test_normalizes_non_unit_quaternion(self):
        # Scaling a rotation quaternion by 2 must not change the rotation,
        # because the result is divided through by the norm inside rotate.
        q = (math.sqrt(2.0) / 2.0, 0.0, 0.0, math.sqrt(2.0) / 2.0)
        scaled = tuple(2.0 * c for c in q)
        self._almost_equal_vec(
            rotate(q, (1.0, 0.0, 0.0)),
            rotate(scaled, (1.0, 0.0, 0.0)),
        )

    def test_zero_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            rotate((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_rotation_is_invertible_via_conjugate(self):
        half = math.sqrt(2.0) / 2.0
        q = (half, 0.0, 0.0, half)  # 90 deg about z
        v = (3.0, -5.0, 7.0)
        rotated = rotate(q, v)
        restored = rotate(conjugate(q), rotated)
        self._almost_equal_vec(restored, v)

    def test_rotation_preserves_length(self):
        half = math.sqrt(2.0) / 2.0
        q = (half, half, 0.0, 0.0)  # 90 deg about x
        v = (1.0, 2.0, 3.0)
        out = rotate(q, v)
        self.assertAlmostEqual(
            math.sqrt(sum(c * c for c in out)),
            math.sqrt(sum(c * c for c in v)),
            delta=1e-9,
        )


class TestQuaternionClass(unittest.TestCase):

    def test_multiply_chains(self):
        a = Quaternion(1.0, 0.0, 0.0, 0.0)
        b = Quaternion(0.0, 0.0, 1.0, 0.0)
        prod = a * b
        self.assertEqual(prod.components, b.components)

    def test_norm_and_conjugate_methods(self):
        q = Quaternion(0.0, 0.0, 0.0, 3.0)
        self.assertAlmostEqual(q.norm(), 3.0)
        self.assertEqual(q.conjugate().components, (0.0, 0.0, 0.0, -3.0))

    def test_rotate_method_matches_function(self):
        half = math.sqrt(2.0) / 2.0
        q_tuple = (half, 0.0, 0.0, half)
        q_obj = Quaternion(*q_tuple)
        v = (1.0, 0.0, 0.0)
        self.assertEqual(rotate(q_tuple, v), q_obj.rotate(v))

    def test_int_components_become_floats(self):
        q = Quaternion(1, 2, 3, 4)
        for c in q.components:
            self.assertIsInstance(c, float)

    def test_equality(self):
        self.assertEqual(Quaternion(1.0, 2.0, 3.0, 4.0), Quaternion(1.0, 2.0, 3.0, 4.0))
        self.assertNotEqual(Quaternion(1.0, 2.0, 3.0, 4.0), Quaternion(1.0, 2.0, 3.0, 5.0))


if __name__ == "__main__":
    unittest.main()
