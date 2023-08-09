use nalgebra::{Matrix3, RealField, Vector3, Vector6};
use std::ops::{Add, AddAssign};

/// 3-dimensional quadratic error function
///
/// ```text
/// Error[x] = (Ax - b)^T (Ax - b)
///          = x^T A^T Ax - 2x^T A^T b + b^T b
/// ```
///
/// The main features are calculating this error and finding `x` to minimize the
/// error. Special care must be taken to handle the cases where `A^T A` is
/// singular.
///
/// This struct stores a compact representation (10 scalars). 6 for the upper
/// triangle of symmetric `A^T A`, 3 for `A^T b` and 1 for `b^T b`.
#[derive(Clone, Debug)]
pub struct Qef3<T> {
    ata: Vector6<T>,
    atb: Vector3<T>,
    btb: T,
}

impl<T: RealField> Default for Qef3<T> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<T: RealField> PartialEq for Qef3<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ata.eq(&other.ata) && self.atb.eq(&other.atb) && self.btb.eq(&other.btb)
    }
}

impl<T: RealField> Qef3<T> {
    /// Construct directly from compact representation.
    pub fn new(ata: Vector6<T>, atb: Vector3<T>, btb: T) -> Self {
        Self { ata, atb, btb }
    }

    pub fn zeros() -> Self {
        Self {
            ata: Vector6::zeros(),
            atb: Vector3::zeros(),
            btb: T::zero(),
        }
    }
}

impl<T: Copy + RealField> Qef3<T> {
    /// Uses `A` and `b` to construct the compact representation:
    ///
    /// - upper triangle of `A^T A`
    /// - `A^T b`
    /// - `b^T b`
    pub fn from_matrix(a: Matrix3<T>, b: Vector3<T>) -> Self {
        let at = a.transpose();
        let ata = at * a; // symmetric
        let atb = at * b;
        let btb = b.dot(&b);
        Self::new(upper_triangle(&ata), atb, btb)
    }

    /// `A^T A`
    #[rustfmt::skip]
    pub fn ata(&self) -> Matrix3<T> {
        Matrix3::new(
            self.ata[0], self.ata[1], self.ata[2],
            self.ata[1], self.ata[3], self.ata[4],
            self.ata[2], self.ata[4], self.ata[5]
        )
    }

    /// Applies a translation by `t` such that:
    ///
    /// ```text
    /// Error'[x] = Error[x + t]
    /// ```
    ///
    /// This corresponds to translating `b` becomes `b + At`, which means:
    ///
    /// - `A^T b` becomes `A^T (b + At) = A^T b + A^T At`
    /// - `b^T b` becomes `(b + At)^T (b + At) = b^T b + 2t^T A^T b + t^T A^T At`
    ///
    /// This may be useful when adding two (or more) QEFs that used different
    /// coordinate origins.
    pub fn translate(&mut self, t: Vector3<T>) {
        let ata = self.ata();
        // Order matters here, since we use atb before re-assignment.
        let ttatb = t.dot(&self.atb);
        self.btb += ttatb + ttatb + t.dot(&(ata * t));
        self.atb += ata * t;
    }

    /// `(Ax - b)^T (Ax - b)`
    pub fn error(&self, x: Vector3<T>) -> T {
        // x^T A^T Ax
        let xtatax = x.dot(&(self.ata() * x));
        // x^T A^T b
        let xtatb = x.dot(&self.atb);
        // x^T A^T Ax - 2x^T A^T b + b^T b
        xtatax - (xtatb + xtatb) + self.btb
    }

    /// Assumes `A^T A` is invertible and solves for `x` minimizing error:
    ///
    /// ```text
    /// inverse(A^T A) A^T b
    /// ```
    ///
    /// This is guaranteed to work for "probabilistic quadrics".
    ///
    /// # Panics
    ///
    /// If `det(A^T A) = 0`.
    pub fn minimizer_with_exact_inverse(&self) -> Vector3<T> {
        self.ata().try_inverse().unwrap() * self.atb
    }

    /// Makes no assumptions about whether `A^T A` is invertible and solves for
    /// `x` minimizing error.
    ///
    /// This uses Singular Value Decomposition (SVD) and kills any eigenvalues
    /// less than `eps` by setting them to 0.
    ///
    /// # Panics
    ///
    /// If `eps < 0`.
    pub fn minimizer_with_pseudo_inverse(&self, eps: T) -> Vector3<T> {
        self.ata().pseudo_inverse(eps).unwrap() * self.atb
    }

    /// This QEF represents the distance from a point `x` to the plane with
    /// origin `o` and normal vector `n`, assuming `n` is a unit vector.
    pub fn plane(o: Vector3<T>, n: Vector3<T>) -> Self {
        let ata = upper_triangle_of_outer_product(n);
        let pn = o.dot(&n);
        let atb = n * pn;
        let btb = pn * pn;
        Self::new(ata, atb, btb)
    }

    /// Constructs the probabilistic QEF from "Fast and Robust QEF Minimization
    /// using Probabilistic Quadrics" by Trettner and Kobbelt.
    ///
    /// - `mean_o`: Sample mean plane origin.
    /// - `mean_n`: Sample mean plane normal.
    /// - `stddev_o`: Standard deviation of plane origin (isometric noise).
    /// - `stddev_n`: Standard deviation of plane normal (isometric noise).
    pub fn probabilistic_plane(
        mean_o: Vector3<T>,
        mean_n: Vector3<T>,
        stddev_o: T,
        stddev_n: T,
    ) -> Self {
        // Ported from https://github.com/Philip-Trettner/probabilistic-quadrics

        // Variances (Var(p) and Var(n)) from standard deviations.
        let sp2 = stddev_o * stddev_o;
        let sn2 = stddev_n * stddev_n;

        // E[A^T A] = nn^T + Var(n)
        let mut ata = upper_triangle_of_outer_product(mean_n);
        add_to_diagonal(&mut ata, sn2);

        // E[A^T b] = nn^T p + Var(n) p
        let d = mean_o.dot(&mean_n);
        let atb = mean_n * d + mean_o * sn2;

        // E[b^T b] = p^T nn^T p + p^T Var(n) p + n^T Var(p) n + Tr[Var(n) Var(q)]
        let sp2sn2 = sp2 * sn2;
        let btb = d * d
            + sn2 * mean_o.dot(&mean_o)
            + sp2 * mean_n.dot(&mean_n)
            + (sp2sn2 + sp2sn2 + sp2sn2);

        Self::new(ata, atb, btb)
    }
}

impl<T: RealField> Add for Qef3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            ata: self.ata + rhs.ata,
            atb: self.atb + rhs.atb,
            btb: self.btb + rhs.btb,
        }
    }
}

impl<T: RealField> AddAssign for Qef3<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.ata += rhs.ata;
        self.atb += rhs.atb;
        self.btb += rhs.btb;
    }
}

fn upper_triangle<T: Copy + RealField>(m: &Matrix3<T>) -> Vector6<T> {
    Vector6::new(
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 2)],
    )
}

fn upper_triangle_of_outer_product<T: Copy + RealField>(n: Vector3<T>) -> Vector6<T> {
    Vector6::new(
        n.x * n.x,
        n.x * n.y,
        n.x * n.z,
        n.y * n.y,
        n.y * n.z,
        n.z * n.z,
    )
}

fn add_to_diagonal<T: Copy + RealField>(upper_tri: &mut Vector6<T>, rhs: T) {
    upper_tri[0] += rhs;
    upper_tri[3] += rhs;
    upper_tri[5] += rhs;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn simple_three_plane_system() -> ([Vector3<f32>; 3], [Vector3<f32>; 3]) {
        let origins = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        (origins, normals)
    }

    #[test]
    fn three_planes_sanity() {
        // This fully-determined system should produce nearly the same results
        // for all methods.
        let (origins, normals) = simple_three_plane_system();

        let mut qef = Qef3::default();
        let mut prob_qef = Qef3::default();
        for (p, n) in origins.into_iter().zip(normals) {
            qef += Qef3::plane(p, n);
            prob_qef += Qef3::probabilistic_plane(p, n, 0.0001, 0.0001);
        }

        let expected = Vector3::new(1.0, 1.0, 1.0);
        assert_relative_eq!(qef.minimizer_with_exact_inverse(), expected);
        assert_relative_eq!(qef.minimizer_with_pseudo_inverse(0.01), expected);
        assert_relative_eq!(prob_qef.minimizer_with_exact_inverse(), expected);
        assert_relative_eq!(prob_qef.minimizer_with_pseudo_inverse(0.01), expected);

        assert_relative_eq!(0.0, qef.error(expected));
        assert_relative_eq!(0.0, prob_qef.error(expected));
    }

    #[test]
    fn translate_sanity() {
        let (origins, normals) = simple_three_plane_system();

        let mut qef = Qef3::default();
        let mut prob_qef = Qef3::default();
        for (p, n) in origins.into_iter().zip(normals) {
            qef += Qef3::plane(p, n);
            prob_qef += Qef3::probabilistic_plane(p, n, 0.0001, 0.0001);
        }

        let translation = Vector3::new(1.0, 2.0, 3.0);
        qef.translate(translation);
        prob_qef.translate(translation);

        let expected = Vector3::new(2.0, 3.0, 4.0);
        assert_relative_eq!(qef.minimizer_with_exact_inverse(), expected);
        assert_relative_eq!(qef.minimizer_with_pseudo_inverse(0.01), expected);
        assert_relative_eq!(prob_qef.minimizer_with_exact_inverse(), expected);
        assert_relative_eq!(prob_qef.minimizer_with_pseudo_inverse(0.01), expected);

        assert_relative_eq!(0.0, qef.error(expected));
        assert_relative_eq!(0.0, prob_qef.error(expected));
    }
}
