//! DCTs with JPEG XL's scaling: the DC is the mean and AC basis functions are
//! scaled by sqrt(2) relative to it, so the inverse is
//! `x[n] = c[0] + sqrt(2) * sum_k c[k] cos(pi k (2n + 1) / 2N)`.
//!
//! Coefficient blocks are kept in natural layout: `c[v * w + u]` holds
//! horizontal frequency `u` and vertical frequency `v`.

/// Forward table for an N-point DCT: t[k * n + i] (reference for tests).
#[cfg(test)]
fn table(n: usize) -> Vec<f32> {
    let mut t = vec![0f32; n * n];
    for k in 0..n {
        for i in 0..n {
            let c = (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
            let s = if k == 0 { 1.0 } else { std::f64::consts::SQRT_2 };
            t[k * n + i] = (c * s / n as f64) as f32;
        }
    }
    t
}

/// Unscaled DCT-II along the first axis of `x`, viewed as `n` rows of `w`
/// values (`X_k = sum_i x_i cos(pi (2i + 1) k / 2N)` for every column), by
/// Lee's recursion. Working on whole rows lets the compiler vectorise.
fn dct_rows(x: &mut [f32], n: usize, w: usize, tmp: &mut [f32], inv_cos: &[Vec<f32>]) {
    if n == 1 {
        return;
    }
    if n == 2 {
        let (a, b) = x.split_at_mut(w);
        for (a, b) in a.iter_mut().zip(b.iter_mut()) {
            let (p, q) = (*a, *b);
            *a = p + q;
            *b = (p - q) * std::f32::consts::FRAC_1_SQRT_2;
        }
        return;
    }
    let h = n / 2;
    let tmp = &mut tmp[..n * w];
    let ic = &inv_cos[n.trailing_zeros() as usize];
    {
        let (te, to) = tmp.split_at_mut(h * w);
        for i in 0..h {
            let k = ic[i];
            let a = &x[i * w..(i + 1) * w];
            let b = &x[(n - 1 - i) * w..(n - i) * w];
            let e = &mut te[i * w..(i + 1) * w];
            let o = &mut to[i * w..(i + 1) * w];
            for j in 0..w {
                e[j] = a[j] + b[j];
                o[j] = (a[j] - b[j]) * k;
            }
        }
        let (xe, xo) = x.split_at_mut(h * w);
        dct_rows(te, h, w, xe, inv_cos);
        dct_rows(to, h, w, xo, inv_cos);
    }
    let (te, to) = tmp.split_at(h * w);
    for m in 0..h {
        x[2 * m * w..(2 * m + 1) * w].copy_from_slice(&te[m * w..(m + 1) * w]);
        let dst = &mut x[(2 * m + 1) * w..(2 * m + 2) * w];
        let o0 = &to[m * w..(m + 1) * w];
        if m + 1 < h {
            let o1 = &to[(m + 1) * w..(m + 2) * w];
            for j in 0..w {
                dst[j] = o0[j] + o1[j];
            }
        } else {
            dst.copy_from_slice(o0);
        }
    }
}

/// Unscaled DCT-III along the first axis (`x_i = sum_k X_k cos(...)`), the
/// transpose of `dct_rows`.
fn idct_rows(x: &mut [f32], n: usize, w: usize, tmp: &mut [f32], inv_cos: &[Vec<f32>]) {
    if n == 1 {
        return;
    }
    if n == 2 {
        let (a, b) = x.split_at_mut(w);
        for (a, b) in a.iter_mut().zip(b.iter_mut()) {
            let (p, q) = (*a, *b * std::f32::consts::FRAC_1_SQRT_2);
            *a = p + q;
            *b = p - q;
        }
        return;
    }
    let h = n / 2;
    let tmp = &mut tmp[..n * w];
    let ic = &inv_cos[n.trailing_zeros() as usize];
    {
        let (te, to) = tmp.split_at_mut(h * w);
        for m in 0..h {
            te[m * w..(m + 1) * w].copy_from_slice(&x[2 * m * w..(2 * m + 1) * w]);
            let o = &mut to[m * w..(m + 1) * w];
            let a = &x[(2 * m + 1) * w..(2 * m + 2) * w];
            if m > 0 {
                let b = &x[(2 * m - 1) * w..(2 * m) * w];
                for j in 0..w {
                    o[j] = a[j] + b[j];
                }
            } else {
                o.copy_from_slice(a);
            }
        }
        let (xe, xo) = x.split_at_mut(h * w);
        idct_rows(te, h, w, xe, inv_cos);
        idct_rows(to, h, w, xo, inv_cos);
    }
    let (te, to) = tmp.split_at(h * w);
    for i in 0..h {
        let k = ic[i];
        let (e, o) = (&te[i * w..(i + 1) * w], &to[i * w..(i + 1) * w]);
        for j in 0..w {
            let v = o[j] * k;
            x[i * w + j] = e[j] + v;
            x[(n - 1 - i) * w + j] = e[j] - v;
        }
    }
}

type V8 = [f32; 8];

#[inline(always)]
fn vadd(a: &V8, b: &V8) -> V8 {
    std::array::from_fn(|j| a[j] + b[j])
}
#[inline(always)]
fn vsub_mul(a: &V8, b: &V8, k: f32) -> V8 {
    std::array::from_fn(|j| (a[j] - b[j]) * k)
}
#[inline(always)]
fn vmul(a: &V8, k: f32) -> V8 {
    std::array::from_fn(|j| a[j] * k)
}

/// 8-point DCT along the rows of an 8x8 block (eight columns at once), the
/// same operations in the same order as `dct_rows` with n = w = 8.
#[inline(always)]
fn dct8_rows(x: &mut [V8; 8], ic8: &[f32], ic4: &[f32]) {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let e = [vadd(&x[0], &x[7]), vadd(&x[1], &x[6]), vadd(&x[2], &x[5]), vadd(&x[3], &x[4])];
    let o = [
        vsub_mul(&x[0], &x[7], ic8[0]),
        vsub_mul(&x[1], &x[6], ic8[1]),
        vsub_mul(&x[2], &x[5], ic8[2]),
        vsub_mul(&x[3], &x[4], ic8[3]),
    ];
    let dct4 = |t: &[V8; 4]| -> [V8; 4] {
        let e = [vadd(&t[0], &t[3]), vadd(&t[1], &t[2])];
        let o = [vsub_mul(&t[0], &t[3], ic4[0]), vsub_mul(&t[1], &t[2], ic4[1])];
        let ee = [vadd(&e[0], &e[1]), vsub_mul(&e[0], &e[1], S)];
        let oo = [vadd(&o[0], &o[1]), vsub_mul(&o[0], &o[1], S)];
        [ee[0], vadd(&oo[0], &oo[1]), ee[1], oo[1]]
    };
    let (ee, oo) = (dct4(&e), dct4(&o));
    x[0] = ee[0];
    x[1] = vadd(&oo[0], &oo[1]);
    x[2] = ee[1];
    x[3] = vadd(&oo[1], &oo[2]);
    x[4] = ee[2];
    x[5] = vadd(&oo[2], &oo[3]);
    x[6] = ee[3];
    x[7] = oo[3];
}

/// Transpose of `dct8_rows` (see `idct_rows`).
#[inline(always)]
fn idct8_rows(x: &mut [V8; 8], ic8: &[f32], ic4: &[f32]) {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let idct4 = |t: &[V8; 4]| -> [V8; 4] {
        let e = [t[0], t[2]];
        let o = [t[1], vadd(&t[3], &t[1])];
        let e2 = [vadd(&e[0], &vmul(&e[1], S)), std::array::from_fn(|j| e[0][j] - e[1][j] * S)];
        let o2 = [vadd(&o[0], &vmul(&o[1], S)), std::array::from_fn(|j| o[0][j] - o[1][j] * S)];
        let f = |i: usize| -> (V8, V8) {
            let v = vmul(&o2[i], ic4[i]);
            (vadd(&e2[i], &v), std::array::from_fn(|j| e2[i][j] - v[j]))
        };
        let (a0, a3) = f(0);
        let (a1, a2) = f(1);
        [a0, a1, a2, a3]
    };
    let e = [x[0], x[2], x[4], x[6]];
    let o = [x[1], vadd(&x[3], &x[1]), vadd(&x[5], &x[3]), vadd(&x[7], &x[5])];
    let (e, o) = (idct4(&e), idct4(&o));
    for i in 0..4 {
        let v = vmul(&o[i], ic8[i]);
        x[i] = vadd(&e[i], &v);
        x[7 - i] = std::array::from_fn(|j| e[i][j] - v[j]);
    }
}

#[inline(always)]
fn transpose8(a: &V8x8) -> V8x8 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
type V8x8 = [V8; 8];

fn transpose(src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    for y in 0..h {
        for x in 0..w {
            dst[x * h + y] = src[y * w + x];
        }
    }
}

// ---------------------------------------------------------------------------
// Width-specialised kernels.
//
// The same operations in the same order as `dct_rows` / `idct_rows`, but with
// the row width `W` and the transform length fixed at compile time (one
// function per recursion level, so everything inlines into one body). With
// `W` constant every inner loop is a whole number of vectors, which the
// compiler vectorises with AVX2 when the body is instantiated under
// `#[target_feature(enable = "avx2")]`. FMA is deliberately not enabled, so
// results are bit-identical to the generic path.

#[inline(always)]
fn dct_rows_2<const W: usize>(x: &mut [f32], _tmp: &mut [f32], _ic: &[Vec<f32>]) {
    let (a, b) = x.split_at_mut(W);
    for j in 0..W {
        let (p, q) = (a[j], b[j]);
        a[j] = p + q;
        b[j] = (p - q) * std::f32::consts::FRAC_1_SQRT_2;
    }
}

#[inline(always)]
fn idct_rows_2<const W: usize>(x: &mut [f32], _tmp: &mut [f32], _ic: &[Vec<f32>]) {
    let (a, b) = x.split_at_mut(W);
    for j in 0..W {
        let (p, q) = (a[j], b[j] * std::f32::consts::FRAC_1_SQRT_2);
        a[j] = p + q;
        b[j] = p - q;
    }
}

macro_rules! rows_level {
    ($d:ident, $i:ident, $hd:ident, $hi:ident, $n:literal, $lvl:literal) => {
        #[inline(always)]
        fn $d<const W: usize>(x: &mut [f32], tmp: &mut [f32], inv_cos: &[Vec<f32>]) {
            const H: usize = $n / 2;
            let tmp = &mut tmp[..$n * W];
            let ic = &inv_cos[$lvl][..];
            {
                let (te, to) = tmp.split_at_mut(H * W);
                for i in 0..H {
                    let k = ic[i];
                    let a = &x[i * W..(i + 1) * W];
                    let b = &x[($n - 1 - i) * W..($n - i) * W];
                    let e = &mut te[i * W..(i + 1) * W];
                    let o = &mut to[i * W..(i + 1) * W];
                    for j in 0..W {
                        e[j] = a[j] + b[j];
                        o[j] = (a[j] - b[j]) * k;
                    }
                }
                let (xe, xo) = x.split_at_mut(H * W);
                $hd::<W>(te, xe, inv_cos);
                $hd::<W>(to, xo, inv_cos);
            }
            let (te, to) = tmp.split_at(H * W);
            for m in 0..H {
                x[2 * m * W..(2 * m + 1) * W].copy_from_slice(&te[m * W..(m + 1) * W]);
                let dst = &mut x[(2 * m + 1) * W..(2 * m + 2) * W];
                let o0 = &to[m * W..(m + 1) * W];
                if m + 1 < H {
                    let o1 = &to[(m + 1) * W..(m + 2) * W];
                    for j in 0..W {
                        dst[j] = o0[j] + o1[j];
                    }
                } else {
                    dst.copy_from_slice(o0);
                }
            }
        }

        #[inline(always)]
        fn $i<const W: usize>(x: &mut [f32], tmp: &mut [f32], inv_cos: &[Vec<f32>]) {
            const H: usize = $n / 2;
            let tmp = &mut tmp[..$n * W];
            let ic = &inv_cos[$lvl][..];
            {
                let (te, to) = tmp.split_at_mut(H * W);
                for m in 0..H {
                    te[m * W..(m + 1) * W].copy_from_slice(&x[2 * m * W..(2 * m + 1) * W]);
                    let o = &mut to[m * W..(m + 1) * W];
                    let a = &x[(2 * m + 1) * W..(2 * m + 2) * W];
                    if m > 0 {
                        let b = &x[(2 * m - 1) * W..(2 * m) * W];
                        for j in 0..W {
                            o[j] = a[j] + b[j];
                        }
                    } else {
                        o.copy_from_slice(a);
                    }
                }
                let (xe, xo) = x.split_at_mut(H * W);
                $hi::<W>(te, xe, inv_cos);
                $hi::<W>(to, xo, inv_cos);
            }
            let (te, to) = tmp.split_at(H * W);
            for i in 0..H {
                let k = ic[i];
                let (e, o) = (&te[i * W..(i + 1) * W], &to[i * W..(i + 1) * W]);
                for j in 0..W {
                    let v = o[j] * k;
                    x[i * W + j] = e[j] + v;
                    x[($n - 1 - i) * W + j] = e[j] - v;
                }
            }
        }
    };
}

rows_level!(dct_rows_4, idct_rows_4, dct_rows_2, idct_rows_2, 4, 2);
rows_level!(dct_rows_8, idct_rows_8, dct_rows_4, idct_rows_4, 8, 3);
rows_level!(dct_rows_16, idct_rows_16, dct_rows_8, idct_rows_8, 16, 4);
rows_level!(dct_rows_32, idct_rows_32, dct_rows_16, idct_rows_16, 32, 5);
rows_level!(dct_rows_64, idct_rows_64, dct_rows_32, idct_rows_32, 64, 6);

#[inline(always)]
fn dct_rows_n<const W: usize>(n: usize, x: &mut [f32], tmp: &mut [f32], ic: &[Vec<f32>]) {
    match n {
        8 => dct_rows_8::<W>(x, tmp, ic),
        16 => dct_rows_16::<W>(x, tmp, ic),
        32 => dct_rows_32::<W>(x, tmp, ic),
        64 => dct_rows_64::<W>(x, tmp, ic),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn idct_rows_n<const W: usize>(n: usize, x: &mut [f32], tmp: &mut [f32], ic: &[Vec<f32>]) {
    match n {
        8 => idct_rows_8::<W>(x, tmp, ic),
        16 => idct_rows_16::<W>(x, tmp, ic),
        32 => idct_rows_32::<W>(x, tmp, ic),
        64 => idct_rows_64::<W>(x, tmp, ic),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn transpose_c<const W: usize, const H: usize>(src: &[f32], dst: &mut [f32]) {
    for y in 0..H {
        for x in 0..W {
            dst[x * H + y] = src[y * W + x];
        }
    }
}

/// `forward_sized` for a `W` x `H` block (`N = W * H`), widths and heights
/// in {8, 16, 32, 64}.
#[inline(always)]
fn forward_wh<const W: usize, const H: usize, const N: usize>(
    inv_cos: &[Vec<f32>],
    src: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    let mut a = [0f32; N];
    let mut tmp = [0f32; N];
    for y in 0..H {
        a[y * W..(y + 1) * W].copy_from_slice(&src[y * stride..y * stride + W]);
    }
    dct_rows_n::<W>(H, &mut a, &mut tmp, inv_cos);
    transpose_c::<W, H>(&a, &mut out[..N]);
    dct_rows_n::<H>(W, &mut out[..N], &mut tmp, inv_cos);
    transpose_c::<H, W>(&out[..N], &mut a);
    let (sw, sh) = (std::f32::consts::SQRT_2 / W as f32, std::f32::consts::SQRT_2 / H as f32);
    let (dw, dh) = (1.0 / W as f32, 1.0 / H as f32);
    for v in 0..H {
        let fv = if v == 0 { dh } else { sh };
        for u in 0..W {
            let fu = if u == 0 { dw } else { sw };
            out[v * W + u] = a[v * W + u] * fu * fv;
        }
    }
}

/// `inverse_sized` for a `W` x `H` block.
#[inline(always)]
fn inverse_wh<const W: usize, const H: usize, const N: usize>(inv_cos: &[Vec<f32>], coeffs: &[f32], out: &mut [f32]) {
    let mut a = [0f32; N];
    let mut tmp = [0f32; N];
    let s = std::f32::consts::SQRT_2;
    for v in 0..H {
        let fv = if v == 0 { 1.0 } else { s };
        for u in 0..W {
            let fu = if u == 0 { 1.0 } else { s };
            a[v * W + u] = coeffs[v * W + u] * fu * fv;
        }
    }
    idct_rows_n::<W>(H, &mut a, &mut tmp, inv_cos);
    transpose_c::<W, H>(&a, &mut out[..N]);
    idct_rows_n::<H>(W, &mut out[..N], &mut tmp, inv_cos);
    transpose_c::<H, W>(&out[..N], &mut a);
    out[..N].copy_from_slice(&a);
}

/// Instantiates the kernels for one size, as a plain and an AVX2 version.
macro_rules! sized_kernels {
    ($(($w:literal, $h:literal)),*) => {
        #[inline(never)]
        fn forward_plain(ic: &[Vec<f32>], src: &[f32], stride: usize, w: usize, h: usize, out: &mut [f32]) -> bool {
            match (w, h) {
                $(($w, $h) => forward_wh::<$w, $h, { $w * $h }>(ic, src, stride, out),)*
                _ => return false,
            }
            true
        }
        #[inline(never)]
        fn inverse_plain(ic: &[Vec<f32>], c: &[f32], w: usize, h: usize, out: &mut [f32]) -> bool {
            match (w, h) {
                $(($w, $h) => inverse_wh::<$w, $h, { $w * $h }>(ic, c, out),)*
                _ => return false,
            }
            true
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn forward_avx2(ic: &[Vec<f32>], src: &[f32], stride: usize, w: usize, h: usize, out: &mut [f32]) -> bool {
            match (w, h) {
                $(($w, $h) => forward_wh::<$w, $h, { $w * $h }>(ic, src, stride, out),)*
                _ => return false,
            }
            true
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn inverse_avx2(ic: &[Vec<f32>], c: &[f32], w: usize, h: usize, out: &mut [f32]) -> bool {
            match (w, h) {
                $(($w, $h) => inverse_wh::<$w, $h, { $w * $h }>(ic, c, out),)*
                _ => return false,
            }
            true
        }
    };
}

sized_kernels!((16, 16), (32, 32), (64, 64), (8, 16), (16, 8), (32, 16), (16, 32), (64, 32), (32, 64));

pub struct Dct {
    /// inv_cos[log2 n][i] = 1 / (2 cos(pi (2i + 1) / 2n))
    inv_cos: Vec<Vec<f32>>,
    avx2: bool,
}

impl Dct {
    pub fn has_avx2(&self) -> bool {
        self.avx2
    }

    pub fn new() -> Self {
        let inv_cos = (0..=6)
            .map(|l| {
                let n = 1usize << l;
                (0..n / 2)
                    .map(|i| (0.5 / (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * n) as f64).cos()) as f32)
                    .collect()
            })
            .collect();
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Dct { inv_cos, avx2 }
    }

    /// 2D forward DCT of a `w` x `h` block read from `src` with row stride
    /// `stride`, into `out[..w * h]`.
    pub fn forward(&self, src: &[f32], stride: usize, w: usize, h: usize, out: &mut [f32]) {
        if w == 8 && h == 8 {
            return self.forward8(src, stride, out);
        }
        #[cfg(target_arch = "x86_64")]
        if self.avx2 && unsafe { forward_avx2(&self.inv_cos, src, stride, w, h, out) } {
            return;
        }
        if forward_plain(&self.inv_cos, src, stride, w, h, out) {
            return;
        }
        // Match scratch size to the transform, including small LF transforms.
        match w * h {
            1 => self.forward_sized::<1>(src, stride, w, h, out),
            2 => self.forward_sized::<2>(src, stride, w, h, out),
            4 => self.forward_sized::<4>(src, stride, w, h, out),
            8 => self.forward_sized::<8>(src, stride, w, h, out),
            16 => self.forward_sized::<16>(src, stride, w, h, out),
            32 => self.forward_sized::<32>(src, stride, w, h, out),
            64 => self.forward_sized::<64>(src, stride, w, h, out),
            128 => self.forward_sized::<128>(src, stride, w, h, out),
            256 => self.forward_sized::<256>(src, stride, w, h, out),
            512 => self.forward_sized::<512>(src, stride, w, h, out),
            1024 => self.forward_sized::<1024>(src, stride, w, h, out),
            2048 => self.forward_sized::<2048>(src, stride, w, h, out),
            4096 => self.forward_sized::<4096>(src, stride, w, h, out),
            _ => unreachable!(),
        }
    }

    fn forward8(&self, src: &[f32], stride: usize, out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            return unsafe { self.forward8_avx2(src, stride, out) };
        }
        self.forward8_impl(src, stride, out)
    }

    fn inverse8(&self, coeffs: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            return unsafe { self.inverse8_avx2(coeffs, out) };
        }
        self.inverse8_impl(coeffs, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn forward8_avx2(&self, src: &[f32], stride: usize, out: &mut [f32]) {
        self.forward8_impl(src, stride, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn inverse8_avx2(&self, coeffs: &[f32], out: &mut [f32]) {
        self.inverse8_impl(coeffs, out)
    }

    /// 8x8 forward DCT, unrolled; identical results to `forward_sized`.
    #[inline(always)]
    fn forward8_impl(&self, src: &[f32], stride: usize, out: &mut [f32]) {
        let (ic8, ic4) = (&self.inv_cos[3][..], &self.inv_cos[2][..]);
        let mut a: V8x8 = std::array::from_fn(|y| std::array::from_fn(|x| src[y * stride + x]));
        dct8_rows(&mut a, ic8, ic4);
        let mut t = transpose8(&a);
        dct8_rows(&mut t, ic8, ic4);
        let a = transpose8(&t);
        let (sw, dw) = (std::f32::consts::SQRT_2 / 8.0, 1.0 / 8.0f32);
        for v in 0..8 {
            let fv = if v == 0 { dw } else { sw };
            for u in 0..8 {
                let fu = if u == 0 { dw } else { sw };
                out[v * 8 + u] = a[v][u] * fu * fv;
            }
        }
    }

    /// 8x8 inverse DCT, unrolled; identical results to `inverse_sized`.
    #[inline(always)]
    fn inverse8_impl(&self, coeffs: &[f32], out: &mut [f32]) {
        let (ic8, ic4) = (&self.inv_cos[3][..], &self.inv_cos[2][..]);
        let s = std::f32::consts::SQRT_2;
        let mut a: V8x8 = std::array::from_fn(|v| {
            std::array::from_fn(|u| {
                let fv = if v == 0 { 1.0 } else { s };
                let fu = if u == 0 { 1.0 } else { s };
                coeffs[v * 8 + u] * fu * fv
            })
        });
        idct8_rows(&mut a, ic8, ic4);
        let mut t = transpose8(&a);
        idct8_rows(&mut t, ic8, ic4);
        let a = transpose8(&t);
        for v in 0..8 {
            out[v * 8..v * 8 + 8].copy_from_slice(&a[v]);
        }
    }

    fn forward_sized<const N: usize>(&self, src: &[f32], stride: usize, w: usize, h: usize, out: &mut [f32]) {
        let mut a = [0f32; N];
        let mut tmp = [0f32; N];
        let n = w * h;
        for y in 0..h {
            a[y * w..(y + 1) * w].copy_from_slice(&src[y * stride..y * stride + w]);
        }
        dct_rows(&mut a[..n], h, w, &mut tmp, &self.inv_cos);
        transpose(&a[..n], w, h, &mut out[..n]);
        dct_rows(&mut out[..n], w, h, &mut tmp, &self.inv_cos);
        transpose(&out[..n], h, w, &mut a[..n]);
        let (sw, sh) = (std::f32::consts::SQRT_2 / w as f32, std::f32::consts::SQRT_2 / h as f32);
        let (dw, dh) = (1.0 / w as f32, 1.0 / h as f32);
        for v in 0..h {
            let fv = if v == 0 { dh } else { sh };
            for u in 0..w {
                let fu = if u == 0 { dw } else { sw };
                out[v * w + u] = a[v * w + u] * fu * fv;
            }
        }
    }

    /// 2D inverse DCT of `coeffs[..w * h]` into `out[..w * h]` (row stride w).
    pub fn inverse(&self, coeffs: &[f32], w: usize, h: usize, out: &mut [f32]) {
        if w == 8 && h == 8 {
            return self.inverse8(coeffs, out);
        }
        #[cfg(target_arch = "x86_64")]
        if self.avx2 && unsafe { inverse_avx2(&self.inv_cos, coeffs, w, h, out) } {
            return;
        }
        if inverse_plain(&self.inv_cos, coeffs, w, h, out) {
            return;
        }
        // Match scratch size to the transform, including small LF transforms.
        match w * h {
            1 => self.inverse_sized::<1>(coeffs, w, h, out),
            2 => self.inverse_sized::<2>(coeffs, w, h, out),
            4 => self.inverse_sized::<4>(coeffs, w, h, out),
            8 => self.inverse_sized::<8>(coeffs, w, h, out),
            16 => self.inverse_sized::<16>(coeffs, w, h, out),
            32 => self.inverse_sized::<32>(coeffs, w, h, out),
            64 => self.inverse_sized::<64>(coeffs, w, h, out),
            128 => self.inverse_sized::<128>(coeffs, w, h, out),
            256 => self.inverse_sized::<256>(coeffs, w, h, out),
            512 => self.inverse_sized::<512>(coeffs, w, h, out),
            1024 => self.inverse_sized::<1024>(coeffs, w, h, out),
            2048 => self.inverse_sized::<2048>(coeffs, w, h, out),
            4096 => self.inverse_sized::<4096>(coeffs, w, h, out),
            _ => unreachable!(),
        }
    }

    fn inverse_sized<const N: usize>(&self, coeffs: &[f32], w: usize, h: usize, out: &mut [f32]) {
        let mut a = [0f32; N];
        let mut tmp = [0f32; N];
        let n = w * h;
        let s = std::f32::consts::SQRT_2;
        for v in 0..h {
            let fv = if v == 0 { 1.0 } else { s };
            for u in 0..w {
                let fu = if u == 0 { 1.0 } else { s };
                a[v * w + u] = coeffs[v * w + u] * fu * fv;
            }
        }
        idct_rows(&mut a[..n], h, w, &mut tmp, &self.inv_cos);
        transpose(&a[..n], w, h, &mut out[..n]);
        idct_rows(&mut out[..n], w, h, &mut tmp, &self.inv_cos);
        transpose(&out[..n], h, w, &mut a[..n]);
        out[..n].copy_from_slice(&a[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference() {
        let d = Dct::new();
        for n in [2usize, 4, 8, 16, 32, 64] {
            let t = table(n);
            let src: Vec<f32> = (0..n).map(|i| ((i * 29 % 17) as f32) / 7.0 - 1.0).collect();
            let mut c = vec![0f32; n];
            d.forward(&src, n, n, 1, &mut c);
            for k in 0..n {
                let r: f32 = (0..n).map(|i| src[i] * t[k * n + i]).sum();
                assert!((r - c[k]).abs() < 1e-4, "n={n} k={k}: {r} vs {}", c[k]);
            }
        }
    }

    #[test]
    fn unrolled_8x8_matches_generic() {
        let d = Dct::new();
        let src: Vec<f32> = (0..64).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
        let (mut a, mut b) = (vec![0f32; 64], vec![0f32; 64]);
        d.forward8(&src, 8, &mut a);
        d.forward_sized::<64>(&src, 8, 8, 8, &mut b);
        assert_eq!(a, b);
        let (mut c, mut e) = (vec![0f32; 64], vec![0f32; 64]);
        d.inverse8(&a, &mut c);
        d.inverse_sized::<64>(&a, 8, 8, &mut e);
        assert_eq!(c, e);
    }

    #[test]
    fn specialised_matches_generic() {
        let d = Dct::new();
        for (w, h) in [(16, 16), (32, 32), (64, 64), (8, 16), (16, 8), (32, 16), (16, 32), (64, 32), (32, 64)] {
            let src: Vec<f32> = (0..w * h).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
            let (mut a, mut b) = (vec![0f32; 4096], vec![0f32; 4096]);
            d.forward(&src, w, w, h, &mut a);
            d.forward_sized::<4096>(&src, w, w, h, &mut b);
            assert_eq!(a[..w * h], b[..w * h], "forward {w}x{h}");
            let (mut c, mut e) = (vec![0f32; 4096], vec![0f32; 4096]);
            d.inverse(&a, w, h, &mut c);
            d.inverse_sized::<4096>(&a, w, h, &mut e);
            assert_eq!(c[..w * h], e[..w * h], "inverse {w}x{h}");
        }
    }

    #[test]
    fn roundtrip() {
        let d = Dct::new();
        let sizes = [1, 2, 4, 8, 16, 32, 64];
        for (w, h) in sizes.into_iter().flat_map(|w| sizes.map(|h| (w, h))) {
            let src: Vec<f32> = (0..w * h).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
            let mut c = vec![0f32; w * h];
            let mut back = vec![0f32; w * h];
            d.forward(&src, w, w, h, &mut c);
            let mean: f32 = src.iter().sum::<f32>() / (w * h) as f32;
            assert!((c[0] - mean).abs() < 1e-4);
            d.inverse(&c, w, h, &mut back);
            for (a, b) in src.iter().zip(&back) {
                assert!((a - b).abs() < 1e-3, "{w}x{h}: {a} vs {b}");
            }
        }
    }
}
