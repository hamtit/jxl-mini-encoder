//! The 8x8 transforms other than the plain DCT: Identity, DCT2x2, DCT4x4,
//! DCT4x8 and DCT8x4. They follow libjxl's coefficient layout exactly: the
//! result is an 8x8 array whose entry 0 is the block mean (the LF value).
//!
//! The DCT scaling is the same as in `dct.rs` (DC is the mean, AC basis
//! functions carry a factor of sqrt(2)).

/// `0.5 / cos(pi * (2i + 1) / (2N))` for Lee's 4-point and 8-point butterflies.
const IC4: [f32; 2] = [0.541_196_1, 1.306_563];
const IC8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_6];

#[inline(always)]
fn dct4_raw(x: [f32; 4]) -> [f32; 4] {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let e0 = x[0] + x[3];
    let e1 = x[1] + x[2];
    let o0 = (x[0] - x[3]) * IC4[0];
    let o1 = (x[1] - x[2]) * IC4[1];
    let ee0 = e0 + e1;
    let ee1 = (e0 - e1) * S;
    let oo0 = o0 + o1;
    let oo1 = (o0 - o1) * S;
    [ee0, oo0 + oo1, ee1, oo1]
}

#[inline(always)]
fn dct4_1d(x: [f32; 4]) -> [f32; 4] {
    const D: f32 = 0.25;
    const S: f32 = std::f32::consts::SQRT_2 * 0.25;
    let r = dct4_raw(x);
    [r[0] * D, r[1] * S, r[2] * S, r[3] * S]
}

#[inline(always)]
fn dct8_1d(x: [f32; 8]) -> [f32; 8] {
    const D: f32 = 0.125;
    const S: f32 = std::f32::consts::SQRT_2 * 0.125;
    let e = dct4_raw([x[0] + x[7], x[1] + x[6], x[2] + x[5], x[3] + x[4]]);
    let o = dct4_raw([(x[0] - x[7]) * IC8[0], (x[1] - x[6]) * IC8[1], (x[2] - x[5]) * IC8[2], (x[3] - x[4]) * IC8[3]]);
    [e[0] * D, (o[0] + o[1]) * S, e[1] * S, (o[1] + o[2]) * S, e[2] * S, (o[2] + o[3]) * S, e[3] * S, o[3] * S]
}

#[inline(always)]
fn idct4_raw(x: [f32; 4]) -> [f32; 4] {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let e0 = x[0];
    let e1 = x[2] * S;
    let o0 = x[1];
    let o1 = (x[3] + x[1]) * S;
    let ee0 = e0 + e1;
    let ee1 = e0 - e1;
    let oo0 = (o0 + o1) * IC4[0];
    let oo1 = (o0 - o1) * IC4[1];
    [ee0 + oo0, ee1 + oo1, ee1 - oo1, ee0 - oo0]
}

#[inline(always)]
fn idct4_1d(x: [f32; 4]) -> [f32; 4] {
    const S: f32 = std::f32::consts::SQRT_2;
    idct4_raw([x[0], x[1] * S, x[2] * S, x[3] * S])
}

#[inline(always)]
fn idct8_1d(x: [f32; 8]) -> [f32; 8] {
    const S: f32 = std::f32::consts::SQRT_2;
    let x = [x[0], x[1] * S, x[2] * S, x[3] * S, x[4] * S, x[5] * S, x[6] * S, x[7] * S];
    let e = idct4_raw([x[0], x[2], x[4], x[6]]);
    let o = idct4_raw([x[1], x[3] + x[1], x[5] + x[3], x[7] + x[5]]);
    let v0 = o[0] * IC8[0];
    let v1 = o[1] * IC8[1];
    let v2 = o[2] * IC8[2];
    let v3 = o[3] * IC8[3];
    [e[0] + v0, e[1] + v1, e[2] + v2, e[3] + v3, e[3] - v3, e[2] - v2, e[1] - v1, e[0] - v0]
}

/// Forward 2-D DCT of an `rows` x `cols` pixel rectangle (rows, cols in {4, 8})
/// into natural layout `out[v * cols + u]` (v vertical, u horizontal frequency).
#[inline(always)]
fn dct2d(src: &[f32], stride: usize, rows: usize, cols: usize, out: &mut [f32]) {
    let mut tmp = [0f32; 64];
    // Horizontal pass: tmp[y * cols + u].
    if cols == 4 {
        for y in 0..rows {
            let r = &src[y * stride..];
            let o = dct4_1d([r[0], r[1], r[2], r[3]]);
            tmp[y * 4..y * 4 + 4].copy_from_slice(&o);
        }
    } else {
        for y in 0..rows {
            let r = &src[y * stride..];
            let o = dct8_1d([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]);
            tmp[y * 8..y * 8 + 8].copy_from_slice(&o);
        }
    }
    // Vertical pass: out[v * cols + u].
    if rows == 4 {
        for u in 0..cols {
            let o = dct4_1d([tmp[u], tmp[cols + u], tmp[2 * cols + u], tmp[3 * cols + u]]);
            for v in 0..4 {
                out[v * cols + u] = o[v];
            }
        }
    } else {
        for u in 0..cols {
            let o = dct8_1d([
                tmp[u],
                tmp[cols + u],
                tmp[2 * cols + u],
                tmp[3 * cols + u],
                tmp[4 * cols + u],
                tmp[5 * cols + u],
                tmp[6 * cols + u],
                tmp[7 * cols + u],
            ]);
            for v in 0..8 {
                out[v * cols + u] = o[v];
            }
        }
    }
}

/// Inverse of `dct2d`; writes an `rows` x `cols` pixel rectangle.
#[inline(always)]
fn idct2d(coef: &[f32], rows: usize, cols: usize, out: &mut [f32], stride: usize) {
    let mut tmp = [0f32; 64];
    // Vertical pass: tmp[y * cols + u].
    if rows == 4 {
        for u in 0..cols {
            let o = idct4_1d([coef[u], coef[cols + u], coef[2 * cols + u], coef[3 * cols + u]]);
            for y in 0..4 {
                tmp[y * cols + u] = o[y];
            }
        }
    } else {
        for u in 0..cols {
            let o = idct8_1d([
                coef[u],
                coef[cols + u],
                coef[2 * cols + u],
                coef[3 * cols + u],
                coef[4 * cols + u],
                coef[5 * cols + u],
                coef[6 * cols + u],
                coef[7 * cols + u],
            ]);
            for y in 0..8 {
                tmp[y * cols + u] = o[y];
            }
        }
    }
    // Horizontal pass: out[y * stride + x].
    if cols == 4 {
        for y in 0..rows {
            let t = &tmp[y * 4..];
            let o = idct4_1d([t[0], t[1], t[2], t[3]]);
            out[y * stride..y * stride + 4].copy_from_slice(&o);
        }
    } else {
        for y in 0..rows {
            let t = &tmp[y * 8..];
            let o = idct8_1d([t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]]);
            out[y * stride..y * stride + 8].copy_from_slice(&o);
        }
    }
}

/// One level of libjxl's DCT2TopBlock on the top-left `s` x `s` of an 8x8.
fn haar_fwd(buf: &mut [f32; 64], src: &[f32], src_stride: usize, s: usize) {
    let n = s / 2;
    let mut tmp = [0f32; 64];
    for y in 0..n {
        for x in 0..n {
            let c00 = src[2 * y * src_stride + 2 * x];
            let c01 = src[2 * y * src_stride + 2 * x + 1];
            let c10 = src[(2 * y + 1) * src_stride + 2 * x];
            let c11 = src[(2 * y + 1) * src_stride + 2 * x + 1];
            tmp[y * 8 + x] = 0.25 * (c00 + c01 + c10 + c11);
            tmp[y * 8 + n + x] = 0.25 * (c00 + c01 - c10 - c11);
            tmp[(n + y) * 8 + x] = 0.25 * (c00 - c01 + c10 - c11);
            tmp[(n + y) * 8 + n + x] = 0.25 * (c00 - c01 - c10 + c11);
        }
    }
    for y in 0..s {
        for x in 0..s {
            buf[y * 8 + x] = tmp[y * 8 + x];
        }
    }
}

fn haar_inv(buf: &mut [f32; 64], s: usize) {
    let n = s / 2;
    let mut tmp = [0f32; 64];
    for y in 0..n {
        for x in 0..n {
            let (r00, r01) = (buf[y * 8 + x], buf[y * 8 + n + x]);
            let (r10, r11) = (buf[(n + y) * 8 + x], buf[(n + y) * 8 + n + x]);
            tmp[2 * y * 8 + 2 * x] = r00 + r01 + r10 + r11;
            tmp[2 * y * 8 + 2 * x + 1] = r00 + r01 - r10 - r11;
            tmp[(2 * y + 1) * 8 + 2 * x] = r00 - r01 + r10 - r11;
            tmp[(2 * y + 1) * 8 + 2 * x + 1] = r00 - r01 - r10 + r11;
        }
    }
    for y in 0..s {
        for x in 0..s {
            buf[y * 8 + x] = tmp[y * 8 + x];
        }
    }
}

/// The four-way mix of the DC values of the 2x2 sub-blocks (and its inverse,
/// which is the same butterfly without the 1/4).
fn mix4(c: &mut [f32]) {
    let (b0, b1, b2, b3) = (c[0], c[1], c[8], c[9]);
    c[0] = (b0 + b1 + b2 + b3) * 0.25;
    c[1] = (b0 + b1 - b2 - b3) * 0.25;
    c[8] = (b0 - b1 + b2 - b3) * 0.25;
    c[9] = (b0 - b1 - b2 + b3) * 0.25;
}

fn unmix4(c: &mut [f32]) {
    let (c0, c1, c8, c9) = (c[0], c[1], c[8], c[9]);
    c[0] = c0 + c1 + c8 + c9;
    c[1] = c0 + c1 - c8 - c9;
    c[8] = c0 - c1 + c8 - c9;
    c[9] = c0 - c1 - c8 + c9;
}

/// Forward transform of the 8x8 pixels at `src` (row stride `stride`) into
/// libjxl's 8x8 coefficient layout.
pub fn forward(kind: Small, src: &[f32], stride: usize, out: &mut [f32]) {
    let out = &mut out[..64];
    out.fill(0.0);
    match kind {
        Small::Identity => {
            for y in 0..2 {
                for x in 0..2 {
                    let at = |iy: usize, ix: usize| src[(y * 4 + iy) * stride + x * 4 + ix];
                    let mut dc = 0.0;
                    for iy in 0..4 {
                        for ix in 0..4 {
                            dc += at(iy, ix);
                        }
                    }
                    dc *= 1.0 / 16.0;
                    let ref_px = at(1, 1);
                    for iy in 0..4 {
                        for ix in 0..4 {
                            if ix == 1 && iy == 1 {
                                continue;
                            }
                            out[(y + iy * 2) * 8 + x + ix * 2] = at(iy, ix) - ref_px;
                        }
                    }
                    out[(y + 2) * 8 + x + 2] = out[y * 8 + x];
                    out[y * 8 + x] = dc;
                }
            }
            mix4(out);
        }
        Small::Dct8x4 => {
            for x in 0..2 {
                let mut b = [0f32; 32];
                dct2d(&src[x * 4..], stride, 8, 4, &mut b);
                // b[v * 4 + u]; stored transposed: row = horizontal frequency.
                for u in 0..4 {
                    for v in 0..8 {
                        out[(x + u * 2) * 8 + v] = b[v * 4 + u];
                    }
                }
            }
            let (b0, b1) = (out[0], out[8]);
            out[0] = (b0 + b1) * 0.5;
            out[8] = (b0 - b1) * 0.5;
        }
        Small::Dct4x8 => {
            for y in 0..2 {
                let mut b = [0f32; 32];
                dct2d(&src[y * 4 * stride..], stride, 4, 8, &mut b);
                for v in 0..4 {
                    for u in 0..8 {
                        out[(y + v * 2) * 8 + u] = b[v * 8 + u];
                    }
                }
            }
            let (b0, b1) = (out[0], out[8]);
            out[0] = (b0 + b1) * 0.5;
            out[8] = (b0 - b1) * 0.5;
        }
        Small::Dct4x4 => {
            for y in 0..2 {
                for x in 0..2 {
                    let mut b = [0f32; 16];
                    dct2d(&src[y * 4 * stride + x * 4..], stride, 4, 4, &mut b);
                    // Transposed: row = horizontal frequency.
                    for u in 0..4 {
                        for v in 0..4 {
                            out[(y + u * 2) * 8 + x + v * 2] = b[v * 4 + u];
                        }
                    }
                }
            }
            mix4(out);
        }
        Small::Dct2x2 => {
            let mut buf = [0f32; 64];
            haar_fwd(&mut buf, src, stride, 8);
            let copy = buf;
            haar_fwd(&mut buf, &copy, 8, 4);
            let copy = buf;
            haar_fwd(&mut buf, &copy, 8, 2);
            out.copy_from_slice(&buf);
        }
    }
}

/// Inverse of `forward`: coefficients (8x8 layout) to an 8x8 pixel block.
pub fn inverse(kind: Small, coef: &[f32], out: &mut [f32], stride: usize) {
    let mut c = [0f32; 64];
    c.copy_from_slice(&coef[..64]);
    match kind {
        Small::Identity => {
            unmix4(&mut c);
            for y in 0..2 {
                for x in 0..2 {
                    let dc = c[y * 8 + x];
                    let mut d = [[0f32; 4]; 4];
                    let mut sum = 0.0;
                    for iy in 0..4 {
                        for ix in 0..4 {
                            if ix == 1 && iy == 1 {
                                continue;
                            }
                            let slot =
                                if ix == 0 && iy == 0 { (y + 2) * 8 + x + 2 } else { (y + iy * 2) * 8 + x + ix * 2 };
                            d[iy][ix] = c[slot];
                            sum += d[iy][ix];
                        }
                    }
                    let centre = dc - sum / 16.0;
                    for iy in 0..4 {
                        for ix in 0..4 {
                            out[(y * 4 + iy) * stride + x * 4 + ix] = d[iy][ix] + centre;
                        }
                    }
                }
            }
        }
        Small::Dct8x4 => {
            let (c0, c8) = (c[0], c[8]);
            c[0] = c0 + c8;
            c[8] = c0 - c8;
            for x in 0..2 {
                let mut b = [0f32; 32];
                for u in 0..4 {
                    for v in 0..8 {
                        b[v * 4 + u] = c[(x + u * 2) * 8 + v];
                    }
                }
                idct2d(&b, 8, 4, &mut out[x * 4..], stride);
            }
        }
        Small::Dct4x8 => {
            let (c0, c8) = (c[0], c[8]);
            c[0] = c0 + c8;
            c[8] = c0 - c8;
            for y in 0..2 {
                let mut b = [0f32; 32];
                for v in 0..4 {
                    for u in 0..8 {
                        b[v * 8 + u] = c[(y + v * 2) * 8 + u];
                    }
                }
                idct2d(&b, 4, 8, &mut out[y * 4 * stride..], stride);
            }
        }
        Small::Dct4x4 => {
            unmix4(&mut c);
            for y in 0..2 {
                for x in 0..2 {
                    let mut b = [0f32; 16];
                    for u in 0..4 {
                        for v in 0..4 {
                            b[v * 4 + u] = c[(y + u * 2) * 8 + x + v * 2];
                        }
                    }
                    idct2d(&b, 4, 4, &mut out[y * 4 * stride + x * 4..], stride);
                }
            }
        }
        Small::Dct2x2 => {
            haar_inv(&mut c, 2);
            haar_inv(&mut c, 4);
            haar_inv(&mut c, 8);
            for y in 0..8 {
                for x in 0..8 {
                    out[y * stride + x] = c[y * 8 + x];
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Small {
    Identity,
    Dct2x2,
    Dct4x4,
    Dct4x8,
    Dct8x4,
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [Small; 5] = [Small::Identity, Small::Dct2x2, Small::Dct4x4, Small::Dct4x8, Small::Dct8x4];

    fn pixels() -> Vec<f32> {
        let mut s = 12345u32;
        (0..8 * 11)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / 16_777_216.0
            })
            .collect()
    }

    #[test]
    fn roundtrip_and_mean() {
        let px = pixels();
        let stride = 11;
        let mean: f32 =
            (0..8).flat_map(|y| (0..8).map(move |x| (y, x))).map(|(y, x)| px[y * stride + x]).sum::<f32>() / 64.0;
        for k in KINDS {
            let mut c = [0f32; 64];
            forward(k, &px, stride, &mut c);
            assert!((c[0] - mean).abs() < 1e-5, "{k:?} dc {} vs {mean}", c[0]);
            let mut out = [0f32; 64];
            inverse(k, &c, &mut out, 8);
            for y in 0..8 {
                for x in 0..8 {
                    let (a, b) = (out[y * 8 + x], px[y * stride + x]);
                    assert!((a - b).abs() < 1e-4, "{k:?} ({x},{y}): {a} vs {b}");
                }
            }
        }
    }
}
