//! The 8x8 transforms other than the plain DCT: Identity, DCT2x2, DCT4x4,
//! DCT4x8 and DCT8x4. They follow libjxl's coefficient layout exactly: the
//! result is an 8x8 array whose entry 0 is the block mean (the LF value).
//!
//! The DCT scaling is the same as in `dct.rs` (DC is the mean, AC basis
//! functions carry a factor of sqrt(2)).

use std::sync::OnceLock;

/// `K[k][i]`: forward 1-D DCT matrix of size N (N = 4 or 8), and `J[i][k]`:
/// its inverse.
struct Basis {
    fwd: [[f32; 8]; 8],
    inv: [[f32; 8]; 8],
}

fn basis(n: usize) -> &'static Basis {
    static B4: OnceLock<Basis> = OnceLock::new();
    static B8: OnceLock<Basis> = OnceLock::new();
    let cell = if n == 4 { &B4 } else { &B8 };
    cell.get_or_init(|| {
        let mut b = Basis {
            fwd: [[0.0; 8]; 8],
            inv: [[0.0; 8]; 8],
        };
        for k in 0..n {
            let s = if k == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
            for i in 0..n {
                let c =
                    (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
                b.fwd[k][i] = (s * c / n as f64) as f32;
                b.inv[i][k] = (s * c) as f32;
            }
        }
        b
    })
}

/// Forward 2-D DCT of an `rows` x `cols` pixel rectangle into natural layout
/// `out[v * cols + u]` (v vertical, u horizontal frequency).
fn dct2d(src: &[f32], stride: usize, rows: usize, cols: usize, out: &mut [f32]) {
    let (br, bc) = (basis(rows), basis(cols));
    let mut tmp = [0f32; 64];
    // Horizontal pass: tmp[y][u].
    for y in 0..rows {
        for u in 0..cols {
            let mut s = 0.0;
            for x in 0..cols {
                s += src[y * stride + x] * bc.fwd[u][x];
            }
            tmp[y * cols + u] = s;
        }
    }
    for v in 0..rows {
        for u in 0..cols {
            let mut s = 0.0;
            for y in 0..rows {
                s += tmp[y * cols + u] * br.fwd[v][y];
            }
            out[v * cols + u] = s;
        }
    }
}

/// Inverse of `dct2d`; writes an `rows` x `cols` pixel rectangle.
fn idct2d(coef: &[f32], rows: usize, cols: usize, out: &mut [f32], stride: usize) {
    let (br, bc) = (basis(rows), basis(cols));
    let mut tmp = [0f32; 64];
    for y in 0..rows {
        for u in 0..cols {
            let mut s = 0.0;
            for v in 0..rows {
                s += coef[v * cols + u] * br.inv[y][v];
            }
            tmp[y * cols + u] = s;
        }
    }
    for y in 0..rows {
        for x in 0..cols {
            let mut s = 0.0;
            for u in 0..cols {
                s += tmp[y * cols + u] * bc.inv[x][u];
            }
            out[y * stride + x] = s;
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
                            let slot = if ix == 0 && iy == 0 {
                                (y + 2) * 8 + x + 2
                            } else {
                                (y + iy * 2) * 8 + x + ix * 2
                            };
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

    const KINDS: [Small; 5] = [
        Small::Identity,
        Small::Dct2x2,
        Small::Dct4x4,
        Small::Dct4x8,
        Small::Dct8x4,
    ];

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
        let mean: f32 = (0..8)
            .flat_map(|y| (0..8).map(move |x| (y, x)))
            .map(|(y, x)| px[y * stride + x])
            .sum::<f32>()
            / 64.0;
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
