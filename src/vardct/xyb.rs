//! sRGB to XYB conversion (libjxl's opsin absorbance model).

use crate::parallel::par_tasks;
const M: [[f32; 3]; 3] = [
    [0.30, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_69, 0.204_767_44, 0.551_809_87],
];
const BIAS: f32 = 0.003_793_073_3;

fn srgb_to_linear_lut() -> [f32; 256] {
    std::array::from_fn(|i| {
        let v = i as f64 / 255.0;
        (if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }) as f32
    })
}

/// Fast cube root for non-negative finite input: bit-trick seed refined by
/// two Halley steps (relative error ~1e-7). Branch-free so loops vectorise.
#[inline(always)]
fn fast_cbrt(x: f32) -> f32 {
    let i = x.to_bits();
    // Seed: divide the exponent by three (+ magic bias).
    let mut y = f32::from_bits(i / 3 + 0x2a51_7d47);
    for _ in 0..2 {
        let y3 = y * y * y;
        y *= (y3 + x + x) / (y3 + y3 + x);
    }
    // x == 0 makes the denominators 0; return 0 there.
    if x > 0.0 { y } else { 0.0 }
}

#[inline(always)]
fn row_to_xyb_impl(
    lut: &[f32; 256],
    row: &[u8],
    channels: usize,
    w: usize,
    bias_cbrt: f32,
    mix: &mut [f32],
    ra: &mut [f32],
    rb: &mut [f32],
    rc: &mut [f32],
) {
    let pw = ra.len();
    let (ml, rest) = mix.split_at_mut(pw);
    let (mm, ms) = rest.split_at_mut(pw);
    for x in 0..w {
        let p = &row[x * channels..];
        let (r, g, b) = (lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]);
        ml[x] = (M[0][0] * r + M[0][1] * g + M[0][2] * b + BIAS).max(0.0);
        mm[x] = (M[1][0] * r + M[1][1] * g + M[1][2] * b + BIAS).max(0.0);
        ms[x] = (M[2][0] * r + M[2][1] * g + M[2][2] * b + BIAS).max(0.0);
    }
    for x in 0..w {
        let l = fast_cbrt(ml[x]) - bias_cbrt;
        let m = fast_cbrt(mm[x]) - bias_cbrt;
        let s = fast_cbrt(ms[x]) - bias_cbrt;
        ra[x] = 0.5 * (l - m);
        rb[x] = 0.5 * (l + m);
        rc[x] = s;
    }
    // Edge replication.
    let (a, b, c) = (ra[w - 1], rb[w - 1], rc[w - 1]);
    ra[w..pw].fill(a);
    rb[w..pw].fill(b);
    rc[w..pw].fill(c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_to_xyb_avx2(
    lut: &[f32; 256],
    row: &[u8],
    channels: usize,
    w: usize,
    bias_cbrt: f32,
    mix: &mut [f32],
    ra: &mut [f32],
    rb: &mut [f32],
    rc: &mut [f32],
) {
    row_to_xyb_impl(lut, row, channels, w, bias_cbrt, mix, ra, rb, rc)
}

fn row_to_xyb(
    lut: &[f32; 256],
    row: &[u8],
    channels: usize,
    w: usize,
    bias_cbrt: f32,
    mix: &mut [f32],
    ra: &mut [f32],
    rb: &mut [f32],
    rc: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: features detected at runtime.
        return unsafe { row_to_xyb_avx2(lut, row, channels, w, bias_cbrt, mix, ra, rb, rc) };
    }
    row_to_xyb_impl(lut, row, channels, w, bias_cbrt, mix, ra, rb, rc)
}

/// Planar XYB image padded (by edge replication) to `pw` x `ph`.
pub struct Xyb {
    pub w: usize,
    pub h: usize,
    pub planes: [Vec<f32>; 3],
}

impl Xyb {
    pub fn from_srgb8(
        data: &[u8],
        channels: usize,
        w: usize,
        h: usize,
        pw: usize,
        ph: usize,
        threads: usize,
    ) -> Self {
        let lut = srgb_to_linear_lut();
        let bias_cbrt = fast_cbrt(BIAS);
        let mut planes = [
            vec![0f32; pw * ph],
            vec![0f32; pw * ph],
            vec![0f32; pw * ph],
        ];
        let [p0, p1, p2] = &mut planes;
        let band = pw * 8;
        let tasks: Vec<_> = p0
            .chunks_mut(band)
            .zip(p1.chunks_mut(band))
            .zip(p2.chunks_mut(band))
            .enumerate()
            .collect();
        par_tasks(tasks, threads, |(i, ((a, b), c))| {
            let mut mix = vec![0f32; 3 * pw];
            for (ry, ((ra, rb), rc)) in a
                .chunks_mut(pw)
                .zip(b.chunks_mut(pw))
                .zip(c.chunks_mut(pw))
                .enumerate()
            {
                let sy = (i * 8 + ry).min(h - 1);
                let row = &data[sy * w * channels..(sy + 1) * w * channels];
                row_to_xyb(&lut, row, channels, w, bias_cbrt, &mut mix, ra, rb, rc);
            }
        });
        Xyb {
            w: pw,
            h: ph,
            planes,
        }
    }
}

/// libjxl's inverse Gaborish: a symmetric 5x5 sharpening filter applied
/// before the DCT so that the decoder's 3x3 smoothing restores the image.
/// Borders are mirrored with the edge pixel repeated.
pub fn gaborish_inverse(x: &mut Xyb, threads: usize) {
    const K: [f32; 5] = [
        -0.094_958_16,
        -0.041_031_726,
        0.013_710_005,
        0.006_510_206,
        -0.001_478_906_3,
    ];
    let n = 1.0 / (1.0 + 4.0 * (K[0] + K[1] + K[2] + K[4] + 2.0 * K[3]));
    // Weights by distance from the centre: [centre-row/col offsets 0, 1, 2]
    // for the vertical offset classes 0, 1, 2.
    let wt: [[f32; 3]; 3] = [
        [n, n * K[0], n * K[2]],
        [n * K[0], n * K[1], n * K[3]],
        [n * K[2], n * K[3], n * K[4]],
    ];
    for plane in &mut x.planes {
        *plane = gaborish_plane(std::mem::take(plane), x.w, x.h, &wt, threads);
    }
}

fn mirror(mut v: isize, len: usize) -> usize {
    let len = len as isize;
    while v < 0 || v >= len {
        v = if v < 0 { -v - 1 } else { 2 * len - 1 - v };
    }
    v as usize
}

/// Sharpen in place using five original rows per band. Snapshot the halos
/// before starting workers so no band reads a neighbour's overwritten pixels.
/// Workspace scales with width * workers, not the image area.
fn gaborish_plane(
    mut src: Vec<f32>,
    w: usize,
    h: usize,
    wt: &[[f32; 3]; 3],
    threads: usize,
) -> Vec<f32> {
    let band_rows = h.div_ceil(threads.max(1)).max(8);
    // Halos (4), ring (5, of which 2 reuse halos), row sums (3).
    if h <= 10 * h.div_ceil(band_rows) {
        return gaborish_out_of_place(&src, w, h, wt, threads);
    }
    let halos: Vec<[Vec<f32>; 4]> = (0..h)
        .step_by(band_rows)
        .map(|start| {
            let end = (start + band_rows).min(h);
            [
                start as isize - 2,
                start as isize - 1,
                end as isize,
                end as isize + 1,
            ]
            .map(|y| {
                let y = mirror(y, h);
                src[y * w..(y + 1) * w].to_vec()
            })
        })
        .collect();
    par_tasks(
        src.chunks_mut(w * band_rows).zip(halos).collect(),
        threads,
        |(chunk, [top2, top1, bottom1, bottom2])| {
            let rows = chunk.len() / w;
            let mut ring = [top2, top1, vec![0.0; w], vec![0.0; w], vec![0.0; w]];
            for i in 0..3 {
                let row = if i < rows {
                    &chunk[i * w..(i + 1) * w]
                } else if i == rows {
                    &bottom1
                } else {
                    &bottom2
                };
                ring[i + 2].copy_from_slice(row);
            }
            let mut s = [vec![0.0; w], vec![0.0; w], vec![0.0; w]];
            for y in 0..rows {
                let r: [&[f32]; 5] = std::array::from_fn(|i| ring[(y + i) % 5].as_slice());
                for i in 0..w {
                    s[0][i] = r[2][i];
                    s[1][i] = r[1][i] + r[3][i];
                    s[2][i] = r[0][i] + r[4][i];
                }
                gaborish_row(&s, wt, &mut chunk[y * w..(y + 1) * w], mirror);
                if y + 1 < rows {
                    let next = y + 3;
                    let row = if next < rows {
                        &chunk[next * w..(next + 1) * w]
                    } else if next == rows {
                        &bottom1
                    } else {
                        &bottom2
                    };
                    ring[y % 5].copy_from_slice(row);
                }
            }
        },
    );
    src
}

fn gaborish_out_of_place(
    src: &[f32],
    w: usize,
    h: usize,
    wt: &[[f32; 3]; 3],
    threads: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0; w * h];
    par_tasks(
        dst.chunks_mut(w * 8).enumerate().collect(),
        threads,
        |(bi, out)| {
            let mut s = [vec![0.0; w], vec![0.0; w], vec![0.0; w]];
            for (ry, orow) in out.chunks_mut(w).enumerate() {
                let y = bi * 8 + ry;
                let r = |d: isize| {
                    let i = mirror(y as isize + d, h);
                    &src[i * w..(i + 1) * w]
                };
                let (rm2, rm1, r0, rp1, rp2) = (r(-2), r(-1), r(0), r(1), r(2));
                for i in 0..w {
                    s[0][i] = r0[i];
                    s[1][i] = rm1[i] + rp1[i];
                    s[2][i] = rm2[i] + rp2[i];
                }
                gaborish_row(&s, wt, orow, mirror);
            }
        },
    );
    dst
}

#[inline(always)]
fn gaborish_row_impl(
    s: &[Vec<f32>; 3],
    wt: &[[f32; 3]; 3],
    out: &mut [f32],
    mirror: impl Fn(isize, usize) -> usize,
) {
    let w = out.len();
    let tap = |i: usize| -> f32 {
        let mut acc = 0f32;
        for (j, sj) in s.iter().enumerate() {
            let at = |d: isize| sj[mirror(i as isize + d, w)];
            acc += wt[j][0] * at(0) + wt[j][1] * (at(-1) + at(1)) + wt[j][2] * (at(-2) + at(2));
        }
        acc
    };
    if w < 5 {
        for (i, o) in out.iter_mut().enumerate() {
            *o = tap(i);
        }
        return;
    }
    for i in [0, 1, w - 2, w - 1] {
        out[i] = tap(i);
    }
    let (s0, s1, s2) = (&s[0], &s[1], &s[2]);
    for i in 2..w - 2 {
        out[i] = wt[0][0] * s0[i]
            + wt[0][1] * (s0[i - 1] + s0[i + 1])
            + wt[0][2] * (s0[i - 2] + s0[i + 2])
            + wt[1][0] * s1[i]
            + wt[1][1] * (s1[i - 1] + s1[i + 1])
            + wt[1][2] * (s1[i - 2] + s1[i + 2])
            + wt[2][0] * s2[i]
            + wt[2][1] * (s2[i - 1] + s2[i + 1])
            + wt[2][2] * (s2[i - 2] + s2[i + 2]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gaborish_row_avx2(
    s: &[Vec<f32>; 3],
    wt: &[[f32; 3]; 3],
    out: &mut [f32],
    mirror: impl Fn(isize, usize) -> usize,
) {
    gaborish_row_impl(s, wt, out, mirror)
}

fn gaborish_row(
    s: &[Vec<f32>; 3],
    wt: &[[f32; 3]; 3],
    out: &mut [f32],
    mirror: impl Fn(isize, usize) -> usize,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: features detected at runtime.
        return unsafe { gaborish_row_avx2(s, wt, out, mirror) };
    }
    gaborish_row_impl(s, wt, out, mirror)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_place_gaborish_matches_reference() {
        let wt = [
            [1.2, -0.1, 0.015],
            [-0.1, -0.04, -0.002],
            [0.015, -0.002, 0.007],
        ];
        for (w, h) in [
            (1, 1),
            (1, 37),
            (2, 65),
            (3, 33),
            (8, 8),
            (31, 67),
            (193, 129),
        ] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| ((i * 1237 % 65521) as f32 - 17000.0) / 257.0)
                .collect();
            let expected = gaborish_out_of_place(&src, w, h, &wt, 1);
            for threads in [1, 2, 3, 4, 8, 100] {
                let actual = gaborish_plane(src.clone(), w, h, &wt, threads);
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{w}x{h} t{threads}"
                );
            }
        }
    }

    #[test]
    fn fast_cbrt_accurate() {
        let mut x = 1e-6f32;
        while x < 2.0 {
            let e = (fast_cbrt(x) - x.cbrt()).abs() / x.cbrt();
            assert!(e < 3e-7, "{x}: {e}");
            x *= 1.01;
        }
        assert_eq!(fast_cbrt(0.0), 0.0);
    }
}
