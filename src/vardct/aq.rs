//! Adaptive quantisation: libjxl's initial quantisation field, a per-block
//! estimate of how much each 8x8 block masks quantisation error.

use super::xyb::Xyb;
use crate::parallel::par_tasks;

const SG_MUL: f32 = 226.772_16;
const SG_MUL2: f32 = 1.0 / 73.377_13;
const INV_LOG2E: f32 = std::f32::consts::LN_2;
const SG_RET_MUL: f32 = SG_MUL2 * 18.658_093 * INV_LOG2E;
const SG_V_OFFSET: f32 = 7.782_599;

/// Ratio of derivatives of the cube root and a simple gamma curve.
#[inline]
fn ratio_of_derivatives(v: f32, invert: bool) -> f32 {
    const EPS: f32 = 1e-2;
    let v = v.max(0.0);
    let v2 = v * v;
    let num = SG_RET_MUL * 3.0 * SG_MUL * v2 + EPS;
    let den = INV_LOG2E * SG_MUL * v * v2 + SG_V_OFFSET * INV_LOG2E + EPS;
    if invert { num / den } else { den / num }
}

#[inline]
fn masking_sqrt(v: f32) -> f32 {
    const LOG_OFFSET: f32 = 27.505_837;
    const MUL: f32 = 211.665_68;
    0.25 * (v * (MUL * 1e8f32).sqrt() + LOG_OFFSET).sqrt()
}

fn compute_mask(out_val: f32) -> f32 {
    const BASE: f32 = -0.7647;
    const MUL4: f32 = 9.470_873;
    const MUL2: f32 = 17.350_365;
    const OFFSET2: f32 = 302.595_88;
    const MUL3: f32 = 6.794_325;
    const OFFSET3: f32 = 3.717_963_6;
    const OFFSET4: f32 = 0.25 * OFFSET3;
    const MUL0: f32 = 0.800_617_6;
    let v1 = (out_val * MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + OFFSET2);
    let v3 = 1.0 / (v1 * v1 + OFFSET3);
    let v4 = 1.0 / (v1 * v1 + OFFSET4);
    BASE + MUL4 * v4 + MUL2 * v2 + MUL3 * v3
}

/// The four smallest values, in increasing order.
#[inline]
fn min4(vals: [f32; 9]) -> [f32; 4] {
    let mut m = [f32::INFINITY; 4];
    for v in vals {
        if v < m[3] {
            let mut i = 3;
            while i > 0 && v < m[i - 1] {
                m[i] = m[i - 1];
                i -= 1;
            }
            m[i] = v;
        }
    }
    m
}

pub struct AqMaps {
    /// Quantisation field per block (larger = finer quantisation).
    pub quant_field: Vec<f32>,
    /// Per-pixel masking, blurred, for transform selection.
    pub mask1x1: Vec<f32>,
}

/// libjxl's `InitialQuantField` on a padded XYB image.
pub fn initial_quant_field(distance: f32, xyb: &Xyb, scale: f32, threads: usize) -> AqMaps {
    let (w, h) = (xyb.w, xyb.h);
    let (bw, bh) = (w / 8, h / 8);
    let py = &xyb.planes[1];
    const MATCH_GAMMA_OFFSET: f32 = 0.019;
    const LIMIT: f32 = 0.2;

    // Per-pixel masking and the 4x4-downsampled pre-erosion map.
    let (qw, qh) = (w / 4, h / 4);
    let mut pre = vec![0f32; qw * qh];
    let mut mask1x1 = vec![0f32; w * h];
    let tasks: Vec<_> = mask1x1
        .chunks_mut(w * 8)
        .zip(pre.chunks_mut(qw * 2))
        .enumerate()
        .collect();
    par_tasks(tasks, threads, |(band, (mask, pre))| {
        for ry in 0..8 {
            let y = band * 8 + ry;
            let y1 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);
            let row = &py[y * w..(y + 1) * w];
            let row1 = &py[y1 * w..(y1 + 1) * w];
            let row2 = &py[y2 * w..(y2 + 1) * w];
            for x in 0..w {
                let x1 = x.saturating_sub(1);
                let x2 = (x + 1).min(w - 1);
                let base = 0.25 * (row2[x] + row1[x] + row[x1] + row[x2]);
                let gammac = ratio_of_derivatives(row[x] + MATCH_GAMMA_OFFSET, false);
                let d = gammac * (row[x] - base);
                mask[ry * w + x] = 1.0 / (d.abs().ln_1p() + 0.01);
                let diff = masking_sqrt((d * d).min(LIMIT));
                pre[(ry / 4) * qw + x / 4] += diff * 0.25;
            }
        }
    });

    // Fuzzy erosion to block resolution.
    let mul = if distance < 2.0 {
        (2.0 - distance) * 0.5
    } else {
        0.0
    };
    let base = [0.125f32, 0.1, 0.09, 0.06];
    let add = [0.0f32, -0.1, -0.09, -0.06];
    let mut k = [0f32; 4];
    for i in 0..4 {
        k[i] = base[i] + mul * add[i];
    }
    let norm: f32 = k.iter().sum();
    for v in &mut k {
        *v *= 0.299_597_06 / norm;
    }
    let mut aq = vec![0f32; bw * bh];
    par_tasks(
        aq.chunks_mut(bw).enumerate().collect(),
        threads,
        |(by, row)| {
            for fy in 2 * by..2 * by + 2 {
                let ym = fy.saturating_sub(1);
                let yp = (fy + 1).min(qh - 1);
                for fx in 0..qw {
                    let xm = fx.saturating_sub(1);
                    let xp = (fx + 1).min(qw - 1);
                    let at = |x: usize, y: usize| pre[y * qw + x];
                    let m = min4([
                        at(fx, fy),
                        at(xm, fy),
                        at(xp, fy),
                        at(xm, ym),
                        at(fx, ym),
                        at(xp, ym),
                        at(xm, yp),
                        at(fx, yp),
                        at(xp, yp),
                    ]);
                    row[fx / 2] += k[0] * m[0] + k[1] * m[1] + k[2] * m[2] + k[3] * m[3];
                }
            }
        },
    );

    drop(pre);

    // Per-block modulations.
    let dampen = if distance >= 2.0 {
        (1.0 - (distance - 2.0) / 12.0).max(0.0)
    } else {
        1.0
    };
    let mulv = scale * dampen;
    let addv = (1.0 - dampen) * 0.48 * scale;
    let (px, pb) = (&xyb.planes[0], &xyb.planes[2]);
    par_tasks(
        aq.chunks_mut(bw).enumerate().collect(),
        threads,
        |(by, row)| {
            for bx in 0..bw {
                let (x0, y0) = (bx * 8, by * 8);
                let mut v = compute_mask(row[bx]);
                // Gamma modulation.
                let mut ratio = 0f32;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let i = (y0 + dy) * w + x0 + dx;
                        let iny = py[i] + 0.16;
                        let inx = px[i];
                        ratio += ratio_of_derivatives(iny - inx, true);
                        ratio += ratio_of_derivatives(iny + inx, true);
                    }
                }
                v += 0.100_561_33 * (ratio * (0.5 / 64.0)).log2();
                // HF modulation.
                let mut sum_y = 0f32;
                for dy in 0..8 {
                    let r = (y0 + dy) * w + x0;
                    let rn = if dy == 7 { r } else { r + w };
                    for dx in 0..8 {
                        if dx < 7 {
                            sum_y += (py[r + dx] - py[r + dx + 1]).abs().min(0.0206);
                        }
                        sum_y += (py[r + dx] - py[rn + dx]).abs().min(0.0206);
                    }
                }
                let hf = v + sum_y * -0.38 + 0.42;
                // Blue modulation.
                const LIMIT_B: f32 = 0.010_474_085;
                let mut sum_b = 0f32;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let i = (y0 + dy) * w + x0 + dx;
                        let eff = py[i] + 0.003_199_477 + px[i].abs();
                        if pb[i] > eff {
                            sum_b += (pb[i] - eff).min(LIMIT_B);
                        }
                    }
                }
                if sum_b >= 32.0 * LIMIT_B {
                    sum_b = 64.0 * LIMIT_B - sum_b;
                }
                sum_b = sum_b.min(15.463_398 * LIMIT_B) * 0.905_908;
                let out = hf.min(v + sum_b);
                row[bx] = out.exp() * mulv + addv;
            }
        },
    );

    AqMaps {
        quant_field: aq,
        mask1x1: blur_mask1x1(mask1x1, w, h, threads),
    }
}

fn blur_mask1x1(m: Vec<f32>, w: usize, h: usize, threads: usize) -> Vec<f32> {
    const K: [f32; 5] = [0.364_911_25, 0.05, 0.168_888_8, 0.221_069_18, 0.306_563_5];
    let sum = 1.0 + 4.0 * (K[0] + K[1] + K[2] + K[4] + 2.0 * K[3]);
    let n = 1.0 / sum;
    symmetric5(
        m,
        w,
        h,
        [n, n * K[0], n * K[2], n * K[1], n * K[4], n * K[3]],
        threads,
    )
}

/// 5x5 symmetric convolution with mirrored borders. Weights: centre, r
/// (distance 1), R (distance 2), d (diagonal), D (2,2), L (1,2) knight moves.
fn symmetric5(mut src: Vec<f32>, w: usize, h: usize, k: [f32; 6], threads: usize) -> Vec<f32> {
    let [c, r, rr, d, dd, l] = k;
    let mirror = |v: isize, n: usize| -> usize {
        let n = n as isize;
        let mut v = v;
        loop {
            if v < 0 {
                v = -v - 1;
            } else if v >= n {
                v = 2 * n - 1 - v;
            } else {
                return v as usize;
            }
        }
    };
    // Snapshot the two boundary rows on either side of every band before
    // workers overwrite the input. Within a band, a five-row ring keeps
    // original samples alive until their last use. No second image plane
    // is needed, and each worker writes only its own disjoint band.
    let band_rows = h.div_ceil(threads.max(1)).max(8);
    // Seven rows per band (halos + ring) and at most ten rows' worth
    // of column indices. Keep the original method for shallow images
    // when this workspace would exceed a second image plane.
    if h <= 7 * h.div_ceil(band_rows) + 10 {
        return symmetric5_out_of_place(&src, w, h, k, threads);
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
    let columns: Vec<[usize; 5]> = (0..w)
        .map(|x| std::array::from_fn(|i| mirror(x as isize + i as isize - 2, w)))
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
            for y in 0..rows {
                let orow = &mut chunk[y * w..(y + 1) * w];
                let window: [&[f32]; 5] = std::array::from_fn(|dy| ring[(y + dy) % 5].as_slice());
                for (x, cols) in columns.iter().enumerate() {
                    let p = |dy: usize, dx: usize| window[dy][cols[dx]];
                    orow[x] = c * p(2, 2)
                        + r * (p(1, 2) + p(3, 2) + p(2, 1) + p(2, 3))
                        + rr * (p(0, 2) + p(4, 2) + p(2, 0) + p(2, 4))
                        + d * (p(1, 1) + p(1, 3) + p(3, 1) + p(3, 3))
                        + dd * (p(0, 0) + p(0, 4) + p(4, 0) + p(4, 4))
                        + l * (p(0, 1)
                            + p(0, 3)
                            + p(4, 1)
                            + p(4, 3)
                            + p(1, 0)
                            + p(3, 0)
                            + p(1, 4)
                            + p(3, 4));
                }
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

fn symmetric5_out_of_place(
    src: &[f32],
    w: usize,
    h: usize,
    k: [f32; 6],
    threads: usize,
) -> Vec<f32> {
    let [c, r, rr, d, dd, l] = k;
    let mirror = |v: isize, n: usize| -> usize {
        let n = n as isize;
        let mut v = v;
        loop {
            if v < 0 {
                v = -v - 1;
            } else if v >= n {
                v = 2 * n - 1 - v;
            } else {
                return v as usize;
            }
        }
    };
    let mut out = vec![0f32; w * h];
    par_tasks(
        out.chunks_mut(w * 8).enumerate().collect(),
        threads,
        |(band, chunk)| {
            for (ry, orow) in chunk.chunks_mut(w).enumerate() {
                let y = band * 8 + ry;
                let rows: [usize; 5] =
                    std::array::from_fn(|i| mirror(y as isize + i as isize - 2, h) * w);
                for x in 0..w {
                    let cols: [usize; 5] =
                        std::array::from_fn(|i| mirror(x as isize + i as isize - 2, w));
                    let p = |dy: usize, dx: usize| src[rows[dy] + cols[dx]];
                    orow[x] = c * p(2, 2)
                        + r * (p(1, 2) + p(3, 2) + p(2, 1) + p(2, 3))
                        + rr * (p(0, 2) + p(4, 2) + p(2, 0) + p(2, 4))
                        + d * (p(1, 1) + p(1, 3) + p(3, 1) + p(3, 3))
                        + dd * (p(0, 0) + p(0, 4) + p(4, 0) + p(4, 4))
                        + l * (p(0, 1)
                            + p(0, 3)
                            + p(4, 1)
                            + p(4, 3)
                            + p(1, 0)
                            + p(3, 0)
                            + p(1, 4)
                            + p(3, 4));
                }
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_place_blur_matches_reference() {
        for (w, h) in [
            (1, 1),
            (1, 19),
            (193, 8),
            (2, 2),
            (3, 7),
            (8, 8),
            (17, 25),
            (31, 33),
            (31, 67),
        ] {
            let src: Vec<_> = (0..w * h)
                .map(|i| ((i * 1237 % 65521) as f32 - 17000.0) / 257.0)
                .collect();
            let k = [0.13, 0.07, 0.035, 0.02, 0.01, 0.005];
            let expected = symmetric5_out_of_place(&src, w, h, k, 1);
            for threads in [1, 2, 3, 8, 100] {
                let actual = symmetric5(src.clone(), w, h, k, threads);
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
}
