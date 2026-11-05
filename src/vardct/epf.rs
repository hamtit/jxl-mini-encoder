//! Per-block EPF sharpness selection (libjxl's `ComputeARHeuristics`).
//!
//! The decoder smooths each 8x8 block with an edge-preserving filter whose
//! strength is `sharpness / 7 * f(quant)`. libjxl (effort >= 6) reconstructs
//! the image under a few candidate strengths, compares each block with the
//! original and keeps the strength that gives the smallest masked error. This
//! module reproduces that: Gaborish, the three EPF stages and the selection
//! rule, working on whole planes padded to a multiple of 8.

use crate::parallel::{par_map, par_tasks};

/// Inverse sigma below which a block is left unfiltered.
const MIN_SIGMA: f32 = -3.905_243;
const INV_SIGMA_NUM: f32 = -1.171_572_9;
const QUANT_MUL: f32 = 0.46;
const CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];
const BORDER_SAD_MUL: f32 = 2.0 / 3.0;
const PAD: usize = 3;
/// Pixels per tile in the filter passes.
const TILE: usize = 64;

/// Weights of the decoder's 3x3 Gaborish (default header, same for X, Y, B).
const GAB_W1: f32 = 1.1 * 0.104_699_57;
const GAB_W2: f32 = 1.1 * 0.055_680_54;

pub struct Planes {
    pub w: usize,
    pub h: usize,
    pub p: [Vec<f32>; 3],
}

fn mirror(v: isize, len: usize) -> usize {
    let len = len as isize;
    let v = if v < 0 { -v - 1 } else { v };
    (if v >= len { 2 * len - 1 - v } else { v }) as usize
}

/// Copy of `src` with `PAD` mirrored pixels on every side.
fn padded(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let pw = w + 2 * PAD;
    let mut out = vec![0f32; pw * (h + 2 * PAD)];
    for y in 0..h + 2 * PAD {
        let sy = mirror(y as isize - PAD as isize, h);
        let (row, dst) = (&src[sy * w..(sy + 1) * w], &mut out[y * pw..(y + 1) * pw]);
        dst[PAD..PAD + w].copy_from_slice(row);
        for i in 0..PAD {
            dst[PAD - 1 - i] = row[mirror(i as isize, w)];
            dst[PAD + w + i] = row[mirror((w + i) as isize, w)];
        }
    }
    out
}

/// The decoder's Gaborish smoothing, in place.
pub fn gaborish(planes: &mut Planes, threads: usize) {
    let (w, h) = (planes.w, planes.h);
    let n = 1.0 / (1.0 + 4.0 * (GAB_W1 + GAB_W2));
    let (wc, we, wd) = (n, n * GAB_W1, n * GAB_W2);
    let pw = w + 2 * PAD;
    for plane in &mut planes.p {
        let src = padded(plane, w, h);
        let tasks: Vec<_> = plane.chunks_mut(w).enumerate().collect();
        par_tasks(tasks, threads, |(y, row)| {
            let base = (y + PAD) * pw + PAD;
            // Rows y-1, y, y+1 of the padded source, each with one pixel of
            // margin on both sides.
            let up = &src[base - pw - 1..base - pw + w + 1];
            let mid = &src[base - 1..base + w + 1];
            let down = &src[base + pw - 1..base + pw + w + 1];
            let (up, mid, down) = (&up[..w + 2], &mid[..w + 2], &down[..w + 2]);
            let row = &mut row[..w];
            for x in 0..w {
                let out = &mut row[x];
                let edges = mid[x] + mid[x + 2] + up[x + 1] + down[x + 1];
                let diags = up[x] + up[x + 2] + down[x] + down[x + 2];
                *out = wc * mid[x + 1] + we * edges + wd * diags;
            }
        });
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Zero,
    One,
    Two,
}

const OFF_0: [(isize, isize); 12] = [
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
];
const OFF_1: [(isize, isize); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

/// One EPF pass. `inv_sigma` holds one value per 8x8 block.
///
/// Works on bands of one block row. For stages 0 and 1 the SAD of a candidate
/// offset is the plus-shaped sum of per-pixel channel differences, so those
/// differences are computed once per band and offset.
fn run_stage(src: &Planes, dst: &mut Planes, inv_sigma: &[f32], stage: Stage, threads: usize) {
    let (w, h) = (src.w, src.h);
    let bw = w / 8;
    let pw = w + 2 * PAD;
    let padded_src: [Vec<f32>; 3] = std::array::from_fn(|c| padded(&src.p[c], w, h));
    let sigma_scale = match stage {
        Stage::Zero => 0.9,
        Stage::One => 1.0,
        Stage::Two => 6.5,
    };
    let sm = sigma_scale * 1.65;
    let bsm = sm * BORDER_SAD_MUL;
    let offs: &[(isize, isize)] = match stage {
        Stage::Zero => &OFF_0,
        _ => &OFF_1,
    };
    let [d0, d1, d2] = &mut dst.p;
    let tasks: Vec<_> = d0
        .chunks_mut(w * 8)
        .zip(d1.chunks_mut(w * 8))
        .zip(d2.chunks_mut(w * 8))
        .enumerate()
        .collect();
    let ctx = StageCtx {
        w,
        bw,
        pw,
        padded_src: &padded_src,
        inv_sigma,
        stage,
        offs,
        sm,
        bsm,
    };
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    par_tasks(tasks, threads, |(band, ((o0, o1), o2))| {
        run_band(&ctx, band, o0, o1, o2, avx2)
    });
}

/// Everything `run_band` reads besides its output band.
struct StageCtx<'a> {
    w: usize,
    bw: usize,
    pw: usize,
    padded_src: &'a [Vec<f32>; 3],
    inv_sigma: &'a [f32],
    stage: Stage,
    offs: &'a [(isize, isize)],
    sm: f32,
    bsm: f32,
}

fn run_band(
    ctx: &StageCtx,
    band: usize,
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
    avx2: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 support was detected at runtime.
        return unsafe { run_band_avx2(ctx, band, o0, o1, o2) };
    }
    let _ = avx2;
    run_band_impl(ctx, band, o0, o1, o2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn run_band_avx2(
    ctx: &StageCtx,
    band: usize,
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
) {
    run_band_impl(ctx, band, o0, o1, o2)
}

#[inline(always)]
fn run_band_impl(ctx: &StageCtx, band: usize, o0: &mut [f32], o1: &mut [f32], o2: &mut [f32]) {
    let StageCtx {
        w,
        bw,
        pw,
        padded_src,
        inv_sigma,
        stage,
        offs,
        sm,
        bsm,
    } = *ctx;
    let y0 = band * 8;
    let rows = o0.len() / w;
    let sigmas = &inv_sigma[band * bw..(band + 1) * bw];
    if sigmas.iter().all(|&v| v < MIN_SIGMA) {
        for (c, out) in [&mut *o0, &mut *o1, &mut *o2].into_iter().enumerate() {
            for r in 0..rows {
                let from = (y0 + r + PAD) * pw + PAD;
                out[r * w..(r + 1) * w].copy_from_slice(&padded_src[c][from..from + w]);
            }
        }
        return;
    }
    // Blocks that are not filtered are copied; the rest is processed in
    // runs of consecutive active blocks.
    for (c, out) in [&mut *o0, &mut *o1, &mut *o2].into_iter().enumerate() {
        for r in 0..rows {
            let from = (y0 + r + PAD) * pw + PAD;
            out[r * w..(r + 1) * w].copy_from_slice(&padded_src[c][from..from + w]);
        }
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (bx, &v) in sigmas.iter().enumerate() {
        if v < MIN_SIGMA {
            continue;
        }
        match runs.last_mut() {
            Some(last) if last.1 == bx => last.1 = bx + 1,
            _ => runs.push((bx, bx + 1)),
        }
    }
    // Per-offset channel differences for rows y0-1..=y0+8, x -1..=w.
    let (dw, drows) = (w + 2, rows + 2);
    let mut diff: Vec<f32> = Vec::new();
    if !matches!(stage, Stage::Two) {
        diff = vec![0f32; offs.len() * drows * dw];
    }
    let mut isig_px = vec![0f32; w];
    let mut active = vec![0f32; w];
    for &(rb0, rb1) in &runs {
        let (xs, xe) = (rb0 * 8, rb1 * 8);
        if !matches!(stage, Stage::Two) {
            for (k, &(dy, dx)) in offs.iter().enumerate() {
                for r in 0..drows {
                    let y = (y0 + r + PAD - 1) as isize;
                    let out = &mut diff[(k * drows + r) * dw..(k * drows + r + 1) * dw];
                    let base = (y * pw as isize + PAD as isize - 1) as usize;
                    let nb = (base as isize + dy * pw as isize + dx) as usize;
                    let (a0, b0) = (&padded_src[0][base..base + dw], &padded_src[0][nb..nb + dw]);
                    let (a1, b1) = (&padded_src[1][base..base + dw], &padded_src[1][nb..nb + dw]);
                    let (a2, b2) = (&padded_src[2][base..base + dw], &padded_src[2][nb..nb + dw]);
                    for x in xs..xe + 2 {
                        let t0 = (a0[x] - b0[x]).abs() * CHANNEL_SCALE[0];
                        let t1 = (a1[x] - b1[x]).abs() * CHANNEL_SCALE[1];
                        let t2 = (a2[x] - b2[x]).abs() * CHANNEL_SCALE[2];
                        out[x] = t0 + t1 + t2;
                    }
                }
            }
        }
        for r in 0..rows {
            let y = y0 + r;
            let border_row = r == 0 || r == 7;
            for (k, (ip, ac)) in isig_px[xs..xe]
                .chunks_exact_mut(8)
                .zip(active[xs..xe].chunks_exact_mut(8))
                .enumerate()
            {
                let isig = sigmas[rb0 + k];
                if isig >= MIN_SIGMA {
                    ac.fill(1.0);
                    ip.fill(isig * if border_row { bsm } else { sm });
                    ip[0] = isig * bsm;
                    ip[7] = isig * bsm;
                } else {
                    ac.fill(0.0);
                    ip.fill(0.0);
                }
            }
            let row = (y + PAD) * pw + PAD;
            // Work in tiles so that the accumulators and every stream they
            // read stay in the L1 cache across the candidate offsets.
            for x0 in (xs..xe).step_by(TILE) {
                let tw = TILE.min(xe - x0);
                let (ip, ac) = (&isig_px[x0..x0 + tw], &active[x0..x0 + tw]);
                let mut ws = [1f32; TILE];
                let mut a0 = [0f32; TILE];
                let mut a1 = [0f32; TILE];
                let mut a2 = [0f32; TILE];
                a0[..tw].copy_from_slice(&padded_src[0][row + x0..row + x0 + tw]);
                a1[..tw].copy_from_slice(&padded_src[1][row + x0..row + x0 + tw]);
                a2[..tw].copy_from_slice(&padded_src[2][row + x0..row + x0 + tw]);
                let (ws, a0, a1, a2) = (&mut ws[..tw], &mut a0[..tw], &mut a1[..tw], &mut a2[..tw]);
                for (k, &(dy, dx)) in offs.iter().enumerate() {
                    let nb = (row as isize + dy * pw as isize + dx) as usize + x0;
                    let (p0, p1, p2) = (
                        &padded_src[0][nb..nb + tw],
                        &padded_src[1][nb..nb + tw],
                        &padded_src[2][nb..nb + tw],
                    );
                    if matches!(stage, Stage::Two) {
                        let (m0, m1, m2) = (
                            &padded_src[0][row + x0..row + x0 + tw],
                            &padded_src[1][row + x0..row + x0 + tw],
                            &padded_src[2][row + x0..row + x0 + tw],
                        );
                        for x in 0..tw {
                            let t0 = (p0[x] - m0[x]).abs() * CHANNEL_SCALE[0];
                            let t1 = (p1[x] - m1[x]).abs() * CHANNEL_SCALE[1];
                            let t2 = (p2[x] - m2[x]).abs() * CHANNEL_SCALE[2];
                            let sad = t0 + t1 + t2;
                            let wt = (sad * ip[x] + 1.0).max(0.0) * ac[x];
                            ws[x] += wt;
                            a0[x] += wt * p0[x];
                            a1[x] += wt * p1[x];
                            a2[x] += wt * p2[x];
                        }
                    } else {
                        // Plus-shaped sum of the difference plane around the pixel.
                        let d = &diff[k * drows * dw..(k + 1) * drows * dw];
                        let line =
                            |l: usize, off: usize| &d[l * dw + off + x0..l * dw + off + x0 + tw];
                        let (c, up, down) = (line(r + 1, 1), line(r, 1), line(r + 2, 1));
                        let (left, right) = (line(r + 1, 0), line(r + 1, 2));
                        for x in 0..tw {
                            let sad = c[x] + up[x] + down[x] + left[x] + right[x];
                            let wt = (sad * ip[x] + 1.0).max(0.0) * ac[x];
                            ws[x] += wt;
                            a0[x] += wt * p0[x];
                            a1[x] += wt * p1[x];
                            a2[x] += wt * p2[x];
                        }
                    }
                }
                let (o0, o1, o2) = (
                    &mut o0[r * w + x0..r * w + x0 + tw],
                    &mut o1[r * w + x0..r * w + x0 + tw],
                    &mut o2[r * w + x0..r * w + x0 + tw],
                );
                for x in 0..tw {
                    o0[x] = a0[x] / ws[x];
                    o1[x] = a1[x] / ws[x];
                    o2[x] = a2[x] / ws[x];
                }
            }
        }
    }
}

/// Source of the original XYB for a range of block rows.
pub type OrigRows<'a> = dyn Fn(usize, usize) -> [Vec<f32>; 3] + Sync + 'a;
/// Source of the decoded (pre-Gaborish) image: fills a row of one channel.
pub type RecRows<'a> = dyn Fn(usize, usize, &mut [f32]) + Sync + 'a;

/// Inputs of the selection.
pub struct Input<'a> {
    /// Original (pre-Gaborish) XYB of the block rows `by0..by1`, as three
    /// planes of stride `w` (so that no full-size copy has to be kept).
    pub orig: &'a OrigRows<'a>,
    /// Perceptual masking per pixel.
    pub mask1x1: &'a [f32],
    /// Per-block `quant_scale * hf_mul` (as in the decoder's sigma).
    pub quant: &'a [f32],
    pub w: usize,
    pub h: usize,
    pub distance: f32,
    pub iters: u32,
    pub gaborish: bool,
}

/// Masked error of the block rows `rows` (local to `rec`, which starts at
/// image row `y_off`) against the original.
fn block_errors(
    inp: &Input,
    rec: &Planes,
    orig: &[Vec<f32>; 3],
    y_off: isize,
    orig_y0: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    const KW: [f32; 3] = [12.339_445, 1.0, 0.2];
    let (w, bw) = (inp.w, inp.w / 8);
    let mut out = Vec::with_capacity(rows.len() * bw);
    let mut err: Vec<[f32; 3]> = vec![[0f32; 3]; bw];
    for br in rows {
        err.fill([0f32; 3]);
        for ly in br * 8..br * 8 + 8 {
            let gy = (y_off + ly as isize) as usize;
            let mask = &inp.mask1x1[gy * w..(gy + 1) * w];
            let oy = gy - orig_y0;
            let o: [&[f32]; 3] = std::array::from_fn(|c| &orig[c][oy * w..(oy + 1) * w]);
            let r: [&[f32]; 3] = std::array::from_fn(|c| &rec.p[c][ly * w..(ly + 1) * w]);
            // Each block keeps its own running sums, added in raster order, so
            // the blocks of a row are independent and can overlap.
            for (bx, e) in err.iter_mut().enumerate() {
                let x0 = bx * 8;
                for x in x0..x0 + 8 {
                    let m2 = mask[x] * mask[x];
                    for c in 0..3 {
                        let d = o[c][x] - r[c][x];
                        e[c] += m2 * d * d;
                    }
                }
            }
        }
        out.extend(
            err.iter()
                .map(|e| KW[0] * e[0] + KW[1] * e[1] + KW[2] * e[2]),
        );
    }
    out
}

/// Block rows per strip. Each strip is decoded on its own with a halo of one
/// block row (the filters reach at most 7 pixels), which bounds the working
/// memory by the strip size instead of the image size.
const STRIP_BLOCK_ROWS: usize = 16;
/// Lower bound of the strip height when splitting finer for more threads.
const MIN_STRIP_BLOCK_ROWS: usize = 4;

/// Decode block rows `by0..by1` under every candidate strength and return the
/// masked error of each block, one vector per candidate.
fn strip_errors(
    inp: &Input,
    rec: &RecRows,
    steps: &[usize],
    by0: usize,
    by1: usize,
) -> Vec<Vec<f32>> {
    let (w, h, bw, bh) = (inp.w, inp.h, inp.w / 8, inp.h / 8);
    // Local rows cover block rows by0-1 .. by1+1, mirrored at the image edges.
    let lo = by0 as isize - 1;
    let local_bh = by1 - by0 + 2;
    let mut strip = Planes {
        w,
        h: local_bh * 8,
        p: std::array::from_fn(|_| vec![0f32; w * local_bh * 8]),
    };
    for c in 0..3 {
        for ly in 0..local_bh * 8 {
            let gy = mirror(lo * 8 + ly as isize, h);
            rec(c, gy, &mut strip.p[c][ly * w..(ly + 1) * w]);
        }
    }
    if inp.gaborish {
        gaborish(&mut strip, 1);
    }
    let interior = 1..local_bh - 1;
    let orig = (inp.orig)(by0, by1);
    let orig_y0 = by0 * 8;
    let skip = flat_neighbourhoods(&strip, local_bh, bw);
    let y_off = by0 as isize * 8 - 8;
    let quant: Vec<f32> = (0..local_bh)
        .flat_map(|lb| {
            let gb = mirror(lo + lb as isize, bh);
            inp.quant[gb * bw..(gb + 1) * bw].iter().copied()
        })
        .collect();
    let mut stages = Vec::new();
    if inp.iters >= 3 {
        stages.push(Stage::Zero);
    }
    if inp.iters >= 1 {
        stages.push(Stage::One);
    }
    if inp.iters >= 2 {
        stages.push(Stage::Two);
    }
    let mut cur = Planes {
        w,
        h: strip.h,
        p: std::array::from_fn(|_| vec![0f32; w * strip.h]),
    };
    let mut tmp = Planes {
        w,
        h: strip.h,
        p: std::array::from_fn(|_| vec![0f32; w * strip.h]),
    };
    let mut result = Vec::with_capacity(steps.len());
    for &val in steps {
        if val == 0 {
            // Zero strength leaves every block unfiltered.
            result.push(block_errors(
                inp,
                &strip,
                &orig,
                y_off,
                orig_y0,
                interior.clone(),
            ));
            continue;
        }
        let inv_sigma: Vec<f32> = quant
            .iter()
            .zip(&skip)
            .map(|(&q, &flat)| {
                if flat {
                    // The filter leaves a flat neighbourhood unchanged.
                    return -1e9;
                }
                let sigma = QUANT_MUL / (q * INV_SIGMA_NUM) * (val as f32 / 7.0);
                1.0 / sigma.min(-1e-4)
            })
            .collect();
        for (i, &st) in stages.iter().enumerate() {
            let src = if i == 0 { &strip } else { &cur };
            run_stage(src, &mut tmp, &inv_sigma, st, 1);
            std::mem::swap(&mut cur, &mut tmp);
        }
        let decoded = if stages.is_empty() { &strip } else { &cur };
        let mut errs = block_errors(inp, decoded, &orig, y_off, orig_y0, interior.clone());
        // Unfiltered blocks have the error of strength zero.
        if let Some(zero) = result.first() {
            for (i, e) in errs.iter_mut().enumerate() {
                if skip[bw + i] {
                    *e = zero[i];
                }
            }
        }
        result.push(errs);
    }
    result
}

/// Per 8x8 block of `strip`: whether the block and its eight neighbours are
/// all one colour, so that no EPF stage (total reach below 8 pixels) can
/// change it. Edge block rows are never flagged.
fn flat_neighbourhoods(strip: &Planes, bh: usize, bw: usize) -> Vec<bool> {
    const EPS: f32 = 1e-5;
    let w = strip.w;
    let mut colour: Vec<Option<[f32; 3]>> = vec![None; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let mut c = [0f32; 3];
            let mut flat = true;
            for (ch, cv) in c.iter_mut().enumerate() {
                let first = strip.p[ch][by * 8 * w + bx * 8];
                *cv = first;
                for y in by * 8..by * 8 + 8 {
                    let row = &strip.p[ch][y * w + bx * 8..y * w + bx * 8 + 8];
                    flat &= row.iter().all(|&v| (v - first).abs() <= EPS);
                }
            }
            if flat {
                colour[by * bw + bx] = Some(c);
            }
        }
    }
    let mut skip = vec![false; bw * bh];
    for by in 1..bh.saturating_sub(1) {
        for bx in 0..bw {
            let Some(c) = colour[by * bw + bx] else {
                continue;
            };
            skip[by * bw + bx] = (by - 1..=by + 1).all(|y| {
                (bx.saturating_sub(1)..=(bx + 1).min(bw - 1)).all(|x| {
                    colour[y * bw + x].is_some_and(|n| (0..3).all(|k| (n[k] - c[k]).abs() <= EPS))
                })
            });
        }
    }
    skip
}

/// Choose an EPF sharpness (0..=7) for each 8x8 block. `rec` is the decoded
/// image before Gaborish.
pub fn choose(inp: &Input, rec: &RecRows, threads: usize) -> Vec<u8> {
    let (bw, bh) = (inp.w / 8, inp.h / 8);
    let steps: &[usize] = if inp.distance > 4.5 {
        &[0, 4]
    } else {
        &[0, 2, 7]
    };
    let mut lut = [0usize; 8];
    for (i, &s) in steps.iter().enumerate() {
        lut[s] = i;
    }
    // Shorter strips give small images enough of them to keep every thread
    // busy, at the price of a larger share of halo rows; the result does not
    // depend on the strip height.
    let strip_rows = bh
        .div_ceil(2 * threads.max(1))
        .clamp(MIN_STRIP_BLOCK_ROWS, STRIP_BLOCK_ROWS);
    let strips = bh.div_ceil(strip_rows);
    let parts = par_map(strips, threads, |i| {
        let by0 = i * strip_rows;
        strip_errors(inp, rec, steps, by0, (by0 + strip_rows).min(bh))
    });
    let mut errors: [Vec<f32>; 8] = Default::default();
    for (k, &val) in steps.iter().enumerate() {
        errors[val] = parts.iter().flat_map(|p| p[k].iter().copied()).collect();
    }

    // First pass: pick the best strength, preferring neighbours' values.
    let mut sharp = vec![0u8; bw * bh];
    let mut histo = vec![[0usize; 8]; 9];
    let mut totals = [1usize; 9];
    for by in 0..bh {
        for bx in 0..bw {
            let top = if by > 0 { sharp[(by - 1) * bw + bx] } else { 0 };
            let left = if bx > 0 { sharp[by * bw + bx - 1] } else { 0 };
            let at = |v: u8| errors[v as usize].get(by * bw + bx).copied();
            // Errors of values that are not candidates are never read: the
            // neighbours always hold candidates.
            let (top_err, left_err) = (at(top).unwrap(), at(left).unwrap());
            let (mut best, mut best_err) = (0u8, f32::MAX);
            for &v in steps {
                let mut e = errors[v][by * bw + bx];
                if v == 0 {
                    e *= 0.99;
                }
                if e < best_err {
                    best = v as u8;
                    best_err = e;
                }
            }
            let chosen = if best_err < top_err.min(left_err) {
                best
            } else if top_err < left_err {
                top
            } else {
                left
            };
            sharp[by * bw + bx] = chosen;
            let ctx = lut[top as usize] * 3 + lut[left as usize];
            histo[ctx][chosen as usize] += 1;
            totals[ctx] += 1;
        }
    }
    // Second pass: weigh by how often each strength follows the context.
    let clamped = inp.distance.min(5.0);
    let c3 = 0.859_703_4f32.max(0.980_171_99f32.powf(clamped));
    let c5 = 0.108_769_04f32;
    let mut mul = [0f32; 27];
    for &t in steps {
        for &l in steps {
            for &v in steps {
                let ctx = lut[t] * 3 + lut[l];
                let i = lut[v] + 3 * ctx;
                // libjxl divides two integers here, so the ratio is zero.
                let ratio = (histo[ctx][v] / totals[ctx]) as f32;
                mul[i] = 1.0 / (1.0 + c5 * ratio.ln_1p() / clamped);
                if v == 0 {
                    mul[i] *= c3;
                }
            }
        }
    }
    for by in 0..bh {
        for bx in 0..bw {
            let top = if by > 0 { sharp[(by - 1) * bw + bx] } else { 0 };
            let left = if bx > 0 { sharp[by * bw + bx - 1] } else { 0 };
            let ctx = lut[top as usize] * 3 + lut[left as usize];
            let (mut best, mut best_err) = (0u8, f32::MAX);
            for &v in steps {
                let e = errors[v][by * bw + bx] * mul[lut[v] + 3 * ctx];
                if e < best_err {
                    best = v as u8;
                    best_err = e;
                }
            }
            sharp[by * bw + bx] = best;
        }
    }
    sharp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, v: [f32; 3]) -> Planes {
        Planes {
            w,
            h,
            p: v.map(|x| vec![x; w * h]),
        }
    }

    #[test]
    fn filters_keep_flat_images_unchanged() {
        let mut a = flat(24, 16, [0.01, 0.4, 0.3]);
        gaborish(&mut a, 2);
        let mut b = flat(24, 16, [0.0; 3]);
        let inv_sigma = vec![-1.0f32; 3 * 2];
        for stage in [Stage::Zero, Stage::One, Stage::Two] {
            run_stage(&a, &mut b, &inv_sigma, stage, 2);
            for c in 0..3 {
                for (x, y) in b.p[c].iter().zip(&a.p[c]) {
                    assert!((x - y).abs() < 1e-5, "{x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn choice_is_deterministic_and_in_range() {
        let (w, h) = (40, 24);
        let mut rec = flat(w, h, [0.0, 0.3, 0.2]);
        let mut orig = flat(w, h, [0.0, 0.3, 0.2]);
        for i in 0..w * h {
            let n = ((i * 2654435761usize) >> 7) % 13;
            rec.p[1][i] += n as f32 * 0.002;
            orig.p[1][i] += (n % 5) as f32 * 0.002;
        }
        let mask = vec![1.0f32; w * h];
        let quant = vec![0.5f32; (w / 8) * (h / 8)];
        for (distance, iters, steps) in [(2.0f32, 2u32, &[0u8, 2, 7][..]), (6.0, 3, &[0, 4][..])] {
            let orig_rows = |by0: usize, by1: usize| -> [Vec<f32>; 3] {
                std::array::from_fn(|c| orig.p[c][by0 * 8 * w..by1 * 8 * w].to_vec())
            };
            let input = Input {
                orig: &orig_rows,
                mask1x1: &mask,
                quant: &quant,
                w,
                h,
                distance,
                iters,
                gaborish: true,
            };
            let rec_rows = |c: usize, y: usize, dst: &mut [f32]| {
                dst.copy_from_slice(&rec.p[c][y * w..(y + 1) * w]);
            };
            let a = choose(&input, &rec_rows, 1);
            let b = choose(&input, &rec_rows, 3);
            assert_eq!(a, b);
            assert_eq!(a.len(), (w / 8) * (h / 8));
            assert!(a.iter().all(|v| steps.contains(v)));
        }
    }
}
