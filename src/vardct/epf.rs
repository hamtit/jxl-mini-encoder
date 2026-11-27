//! Per-block EPF sharpness selection (libjxl's `ComputeARHeuristics`).
//!
//! The decoder smooths each 8x8 block with an edge-preserving filter whose
//! strength is `sharpness / 7 * f(quant)`. libjxl (effort >= 6) reconstructs
//! the image under a few candidate strengths, compares each block with the
//! original and keeps the strength that gives the smallest masked error. This
//! module reproduces that: Gaborish, the three EPF stages and the selection
//! rule, working on whole planes padded to a multiple of 8.

use crate::bit_writer::unpack_f16;
use crate::parallel::{par_map, par_tasks};

/// Inverse sigma below which a block is left unfiltered.
const MIN_SIGMA: f32 = -3.905_243;
const INV_SIGMA_NUM: f32 = -1.171_572_9;
const QUANT_MUL: f32 = 0.46;
const CHANNEL_SCALE: [f32; 3] = [40.18, 3.487, 3.016];
const BORDER_SAD_MUL: f32 = 2.0 / 3.0;
const PAD: usize = 3;
/// Pixels per tile in the filter passes.
const TILE: usize = 64;

/// Weights of the decoder's 3x3 Gaborish (default header, same for X, Y, B).
const GAB_W1: f32 = 1.1 * 0.104_699_57;
const GAB_W2: f32 = 1.1 * 0.055_680_54;

/// Three XYB planes stored with `PAD` mirrored border pixels on every side,
/// so Gaborish and all EPF stages read neighbours directly without copying.
pub struct Planes {
    pub w: usize,
    pub h: usize,
    pub pw: usize,
    pub p: [Vec<f32>; 3],
}

impl Planes {
    fn new(w: usize, h: usize) -> Self {
        let pw = w + 2 * PAD;
        let len = pw * (h + 2 * PAD);
        Planes { w, h, pw, p: std::array::from_fn(|_| vec![0f32; len]) }
    }

    #[inline]
    fn row_mut(&mut self, c: usize, y: usize) -> &mut [f32] {
        let start = (y + PAD) * self.pw + PAD;
        &mut self.p[c][start..start + self.w]
    }

    /// Mirror `PAD` pixels on the left, right, top, and bottom borders in place.
    fn fill_borders(&mut self) {
        let (w, h, pw) = (self.w, self.h, self.pw);
        for plane in &mut self.p {
            // Left and right borders of the interior rows.
            for y in 0..h {
                let row = &mut plane[(y + PAD) * pw..(y + PAD + 1) * pw];
                for i in 0..PAD {
                    row[PAD - 1 - i] = row[PAD + mirror(i as isize, w)];
                    row[PAD + w + i] = row[PAD + mirror((w + i) as isize, w)];
                }
            }
            // Top and bottom border rows (mirror interior rows).
            for i in 0..PAD {
                let sy = PAD + mirror(-(i as isize) - 1, h);
                let dy = PAD - 1 - i;
                plane.copy_within(sy * pw..(sy + 1) * pw, dy * pw);

                let sy = PAD + mirror((h + i) as isize, h);
                let dy = PAD + h + i;
                plane.copy_within(sy * pw..(sy + 1) * pw, dy * pw);
            }
        }
    }
}

fn mirror(v: isize, len: usize) -> usize {
    let len = len as isize;
    let v = if v < 0 { -v - 1 } else { v };
    (if v >= len { 2 * len - 1 - v } else { v }) as usize
}

#[inline(always)]
fn gaborish_row_impl(up: &[f32], mid: &[f32], down: &[f32], row: &mut [f32], wc: f32, we: f32, wd: f32) {
    let w = row.len();
    let (up, mid, down) = (&up[..w + 2], &mid[..w + 2], &down[..w + 2]);
    for x in 0..w {
        let edges = mid[x] + mid[x + 2] + up[x + 1] + down[x + 1];
        let diags = up[x] + up[x + 2] + down[x] + down[x + 2];
        row[x] = wc * mid[x + 1] + we * edges + wd * diags;
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gaborish_row_simd(up: &[f32], mid: &[f32], down: &[f32], row: &mut [f32], wc: f32, we: f32, wd: f32) {
    gaborish_row_impl(up, mid, down, row, wc, we, wd);
}

#[inline]
fn gaborish_row(simd: bool, up: &[f32], mid: &[f32], down: &[f32], row: &mut [f32], wc: f32, we: f32, wd: f32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if simd {
        // SAFETY: AVX2+FMA (or NEON) support was detected at runtime.
        return unsafe { gaborish_row_simd(up, mid, down, row, wc, we, wd) };
    }
    let _ = simd;
    gaborish_row_impl(up, mid, down, row, wc, we, wd);
}

/// The decoder's Gaborish smoothing from padded `src` into padded `dst`.
pub fn gaborish(src: &Planes, dst: &mut Planes, threads: usize) {
    let (w, h, pw) = (src.w, src.h, src.pw);
    let n = 1.0 / (1.0 + 4.0 * (GAB_W1 + GAB_W2));
    let (wc, we, wd) = (n, n * GAB_W1, n * GAB_W2);
    let simd = crate::simd_available();
    for (sp, dp) in src.p.iter().zip(&mut dst.p) {
        let interior = &mut dp[PAD * pw..(PAD + h) * pw];
        if threads <= 1 {
            for (y, drow) in interior.chunks_mut(pw).enumerate() {
                let base = (y + PAD) * pw + PAD;
                let up = &sp[base - pw - 1..base - pw + w + 1];
                let mid = &sp[base - 1..base + w + 1];
                let down = &sp[base + pw - 1..base + pw + w + 1];
                gaborish_row(simd, up, mid, down, &mut drow[PAD..PAD + w], wc, we, wd);
            }
        } else {
            let tasks: Vec<_> = interior.chunks_mut(pw).enumerate().collect();
            par_tasks(tasks, threads, |(y, drow)| {
                let base = (y + PAD) * pw + PAD;
                let up = &sp[base - pw - 1..base - pw + w + 1];
                let mid = &sp[base - 1..base + w + 1];
                let down = &sp[base + pw - 1..base + pw + w + 1];
                gaborish_row(simd, up, mid, down, &mut drow[PAD..PAD + w], wc, we, wd);
            });
        }
    }
    dst.fill_borders();
}

#[derive(Clone, Copy)]
enum Stage {
    Zero,
    One,
    Two,
}

const OFF_0: [(isize, isize); 12] =
    [(-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)];
const OFF_1: [(isize, isize); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

/// Reusable per-band scratch buffers, allocated once per strip instead of per band.
#[derive(Default)]
struct BandScratch {
    runs: Vec<(usize, usize)>,
    diff: Vec<f32>,
    isig_px: Vec<f32>,
    active: Vec<f32>,
}

/// One EPF pass from padded `src` into padded `dst`. `inv_sigma` holds one value per 8x8 block.
///
/// Works on bands of one block row. For stages 0 and 1 the SAD of a candidate
/// offset is the plus-shaped sum of per-pixel channel differences, so those
/// differences are computed once per band and offset.
fn run_stage(
    src: &Planes,
    dst: &mut Planes,
    inv_sigma: &[f32],
    stage: Stage,
    threads: usize,
    scratch: &mut BandScratch,
) {
    let (w, h, pw) = (src.w, src.h, src.pw);
    let bw = w / 8;
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
    let (i0, i1, i2) =
        (&mut d0[PAD * pw..(PAD + h) * pw], &mut d1[PAD * pw..(PAD + h) * pw], &mut d2[PAD * pw..(PAD + h) * pw]);
    let ctx = StageCtx { w, bw, pw, padded_src: &src.p, inv_sigma, stage, offs, sm, bsm };
    let simd = crate::simd_available();
    if threads <= 1 {
        for (band, ((o0, o1), o2)) in
            i0.chunks_mut(pw * 8).zip(i1.chunks_mut(pw * 8)).zip(i2.chunks_mut(pw * 8)).enumerate()
        {
            run_band(&ctx, band, o0, o1, o2, simd, scratch);
        }
    } else {
        let tasks: Vec<_> =
            i0.chunks_mut(pw * 8).zip(i1.chunks_mut(pw * 8)).zip(i2.chunks_mut(pw * 8)).enumerate().collect();
        par_tasks(tasks, threads, |(band, ((o0, o1), o2))| {
            let mut local = BandScratch::default();
            run_band(&ctx, band, o0, o1, o2, simd, &mut local);
        });
    }
    dst.fill_borders();
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
    simd: bool,
    scratch: &mut BandScratch,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if simd {
        // SAFETY: AVX2+FMA (or NEON) support was detected at runtime.
        return unsafe { run_band_simd(ctx, band, o0, o1, o2, scratch) };
    }
    let _ = simd;
    run_band_impl(ctx, band, o0, o1, o2, scratch)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn run_band_simd(
    ctx: &StageCtx,
    band: usize,
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
    scratch: &mut BandScratch,
) {
    run_band_impl(ctx, band, o0, o1, o2, scratch)
}

#[inline(always)]
fn run_band_impl(
    ctx: &StageCtx,
    band: usize,
    o0: &mut [f32],
    o1: &mut [f32],
    o2: &mut [f32],
    scratch: &mut BandScratch,
) {
    let StageCtx { w, bw, pw, padded_src, inv_sigma, stage, offs, sm, bsm } = *ctx;
    let y0 = band * 8;
    let rows = o0.len() / pw;
    let sigmas = &inv_sigma[band * bw..(band + 1) * bw];
    if sigmas.iter().all(|&v| v < MIN_SIGMA) {
        for (c, out) in [&mut *o0, &mut *o1, &mut *o2].into_iter().enumerate() {
            for r in 0..rows {
                let from = (y0 + r + PAD) * pw + PAD;
                let dst = r * pw + PAD;
                out[dst..dst + w].copy_from_slice(&padded_src[c][from..from + w]);
            }
        }
        return;
    }
    // Blocks that are not filtered are copied; the rest is processed in
    // runs of consecutive active blocks.
    for (c, out) in [&mut *o0, &mut *o1, &mut *o2].into_iter().enumerate() {
        for r in 0..rows {
            let from = (y0 + r + PAD) * pw + PAD;
            let dst = r * pw + PAD;
            out[dst..dst + w].copy_from_slice(&padded_src[c][from..from + w]);
        }
    }
    scratch.runs.clear();
    for (bx, &v) in sigmas.iter().enumerate() {
        if v < MIN_SIGMA {
            continue;
        }
        match scratch.runs.last_mut() {
            Some(last) if last.1 == bx => last.1 = bx + 1,
            _ => scratch.runs.push((bx, bx + 1)),
        }
    }
    // Per-offset channel differences for rows y0-1..=y0+8, x -1..=w.
    let (dw, drows) = (w + 2, rows + 2);
    if !matches!(stage, Stage::Two) {
        let need = offs.len() * drows * dw;
        if scratch.diff.len() < need {
            scratch.diff.resize(need, 0.0);
        }
    }
    if scratch.isig_px.len() < w {
        scratch.isig_px.resize(w, 0.0);
        scratch.active.resize(w, 0.0);
    }
    let diff = &mut scratch.diff;
    let isig_px = &mut scratch.isig_px[..w];
    let active = &mut scratch.active[..w];
    for &(rb0, rb1) in &scratch.runs {
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
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(active[xs..xe].as_chunks_mut::<8>().0.iter_mut())
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
                    let (p0, p1, p2) =
                        (&padded_src[0][nb..nb + tw], &padded_src[1][nb..nb + tw], &padded_src[2][nb..nb + tw]);
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
                        let line = |l: usize, off: usize| &d[l * dw + off + x0..l * dw + off + x0 + tw];
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
                let dst = r * pw + PAD + x0;
                let (o0, o1, o2) = (&mut o0[dst..dst + tw], &mut o1[dst..dst + tw], &mut o2[dst..dst + tw]);
                for x in 0..tw {
                    let inv_w = 1.0 / ws[x];
                    o0[x] = a0[x] * inv_w;
                    o1[x] = a1[x] * inv_w;
                    o2[x] = a2[x] * inv_w;
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
    /// Perceptual masking per pixel (f16-packed).
    pub mask1x1: &'a [u16],
    /// Per-block `quant_scale * hf_mul` (as in the decoder's sigma).
    pub quant: &'a [f32],
    pub w: usize,
    pub h: usize,
    pub distance: f32,
    pub iters: u32,
    pub gaborish: bool,
}

#[inline(always)]
fn block_errors_impl(
    inp: &Input,
    rec: &Planes,
    orig: &[Vec<f32>; 3],
    y_off: isize,
    orig_y0: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    const KW: [f32; 3] = [12.339_445, 1.0, 0.2];
    let (w, bw, pw) = (inp.w, inp.w / 8, rec.pw);
    let mut out = Vec::with_capacity(rows.len() * bw);
    let mut err: Vec<[f32; 3]> = vec![[0f32; 3]; bw];
    for br in rows {
        err.fill([0f32; 3]);
        for ly in br * 8..br * 8 + 8 {
            let gy = (y_off + ly as isize) as usize;
            let mask = &inp.mask1x1[gy * w..(gy + 1) * w];
            let oy = gy - orig_y0;
            let o: [&[f32]; 3] = std::array::from_fn(|c| &orig[c][oy * w..(oy + 1) * w]);
            let rbase = (ly + PAD) * pw + PAD;
            let r: [&[f32]; 3] = std::array::from_fn(|c| &rec.p[c][rbase..rbase + w]);
            // Each block keeps its own running sums, added in raster order, so
            // the blocks of a row are independent and can overlap.
            for (bx, e) in err.iter_mut().enumerate() {
                let x0 = bx * 8;
                for x in x0..x0 + 8 {
                    let m = unpack_f16(mask[x]);
                    let m2 = m * m;
                    let d0 = o[0][x] - r[0][x];
                    let d1 = o[1][x] - r[1][x];
                    let d2 = o[2][x] - r[2][x];
                    e[0] += m2 * d0 * d0;
                    e[1] += m2 * d1 * d1;
                    e[2] += m2 * d2 * d2;
                }
            }
        }
        out.extend(err.iter().map(|e| KW[0] * e[0] + KW[1] * e[1] + KW[2] * e[2]));
    }
    out
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn block_errors_simd(
    inp: &Input,
    rec: &Planes,
    orig: &[Vec<f32>; 3],
    y_off: isize,
    orig_y0: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    block_errors_impl(inp, rec, orig, y_off, orig_y0, rows)
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
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if crate::simd_available() {
        // SAFETY: AVX2+FMA (or NEON) support was detected at runtime.
        return unsafe { block_errors_simd(inp, rec, orig, y_off, orig_y0, rows) };
    }
    block_errors_impl(inp, rec, orig, y_off, orig_y0, rows)
}

/// Block rows per strip. Each strip is decoded on its own with a halo of one
/// block row (the filters reach at most 7 pixels), which bounds the working
/// memory by the strip size instead of the image size.
const STRIP_BLOCK_ROWS: usize = 16;
/// Lower bound of the strip height when splitting finer for more threads.
const MIN_STRIP_BLOCK_ROWS: usize = 4;

/// Decode block rows `by0..by1` under every candidate strength and return the
/// masked error of each block, one vector per candidate.
fn strip_errors(inp: &Input, rec: &RecRows, steps: &[usize], by0: usize, by1: usize) -> Vec<Vec<f32>> {
    let (w, h, bw, bh) = (inp.w, inp.h, inp.w / 8, inp.h / 8);
    // Local rows cover block rows by0-1 .. by1+1, mirrored at the image edges.
    let lo = by0 as isize - 1;
    let local_bh = by1 - by0 + 2;
    let mut strip = Planes::new(w, local_bh * 8);
    for c in 0..3 {
        for ly in 0..local_bh * 8 {
            let gy = mirror(lo * 8 + ly as isize, h);
            rec(c, gy, strip.row_mut(c, ly));
        }
    }
    strip.fill_borders();
    let mut cur = Planes::new(w, strip.h);
    let mut tmp = Planes::new(w, strip.h);
    if inp.gaborish {
        gaborish(&strip, &mut tmp, 1);
        std::mem::swap(&mut strip, &mut tmp);
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
    let mut scratch = BandScratch::default();
    let mut result = Vec::with_capacity(steps.len());
    for &val in steps {
        if val == 0 {
            // Zero strength leaves every block unfiltered.
            result.push(block_errors(inp, &strip, &orig, y_off, orig_y0, interior.clone()));
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
            run_stage(src, &mut tmp, &inv_sigma, st, 1, &mut scratch);
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
    let pw = strip.pw;
    let mut colour: Vec<Option<[f32; 3]>> = vec![None; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let mut c = [0f32; 3];
            let mut flat = true;
            for (ch, cv) in c.iter_mut().enumerate() {
                let first = strip.p[ch][(by * 8 + PAD) * pw + bx * 8 + PAD];
                *cv = first;
                for y in by * 8..by * 8 + 8 {
                    let base = (y + PAD) * pw + bx * 8 + PAD;
                    let row = &strip.p[ch][base..base + 8];
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
                (bx.saturating_sub(1)..=(bx + 1).min(bw - 1))
                    .all(|x| colour[y * bw + x].is_some_and(|n| (0..3).all(|k| (n[k] - c[k]).abs() <= EPS)))
            });
        }
    }
    skip
}

/// Choose an EPF sharpness (0..=7) for each 8x8 block. `rec` is the decoded
/// image before Gaborish.
pub fn choose(inp: &Input, rec: &RecRows, threads: usize) -> Vec<u8> {
    let (bw, bh) = (inp.w / 8, inp.h / 8);
    let steps: &[usize] = if inp.distance > 4.5 { &[0, 4] } else { &[0, 2, 7] };
    let mut lut = [0usize; 8];
    for (i, &s) in steps.iter().enumerate() {
        lut[s] = i;
    }
    // Shorter strips give small images enough of them to keep every thread
    // busy, at the price of a larger share of halo rows; the result does not
    // depend on the strip height.
    let strip_rows = bh.div_ceil(2 * threads.max(1)).clamp(MIN_STRIP_BLOCK_ROWS, STRIP_BLOCK_ROWS);
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
                let ratio = histo[ctx][v] as f32 / totals[ctx] as f32;
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
    use crate::bit_writer::pack_f16;

    fn flat(w: usize, h: usize, v: [f32; 3]) -> Planes {
        let mut pl = Planes::new(w, h);
        for c in 0..3 {
            pl.p[c].fill(v[c]);
        }
        pl
    }

    #[test]
    fn filters_keep_flat_images_unchanged() {
        let a = flat(24, 16, [0.01, 0.4, 0.3]);
        let mut g = flat(24, 16, [0.0; 3]);
        gaborish(&a, &mut g, 2);
        let mut b = flat(24, 16, [0.0; 3]);
        let inv_sigma = vec![-1.0f32; 3 * 2];
        let mut scratch = BandScratch::default();
        for stage in [Stage::Zero, Stage::One, Stage::Two] {
            run_stage(&g, &mut b, &inv_sigma, stage, 2, &mut scratch);
            for c in 0..3 {
                for (x, y) in b.p[c].iter().zip(&g.p[c]) {
                    assert!((x - y).abs() < 1e-5, "{x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn choice_is_deterministic_and_in_range() {
        let (w, h) = (40, 24);
        let mut rec = [vec![0.0f32; w * h], vec![0.3f32; w * h], vec![0.2f32; w * h]];
        let mut orig = [vec![0.0f32; w * h], vec![0.3f32; w * h], vec![0.2f32; w * h]];
        for i in 0..w * h {
            let n = ((i * 2654435761usize) >> 7) % 13;
            rec[1][i] += n as f32 * 0.002;
            orig[1][i] += (n % 5) as f32 * 0.002;
        }
        let mask = vec![pack_f16(1.0f32); w * h];
        let quant = vec![0.5f32; (w / 8) * (h / 8)];
        for (distance, iters, steps) in [(2.0f32, 2u32, &[0u8, 2, 7][..]), (6.0, 3, &[0, 4][..])] {
            let orig_rows = |by0: usize, by1: usize| -> [Vec<f32>; 3] {
                std::array::from_fn(|c| orig[c][by0 * 8 * w..by1 * 8 * w].to_vec())
            };
            let input =
                Input { orig: &orig_rows, mask1x1: &mask, quant: &quant, w, h, distance, iters, gaborish: true };
            let rec_rows = |c: usize, y: usize, dst: &mut [f32]| {
                dst.copy_from_slice(&rec[c][y * w..(y + 1) * w]);
            };
            let a = choose(&input, &rec_rows, 1);
            let b = choose(&input, &rec_rows, 3);
            assert_eq!(a, b);
            assert_eq!(a.len(), (w / 8) * (h / 8));
            assert!(a.iter().all(|v| steps.contains(v)));
        }
    }
}
