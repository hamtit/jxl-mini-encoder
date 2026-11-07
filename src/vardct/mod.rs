//! Lossy encoding: XYB, DCT, quantisation and the VarDCT bitstream.

mod acs;
mod aq;
mod dct;
mod epf;
mod noise;
mod quant;
mod small;
#[cfg(test)]
mod tests;
pub(crate) mod xyb;

use crate::bit_writer::BitWriter;
use crate::bit_writer::U32Dist::{Bits, Val};
use crate::bit_writer::{ceil_log2, pack_f16, unpack_f16};
use crate::entropy::{CodeOptions, EntropyCode, Token, pack_signed};
use crate::headers::{Encoding, FrameInfo, FramePlacement, GROUP_DIM, Layout, write_frame_header, write_sections};
use crate::modular::{Channel, GlobalTree, ModularOptions, write_header, write_standalone};
use crate::parallel::par_map;
use crate::{EncoderOptions, ImageRef};
use acs::{Acs, BlockAcs, SearchInput, Tables};
use dct::Dct;
use quant::{Quantizer, dequant_bias};
use xyb::Xyb;

const NUM_BLOCK_CLUSTERS: u32 = 15;
const BLOCK_CTX_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];
const COEFF_FREQ_CONTEXT: [u16; 63] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30,
    30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [u16; 63] = [
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
];
const NUM_HF_CTX: usize = 495 * NUM_BLOCK_CLUSTERS as usize;

/// Distance-dependent encoder parameters.
struct Params {
    distance: f32,
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    adaptive_quant: bool,
    choose_transforms: bool,
    quant_dc: f32,
}

impl Params {
    fn new(distance: f32, effort: u32) -> Self {
        let mut x_qm_scale = 3;
        for step in [2.5, 5.5, 9.5] {
            if distance > step {
                x_qm_scale += 1;
            }
        }
        let epf_iters = [0.7, 1.5, 4.0].iter().filter(|&&t| distance >= t).count() as u32;
        Params {
            distance,
            x_qm_scale,
            b_qm_scale: 2,
            epf_iters,
            adaptive_quant: effort >= 5,
            choose_transforms: effort >= 5,
            quant_dc: initial_quant_dc(distance),
        }
    }
}

/// libjxl's pixel statistics for chromacity adjustment (effort >= 7): the
/// largest neighbour differences in X and B raise the X/B quantisation
/// multipliers on saturated, high-contrast content.
fn chroma_scales(xyb: &Xyb, w: usize, h: usize) -> (u32, u32) {
    let stride = xyb.w;
    let (px, py, pb) = (&xyb.planes[0], &xyb.planes[1], &xyb.planes[2]);
    let (mut dx, mut db, mut exposed) = (0f32, 0f32, 0f32);
    for y in 1..h {
        for x in 1..w {
            let i = y * stride + x;
            dx = dx.max((px[i] - px[i - 1]).abs()).max((px[i] - px[i - stride]).abs());
            let diff = pb[i] - py[i];
            let diff_prev = pb[i - 1] - py[i - 1];
            let diff_row = pb[i - stride] - py[i - stride];
            db = db.max((diff - diff_prev).abs()).max((diff - diff_row).abs());
            let exposed_b = pb[i] - py[i] * 1.2;
            if exposed_b >= 0.0 {
                let edge = (pb[i] - pb[i - 1]).abs() + (pb[i] - pb[i - stride]).abs();
                exposed = exposed.max(exposed_b * edge);
            }
        }
    }
    let x_extra = match dx {
        v if v >= 0.026 => 3,
        v if v >= 0.022 => 2,
        v if v >= 0.015 => 1,
        _ => 0,
    };
    let add = u32::from(exposed >= 0.13);
    let b_extra = match db {
        v if v > 0.38 => 2 + add,
        v if v > 0.33 => 1 + add,
        v if v > 0.28 => add,
        _ => 0,
    };
    (2 + x_extra, 2 + b_extra)
}

/// Estimated relative gain of the small transforms (see `probe_small`) that
/// is needed to enable them. Photos stay below it (<= 0.0085 for d >= 1.2,
/// <= 0.03 below) while graphics are above (>= 0.013, >= 0.07 below); at
/// high quality both estimates grow, hence the higher bar.
fn small_threshold(distance: f32) -> f32 {
    if distance < 1.2 { 0.045 } else { 0.012 }
}

/// Reduce EPF smoothing at medium/low fidelity when inverse Gaborish is
/// active. This preserves photographic texture without changing the AC data.
/// Keep the original strength for high fidelity and unsharpened graphics.
fn epf_sharpness(distance: f32, gaborish: bool) -> i32 {
    if gaborish && distance >= 1.5 { 3 } else { 4 }
}

/// Multiplier on the adaptive quantisation field, calibrated so that the
/// SSIMULACRA2 score at a given quality and effort stays close to libjxl's
/// (measured on photographs, 8 Kodak images, qualities 30-95).
///
/// Effort 5 already matches without it. For effort 6 and 7 libjxl's search
/// finds somewhat better transforms and filter strengths than this encoder,
/// so a slightly finer quantiser is needed at the same distance; the factor
/// fades towards high fidelity where the gap closes. From effort 8 libjxl
/// runs a Butteraugli-driven quantisation loop that yields smaller files at a
/// lower score, so a coarser quantiser tracks it.
fn aq_calibration(distance: f32, effort: u32) -> f32 {
    match effort {
        ..=5 => 1.0,
        6..=7 => 1.0 + 0.05 * ((distance - 0.8) / 2.4).clamp(0.0, 1.0),
        _ => 0.96,
    }
}

fn initial_quant_dc(distance: f32) -> f32 {
    const DC_MUL: f32 = 0.3;
    let d = (0.5 * distance).max(distance.min(DC_MUL * ((1.0 / DC_MUL) * distance).powf(0.83)));
    (1.095_924 / d).min(50.0)
}

/// Colour-correlation factors are coded in units of 1/84.
const CFL_FACTOR: f32 = 84.0;
/// Cost-proxy gain a tile needs before it leaves the default correlation.
const CFL_MIN_GAIN: f32 = 12.0;
/// Tiles with fewer non-zero Y coefficients keep the default correlation.
const CFL_MIN_COEFFS: usize = 16;

/// Cheap piecewise-linear log2 (exact at powers of two).
#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    x.to_bits() as f32 * (1.0 / 8_388_608.0) - 127.0
}

/// Sum of `fast_log2(1 + |(s - r y) * wm|)`, in eight interleaved lanes.
fn correlation_cost(avx2: bool, s: &[f32], y: &[f32], wm: &[f32], r: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: AVX2 support was detected at runtime.
        return unsafe { correlation_cost_avx2(s, y, wm, r) };
    }
    let _ = avx2;
    correlation_cost_impl(s, y, wm, r)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn correlation_cost_avx2(s: &[f32], y: &[f32], wm: &[f32], r: f32) -> f32 {
    correlation_cost_impl(s, y, wm, r)
}

#[inline(always)]
fn correlation_cost_impl(s: &[f32], y: &[f32], wm: &[f32], r: f32) -> f32 {
    let n = s.len();
    let (y, wm) = (&y[..n], &wm[..n]);
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 8 <= n {
        for j in 0..8 {
            let v = 1.0 + ((s[i + j] - r * y[i + j]) * wm[i + j]).abs();
            // v >= 1, so the bit pattern fits a signed int (cheaper to convert).
            acc[j] += (v.to_bits() as i32) as f32 * (1.0 / 8_388_608.0) - 127.0;
        }
        i += 8;
    }
    let mut total = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    while i < n {
        total += fast_log2(1.0 + ((s[i] - r * y[i]) * wm[i]).abs());
        i += 1;
    }
    total
}

/// A block whose HF coefficients are held back until its tile's colour
/// correlation is known.
struct Blk {
    bx: usize,
    by: usize,
    acs: Acs,
    qac: f32,
    off: usize,
}

/// Shared read-only state for coding groups.
struct Ctx<'a> {
    xyb: &'a Xyb,
    layout: Layout,
    bw: usize,
    quantizer: &'a Quantizer,
    acs: &'a [BlockAcs],
    tables: &'a Tables,
    qm_mult: [f32; 3],
    dct: &'a Dct,
    /// libjxl's per-block quantiser adjustment (effort >= 5).
    adjust: bool,
    /// Also reconstruct the decoded image (for the EPF strength choice).
    recon: bool,
}

/// Output of coding one group: HF tokens and the block LF values.
struct GroupOut {
    /// Colour correlation (X, B) of each 64x64 tile of the group, row-major.
    cmap: Vec<[i32; 2]>,
    tokens: Vec<Token>,
    /// (block x, block y, [X, Y, B] LF)
    dc: Vec<(usize, usize, [f32; 3])>,
    /// (block index in the whole image, new raw quantiser value)
    raw_updates: Vec<(usize, u8)>,
    /// Decoded X, Y, B of the group before the loop filters (gbw*8 wide).
    recon: [Vec<u16>; 3],
}

impl Ctx<'_> {
    fn code_group(&self, g: usize) -> GroupOut {
        let (gx, gy, gw, gh) = self.layout.group_rect(g);
        let (bx0, by0) = (gx / 8, gy / 8);
        let (gbw, gbh) = (gw.div_ceil(8), gh.div_ceil(8));
        let mut tokens = Vec::with_capacity(gbw * gbh * 3 * 8);
        let mut dc = Vec::with_capacity(gbw * gbh);
        let mut raw_updates = Vec::new();
        let mut nz_row = [vec![0u32; gbw], vec![0u32; gbw], vec![0u32; gbw]];
        let stride = self.xyb.w;
        let mut coeffs: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; 64 * 64]);
        let mut q: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0i32; 64 * 64]);
        let mut y_rec = vec![0f32; 64 * 64];
        let mut lf = [[0f32; 64]; 3];
        // Pass 1: transform every block and quantise Y; X and B are kept
        // unquantised until the colour correlation of their tile is known.
        let mut blks: Vec<Blk> = Vec::new();
        let (mut x_all, mut b_all, mut yr_all, mut qy_all) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut y_all = Vec::new();
        for by in 0..gbh {
            for bx in 0..gbw {
                let (abx, aby) = (bx0 + bx, by0 + by);
                let b = self.acs[aby * self.bw + abx];
                if (b.ox as usize, b.oy as usize) != (abx, aby) {
                    continue;
                }
                let acs = b.acs;
                let t = self.tables.get(acs);
                let n = t.w * t.h;
                let (nbx, nby) = acs.blocks();
                let origin = aby * 8 * stride + abx * 8;
                for c in 0..3 {
                    acs::forward_block(self.dct, acs, &self.xyb.planes[c][origin..], stride, &mut coeffs[c]);
                    acs::lf_from_llf(self.dct, &coeffs[c], acs, &mut lf[c]);
                }
                for iy in 0..nby {
                    for ix in 0..nbx {
                        let i = iy * nbx + ix;
                        dc.push((abx + ix, aby + iy, [lf[0][i], lf[1][i], lf[2][i]]));
                    }
                }
                let mut hf_mul = i32::from(self.quantizer.raw[aby * self.bw + abx]) + 1;
                let mut thres_y = [0.56, 0.62, 0.62, 0.62];
                if self.adjust && acs.small().is_none() {
                    let orig = hf_mul;
                    let mut best = 0;
                    for c in [1usize, 0, 2] {
                        let mut th = [0.58, 0.64, 0.64, 0.64];
                        let mut qn = orig;
                        self.adjust_quant(&coeffs[c][..n], t, c, acs, &mut th, &mut qn);
                        if c == 1 {
                            thres_y = th;
                        }
                        best = best.max(qn);
                    }
                    // The decoder can represent multipliers 1..=256 only.
                    hf_mul = best.clamp(1, 256);
                    raw_updates.push((aby * self.bw + abx, (hf_mul - 1) as u8));
                }
                let qac = self.quantizer.scale() * hf_mul as f32;
                let nb = nbx * nby;
                // Y first; X and B are coded relative to the reconstructed Y.
                self.quantize(&coeffs[1][..n], t, 1, nb, qac, thres_y, &mut q[1]);
                let inv_qac = 1.0 / qac;
                for y in 0..t.h {
                    for x in 0..t.w {
                        let i = y * t.w + x;
                        y_rec[i] = if x < nbx && y < nby {
                            0.0
                        } else {
                            dequant_bias(q[1][i], 1) * t.inv_weights[1][i] * inv_qac
                        };
                    }
                }
                blks.push(Blk { bx, by, acs, qac, off: x_all.len() });
                if self.recon {
                    y_all.extend_from_slice(&coeffs[1][..n]);
                }
                x_all.extend_from_slice(&coeffs[0][..n]);
                b_all.extend_from_slice(&coeffs[2][..n]);
                yr_all.extend_from_slice(&y_rec[..n]);
                qy_all.extend_from_slice(&q[1][..n]);
            }
        }
        // Pass 2: colour correlation per 64x64 tile.
        let (tw, th) = (gw.div_ceil(64), gh.div_ceil(64));
        let mut cmap = vec![[0i32; 2]; tw * th];
        {
            for (t, slot) in cmap.iter_mut().enumerate() {
                let (tx, ty) = (t % tw, t / tw);
                let list: Vec<&Blk> = blks.iter().filter(|b| b.bx / 8 == tx && b.by / 8 == ty).collect();
                for (k, c, src, base) in [(0usize, 0usize, &x_all, 0.0f32), (1, 2, &b_all, 1.0)] {
                    slot[k] = self.pick_correlation(&list, c, src, &yr_all, base);
                }
            }
        }
        // Pass 3: quantise X and B and emit tokens in block order.
        let rstride = gbw * 8;
        let mut recon: [Vec<u16>; 3] =
            if self.recon { std::array::from_fn(|_| vec![0u16; rstride * gbh * 8]) } else { Default::default() };
        let mut rc: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; 64 * 64]);
        let mut pix = vec![0f32; 64 * 64];
        for blk in &blks {
            let t = self.tables.get(blk.acs);
            let n = t.w * t.h;
            let nb = blk.acs.num_blocks();
            let [kx, kb] = cmap[(blk.by / 8) * tw + blk.bx / 8];
            let (rx, rb) = (kx as f32 / CFL_FACTOR, 1.0 + kb as f32 / CFL_FACTOR);
            let yr = &yr_all[blk.off..blk.off + n];
            for i in 0..n {
                coeffs[0][i] = x_all[blk.off + i] - rx * yr[i];
                coeffs[2][i] = b_all[blk.off + i] - rb * yr[i];
            }
            q[1][..n].copy_from_slice(&qy_all[blk.off..blk.off + n]);
            self.quantize(&coeffs[0][..n], t, 0, nb, blk.qac, [0.58, 0.62, 0.62, 0.62], &mut q[0]);
            self.quantize(&coeffs[2][..n], t, 2, nb, blk.qac, [0.58, 0.62, 0.62, 0.62], &mut q[2]);
            self.tokenize(blk.bx, blk.by, blk.acs, &q, &mut nz_row, &mut tokens);
            if self.recon {
                let (nbx, nby) = blk.acs.blocks();
                let inv_qac = 1.0 / blk.qac;
                let orig = [&x_all[blk.off..], &y_all[blk.off..], &b_all[blk.off..]];
                // Dequantise everything without branching (so that the loop
                // vectorises), then put the lowest frequencies back: they
                // come from the LF image, which is kept almost exactly.
                {
                    let (q0, q2) = (&q[0][..n], &q[2][..n]);
                    let (iw0, iw2) = (&t.inv_weights[0][..n], &t.inv_weights[2][..n]);
                    let yr = &yr[..n];
                    let [rc0, rc1, rc2] = &mut rc;
                    let (rc0, rc1, rc2) = (&mut rc0[..n], &mut rc1[..n], &mut rc2[..n]);
                    for i in 0..n {
                        let yv = yr[i];
                        rc1[i] = yv;
                        rc0[i] = rx * yv + dequant_bias(q0[i], 0) * iw0[i] * inv_qac / self.qm_mult[0];
                        rc2[i] = rb * yv + dequant_bias(q2[i], 2) * iw2[i] * inv_qac / self.qm_mult[2];
                    }
                }
                for y in 0..nby {
                    for x in 0..nbx {
                        let i = y * t.w + x;
                        for c in 0..3 {
                            rc[c][i] = orig[c][i];
                        }
                    }
                }
                let (pw, ph) = (nbx * 8, nby * 8);
                for c in 0..3 {
                    acs::inverse_block(self.dct, blk.acs, &rc[c][..n], &mut pix[..n]);
                    for y in 0..ph {
                        let dst = (blk.by * 8 + y) * rstride + blk.bx * 8;
                        for (o, &v) in recon[c][dst..dst + pw].iter_mut().zip(&pix[y * pw..(y + 1) * pw]) {
                            *o = pack_f16(v);
                        }
                    }
                }
            }
        }
        GroupOut { tokens, dc, cmap, raw_updates, recon }
    }

    /// libjxl's AdjustQuantBlockAC: nudges the block's quantiser (and the
    /// dead-zone thresholds) from the quantisation error of its HF part.
    fn adjust_quant(
        &self,
        block: &[f32],
        t: &acs::TransformTables,
        c: usize,
        acs: Acs,
        thres: &mut [f32; 4],
        quant: &mut i32,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.dct.has_avx2() {
            // SAFETY: AVX2 support was detected at runtime.
            return unsafe { self.adjust_quant_avx2(block, t, c, acs, thres, quant) };
        }
        self.adjust_quant_impl(block, t, c, acs, thres, quant)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn adjust_quant_avx2(
        &self,
        block: &[f32],
        t: &acs::TransformTables,
        c: usize,
        acs: Acs,
        thres: &mut [f32; 4],
        quant: &mut i32,
    ) {
        self.adjust_quant_impl(block, t, c, acs, thres, quant)
    }

    #[inline(always)]
    fn adjust_quant_impl(
        &self,
        block: &[f32],
        t: &acs::TransformTables,
        c: usize,
        acs: Acs,
        thres: &mut [f32; 4],
        quant: &mut i32,
    ) {
        const QUANT_MAX: i32 = 256;
        let (xs, ys) = (t.w / 8, t.h / 8);
        let area = (xs * ys) as f32;
        if xs > 1 || ys > 1 {
            for th in thres.iter_mut() {
                *th -= (0.003 * area).clamp(0.0, 0.08);
                if *th < 0.54 {
                    *th = 0.54;
                }
            }
        }
        let qm_mult = self.qm_mult[c];
        let qac = self.quantizer.scale() * *quant as f32;
        let w = &t.weights[c];
        let (mut sum_hf_corner, mut sum_err, mut sum_vals) = (0f32, 0f32, 0f32);
        let mut hf_nz = [0f32; 4];
        let mut hf_max_err = [0f32; 4];
        let m = qac * qm_mult;
        for y in 0..t.h {
            let row = y * t.w..(y + 1) * t.w;
            let (br, wr) = (&block[row.clone()], &w[row]);
            let hy = usize::from(y >= t.h / 2) * 2;
            let x0 = if y < ys { xs } else { 0 };
            for x in x0..t.w {
                let hfix = hy + usize::from(x >= t.w / 2);
                let val = br[x] * (wr[x] * m);
                let v = if val.abs() < thres[hfix] { 0.0 } else { val.round_ties_even() };
                let err = (val - v).abs();
                sum_err += err;
                sum_vals += v.abs();
                if v == 0.0 {
                    if c == 1 && hf_max_err[hfix] < err {
                        hf_max_err[hfix] = err;
                    }
                } else {
                    hf_nz[hfix] += v.abs();
                    let in_corner = y >= 7 * ys && x >= 7 * xs;
                    let on_border = y == ys * 8 - 1 || x == xs * 8 - 1;
                    let in_larger_corner = x >= 4 * xs && y >= 4 * ys;
                    if in_corner || (on_border && in_larger_corner) {
                        sum_hf_corner += val.abs();
                    }
                }
            }
        }
        if c == 1 && sum_vals * 8.0 < area {
            const LIMIT: f32 = 0.46;
            const MUL: f32 = 0.9999;
            let orig = *quant;
            let mut new_q = *quant;
            for i in 1..4 {
                if hf_nz[i] == 0.0 && hf_max_err[i] > LIMIT {
                    new_q = orig + 1;
                    break;
                }
            }
            *quant = new_q;
            let r = new_q as f32 / orig as f32;
            if hf_nz[3] == 0.0 && hf_max_err[3] > LIMIT {
                thres[3] = MUL * hf_max_err[3] * r;
            } else if (hf_nz[1] == 0.0 && hf_max_err[1] > LIMIT) || (hf_nz[2] == 0.0 && hf_max_err[2] > LIMIT) {
                thres[1] = MUL * hf_max_err[1].max(hf_max_err[2]) * r;
                thres[2] = thres[1];
            } else if hf_nz[0] == 0.0 && hf_max_err[0] > LIMIT {
                thres[0] = MUL * hf_max_err[0] * r;
            }
        }
        {
            let all = hf_nz.iter().sum::<f32>() + 1.0;
            let mul = [70.0f32, 30.0, 60.0][c];
            if mul * sum_hf_corner >= all {
                *quant = (*quant as f32 + mul * sum_hf_corner / all) as i32;
                if *quant >= QUANT_MAX {
                    *quant = QUANT_MAX - 1;
                }
            }
        }
        if acs == Acs::Dct8 && hf_nz.iter().sum::<f32>() < 11.0 {
            *quant += 1;
            if *quant >= QUANT_MAX {
                *quant = QUANT_MAX - 1;
            }
        }
        if t.w * t.h > 64 {
            const KMUL1: [[f64; 3]; 4] = [
                [0.22080615753848404, 0.45797479824262011, 0.29859235095977965],
                [0.70109486510286834, 0.16185281305512639, 0.14387691730035473],
                [0.114985964456218638, 0.44656840441027695, 0.10587658215149048],
                [0.46849665264409396, 0.41239077937781954, 0.088667407767185444],
            ];
            const KMUL2: [[f64; 3]; 4] = [
                [0.27450281941822197, 1.1255766549984996, 0.98950459134128388],
                [0.4652168675598285, 0.40945807983455818, 0.36581899811751367],
                [0.28034972424715715, 0.9182653201929738, 1.5581531543057416],
                [0.26873118114033728, 0.68863712390392484, 1.2082185408666786],
            ];
            const NORM: f64 = 2.294_270_834_328_472_1;
            let se = f64::from(sum_err) * NORM;
            let sv = f64::from(sum_vals) * NORM;
            let ix = match acs {
                Acs::Dct32x16 | Acs::Dct16x32 => 1,
                Acs::Dct16 => 0,
                Acs::Dct32 => 2,
                _ => 3,
            };
            let lim = KMUL1[ix][c] * (xs * ys * 64) as f64 + KMUL2[ix][c] * sv;
            let step = ((se / lim) as i32).clamp(0, 2);
            if se > lim {
                *quant += step;
                if *quant >= QUANT_MAX {
                    *quant = QUANT_MAX - 1;
                }
            }
        }
        {
            let div = (xs * ys) as i32;
            let min_nz = hf_nz.iter().copied().fold(f32::INFINITY, f32::min);
            let mut activity = 15;
            if min_nz < 15.0 * div as f32 {
                activity = (min_nz as i32 + div / 2) / div;
            }
            let mut qp = *quant - activity;
            if c == 1 {
                for th in thres.iter_mut().skip(1) {
                    *th += 0.01 * activity as f32;
                }
            }
            let limit = 4.max(*quant / 2);
            if qp < limit {
                qp = limit;
            }
            *quant = qp;
        }
    }

    /// Pick the correlation factor of channel `c` (X or B) for one tile by
    /// minimising a log-magnitude cost proxy of the quantised residual.
    fn pick_correlation(&self, list: &[&Blk], c: usize, src: &[f32], yr: &[f32], base: f32) -> i32 {
        // Only coefficients with a non-zero reconstructed Y depend on the
        // factor; gather them (with their quantisation scale) once.
        let (mut s, mut y, mut wm) = (Vec::new(), Vec::new(), Vec::new());
        for b in list {
            let t = self.tables.get(b.acs);
            let n = t.w * t.h;
            let m = b.qac * self.qm_mult[c];
            for i in 0..n {
                let yv = yr[b.off + i];
                if yv != 0.0 {
                    s.push(src[b.off + i]);
                    y.push(yv);
                    wm.push(t.weights[c][i] * m);
                }
            }
        }
        if s.len() < CFL_MIN_COEFFS {
            return 0;
        }
        let avx2 = self.dct.has_avx2();
        let cost = |k: i32| -> f32 {
            let r = base + k as f32 / CFL_FACTOR;
            correlation_cost(avx2, &s, &y, &wm, r)
        };
        let zero = cost(0);
        let mut best = (zero, 0i32);
        for k in (-96..=96).step_by(16) {
            if k != 0 {
                let c = cost(k);
                if c < best.0 {
                    best = (c, k);
                }
            }
        }
        for step in [4, 1] {
            let centre = best.1;
            for k in [centre - step, centre + step] {
                let k = k.clamp(-127, 127);
                let c = cost(k);
                if c < best.0 {
                    best = (c, k);
                }
            }
        }
        if best.0 + CFL_MIN_GAIN < zero { best.1 } else { 0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize(
        &self,
        coeffs: &[f32],
        t: &acs::TransformTables,
        c: usize,
        nb: usize,
        qac: f32,
        mut thres: [f32; 4],
        out: &mut [i32],
    ) {
        if c != 1 && nb >= 4 {
            for v in &mut thres {
                *v = (*v - 0.00744 * nb as f32).max(0.5);
            }
        }
        let m = qac * self.qm_mult[c];
        #[cfg(target_arch = "x86_64")]
        if self.dct.has_avx2() {
            // SAFETY: AVX2 support was detected at runtime.
            return unsafe { quantize_avx2(coeffs, &t.weights[c], t.w, t.h, m, thres, out) };
        }
        quantize_impl(coeffs, &t.weights[c], t.w, t.h, m, thres, out)
    }

    fn tokenize(
        &self,
        bx: usize,
        by: usize,
        acs: Acs,
        q: &[Vec<i32>; 3],
        nz_row: &mut [Vec<u32>; 3],
        out: &mut Vec<Token>,
    ) {
        let t = self.tables.get(acs);
        let nb = acs.num_blocks();
        let log = nb.trailing_zeros();
        let (nbx, _) = acs.blocks();
        let order = &t.order[nb..];
        for (ci, c) in [(0usize, 1usize), (1, 0), (2, 2)] {
            let block_ctx = u32::from(BLOCK_CTX_MAP[ci * 13 + acs.order_id()]);
            let row = &mut nz_row[c];
            let predicted = match (bx, by) {
                (0, 0) => 32,
                (_, 0) => row[bx - 1],
                (0, _) => row[0],
                _ => (row[bx] + row[bx - 1] + 1) >> 1,
            };
            let nz_ctx_idx = if predicted >= 8 { 4 + predicted / 2 } else { predicted };
            let qc = &q[c];
            let nz = order.iter().filter(|&&p| qc[p as usize] != 0).count() as u32;
            out.push(Token::new(block_ctx + nz_ctx_idx * NUM_BLOCK_CLUSTERS, nz));
            let val = (nz + nb as u32 - 1) >> log;
            row[bx..bx + nbx].fill(val);
            if nz == 0 {
                continue;
            }
            let base = 37 * NUM_BLOCK_CLUSTERS + block_ctx * 458;
            let mut prev = u32::from(nz <= nb as u32 * 4);
            let mut remaining = nz;
            for (idx, &p) in order.iter().enumerate() {
                let v = qc[p as usize];
                let ctx = base
                    + (u32::from(COEFF_NUM_NONZERO_CONTEXT[((remaining - 1) >> log) as usize])
                        + u32::from(COEFF_FREQ_CONTEXT[idx >> log]))
                        * 2
                    + prev;
                out.push(Token::new(ctx, pack_signed(v)));
                prev = u32::from(v != 0);
                if v != 0 {
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn quantize_impl(coeffs: &[f32], weights: &[f32], w: usize, h: usize, m: f32, thres: [f32; 4], out: &mut [i32]) {
    for y in 0..h {
        let ty = usize::from(y >= h / 2) * 2;
        let (lo, hi) = (thres[ty], thres[ty + 1]);
        let r = y * w..(y + 1) * w;
        let (cr, wr, or) = (&coeffs[r.clone()], &weights[r.clone()], &mut out[r]);
        for x in 0..w {
            let th = if x >= w / 2 { hi } else { lo };
            let v = cr[x] * wr[x] * m;
            or[x] = if v.abs() >= th { v.round() as i32 } else { 0 };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(coeffs: &[f32], weights: &[f32], w: usize, h: usize, m: f32, thres: [f32; 4], out: &mut [i32]) {
    quantize_impl(coeffs, weights, w, h, m, thres, out)
}

/// Conservatively recognize flat graphics in the original RGB samples.
/// Inspect at most 1024 complete 8x8 tiles, ignoring alpha. Requiring whole
/// tiles to be constant avoids mistaking smooth photographic gradients for UI.
fn flat_graphics(img: &ImageRef) -> bool {
    let (w, h) = (img.width as usize, img.height as usize);
    let (tw, th) = (w / 8, h / 8);
    let total = tw * th;
    if total < 4 {
        return false;
    }
    let count = total.min(1024);
    let stride = img.channels();
    let mut flat = 0;
    for i in 0..count {
        // Evenly distribute samples; split the product to avoid overflow on
        // 32-bit targets with large input buffers.
        let tile = i * (total / count) + i * (total % count) / count;
        let (x, y) = (tile % tw * 8, tile / tw * 8);
        let origin = (y * w + x) * stride;
        let color = &img.data[origin..origin + 3];
        let constant = (0..8).all(|dy| {
            (0..8).all(|dx| {
                let p = ((y + dy) * w + x + dx) * stride;
                &img.data[p..p + 3] == color
            })
        });
        flat += usize::from(constant);
    }
    flat * 100 >= count * 85
}

/// The alpha tree, learned either on a worker or (single-thread encodes) later
/// on the calling thread.
enum AlphaJob {
    Worker(std::thread::JoinHandle<GlobalTree>),
    Inline(Vec<(u32, Vec<Channel>)>, ModularOptions),
}

/// Median absolute 4-neighbour Laplacian of the luma, in 8-bit levels (a
/// noise estimate: sensor/video noise and GIF dithering score above ~8, clean
/// photographs and rendered images around 2). Samples at most ~250k pixels.
pub(crate) fn noise_level(img: &ImageRef) -> u32 {
    let (w, h) = (img.width as usize, img.height as usize);
    if w < 3 || h < 3 {
        return 0;
    }
    let stride = img.channels();
    let luma = |x: usize, y: usize| {
        let p = (y * w + x) * stride;
        let d = img.data;
        (77 * u32::from(d[p]) + 150 * u32::from(d[p + 1]) + 29 * u32::from(d[p + 2]) + 128) >> 8
    };
    let step = ((w * h) / 250_000).max(1).isqrt().max(1);
    let mut hist = [0u32; 1021];
    let mut n = 0u32;
    for y in (1..h - 1).step_by(step) {
        for x in (1..w - 1).step_by(step) {
            let l = 4 * luma(x, y) as i32
                - luma(x - 1, y) as i32
                - luma(x + 1, y) as i32
                - luma(x, y - 1) as i32
                - luma(x, y + 1) as i32;
            hist[l.unsigned_abs() as usize] += 1;
            n += 1;
        }
    }
    let mut acc = 0;
    for (v, c) in hist.iter().enumerate() {
        acc += c;
        if acc * 2 >= n {
            return v as u32;
        }
    }
    0
}

/// Noise estimate above which inverse Gaborish costs more than it saves.
/// Sharpening before quantization amplifies noise and dithering, which then
/// has to be coded; on video frames and GIFs that is 5-6% larger at equal
/// SSIMULACRA2, while clean photographs gain 5-12% from it.
pub(crate) const NOISY_LEVEL: u32 = 8;

pub fn encode(
    img: &ImageRef,
    opts: &EncoderOptions,
    placement: FramePlacement,
    noise: Option<u32>,
    out: &mut BitWriter,
) {
    let (w, h) = (img.width as usize, img.height as usize);
    let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
    let mut params = Params::new(opts.distance, opts.effort);
    // The alpha channel does not depend on the colour pipeline: learn its
    // tree and tokenise it on a worker while the colour channels are coded.
    let alpha_mopts = ModularOptions::for_effort(opts.effort, w * h).with_threads(opts.threads);
    let small = w <= GROUP_DIM && h <= GROUP_DIM;
    let alpha_rect = |x: usize, y: usize, rw: usize, rh: usize| {
        let mut data = Vec::with_capacity(rw * rh);
        for row in y..y + rh {
            let base = (row * w + x) * 4 + 3;
            data.extend(img.data[base..base + (rw - 1) * 4 + 1].iter().step_by(4).map(|&v| i32::from(v)));
        }
        Channel { w: rw, h: rh, data }
    };
    let alpha_job = (img.has_alpha() && !small).then(|| {
        let alpha_layout = Layout::new(w, h);
        let (ng, nlf) = (alpha_layout.num_groups(), alpha_layout.num_lf_groups());
        let streams: Vec<(u32, Vec<Channel>)> = (0..ng)
            .map(|g| {
                let (x, y, gw, gh) = alpha_layout.group_rect(g);
                ((1 + 3 * nlf + 17 + g) as u32, vec![alpha_rect(x, y, gw, gh)])
            })
            .collect();
        let mopts = alpha_mopts.clone();
        if opts.threads > 1 {
            AlphaJob::Worker(std::thread::spawn(move || GlobalTree::build_owned(streams, &mopts)))
        } else {
            // A single-thread encode stays on the calling thread.
            AlphaJob::Inline(streams, mopts)
        }
    });
    let xyb = Xyb::from_srgb8(img.data, img.channels(), w, h, bw * 8, bh * 8, opts.threads);
    if opts.effort >= 7 && std::env::var_os("MINI_NO_CHROMA_ADJUST").is_none() {
        let (x, b) = chroma_scales(&xyb, w, h);
        params.x_qm_scale = params.x_qm_scale.max(x);
        params.b_qm_scale = b;
    }
    let d = params.distance;
    let noise_lut = noise::params(&xyb, w, h, d, opts.noise);
    // Patches (effort >= 7): repeated glyph-like shapes are coded once in a
    // reference frame and removed from the image.
    let mut xyb = xyb;
    let patches = if opts.effort >= 7 && std::env::var_os("MINI_NO_PATCHES").is_none() {
        crate::patches::find(&xyb, w, h, d)
    } else {
        None
    };
    if let Some(p) = &patches {
        p.subtract(&mut xyb);
    }
    // Tuned for butteraugli (closer to libjxl there, worse on SSIMULACRA2);
    // MINI_NO_GABORISH=1 turns it off. Like libjxl, only above distance 0.5:
    // at higher fidelity the sharpening overshoots on hard edges (graphics,
    // text) and costs several times the bytes for no gain.
    let gaborish = params.adaptive_quant
        && d > 0.5
        && std::env::var_os("MINI_NO_GABORISH").is_none()
        && !flat_graphics(img)
        && noise.unwrap_or_else(|| noise_level(img)) < NOISY_LEVEL;
    let (mut field, mask1x1, q_median) = if params.adaptive_quant {
        // Without Gaborish, scale the AQ target as libjxl does. Flat
        // graphics avoid the sharpening overshoot and extra edge coefficients.
        let d_iqf = if gaborish { d } else { d * 0.62 };
        let maps = aq::initial_quant_field(d_iqf, &xyb, 0.765 / d_iqf, opts.threads);
        (maps.quant_field, maps.mask1x1, 0.39 / d)
    } else {
        let q = 0.79 / d;
        (vec![q; bw * bh], Vec::new(), q)
    };
    // libjxl (effort >= 6) picks the EPF strength of every block by decoding
    // under a few candidates; that needs the pre-Gaborish original.
    let dynamic_epf = opts.effort >= 6
        && params.epf_iters > 0
        && d >= 0.5
        && !mask1x1.is_empty()
        && std::env::var_os("MINI_NO_DYNAMIC_EPF").is_none();
    if gaborish {
        xyb::gaborish_inverse(&mut xyb, opts.threads);
    }
    let qk = if params.adaptive_quant { aq_calibration(d, opts.effort) } else { 1.0 };
    field.iter_mut().for_each(|v| *v *= qk);
    let q_median = q_median * qk;
    let dct = Dct::new();
    let tables = Tables::new();
    let mut acs_map: Vec<BlockAcs> =
        (0..bw * bh).map(|i| BlockAcs { acs: Acs::Dct8, ox: (i % bw) as u32, oy: (i / bw) as u32 }).collect();
    if params.choose_transforms {
        let input = SearchInput {
            planes: [&xyb.planes[0], &xyb.planes[1], &xyb.planes[2]],
            stride: xyb.w,
            bw,
            bh,
            quant: &field,
            mask1x1: &mask1x1,
            cmap: [0.0, 0.0, 1.0],
            distance: d,
            effort: opts.effort,
        };
        let mut search = acs::Search::new(&input, &tables, &dct);
        // The small transforms cost search time and only paid off on
        // graphics-like content, so try them only if a sample of tiles
        // shows a clear estimated gain.
        if params.choose_transforms && opts.effort >= 5 {
            let gain = search.probe_small(opts.threads);
            search.set_small(gain >= small_threshold(d));
        }
        let (tx, ty) = (bw.div_ceil(8), bh.div_ceil(8));
        let tiles: Vec<Vec<BlockAcs>> = par_map(tx * ty, opts.threads, |i| {
            let (bx, by) = ((i % tx) * 8, (i / tx) * 8);
            let mut out = vec![acs_map[0]; 64];
            search.tile(bx, by, &mut out, 8);
            out
        });
        for (i, t) in tiles.iter().enumerate() {
            let (bx, by) = ((i % tx) * 8, (i / tx) * 8);
            for y in 0..(bh - by).min(8) {
                for x in 0..(bw - bx).min(8) {
                    acs_map[(by + y) * bw + bx + x] = t[y * 8 + x];
                }
            }
        }
        acs::adjust_quant_field(&acs_map, bw, d, &mut field);
    }
    let mut quantizer = Quantizer::new(params.quant_dc, q_median, &field);
    drop(field);
    let layout = Layout::new(w, h);
    let ng = layout.num_groups();
    let nlf = layout.num_lf_groups();
    let groups: Vec<GroupOut> = {
        let ctx = Ctx {
            xyb: &xyb,
            layout,
            bw,
            quantizer: &quantizer,
            acs: &acs_map,
            tables: &tables,
            qm_mult: [1.25f32.powi(params.x_qm_scale as i32 - 2), 1.0, 1.25f32.powi(params.b_qm_scale as i32 - 2)],
            dct: &dct,
            adjust: params.adaptive_quant && std::env::var_os("MINI_NO_ADJUST_QUANT").is_none(),
            recon: dynamic_epf,
        };
        par_map(ng, opts.threads, |g| ctx.code_group(g))
    };
    let (xyb_w, xyb_h) = (xyb.w, xyb.h);
    drop(xyb);
    let mut groups = groups;
    for g in &groups {
        for &(i, r) in &g.raw_updates {
            quantizer.raw[i] = r;
        }
    }
    let epf_map: Option<Vec<u8>> = dynamic_epf.then(|| {
        // The decoded image is read straight from the group reconstructions,
        // one row at a time, and the original XYB of each strip is
        // recomputed from the input pixels, so neither is held as a
        // full-size plane.
        let recon_row = |c: usize, y: usize, dst: &mut [f32]| {
            let gr = y / GROUP_DIM;
            for gxi in 0..layout.groups_x {
                let g = gr * layout.groups_x + gxi;
                let (gx, gy, gw, _) = layout.group_rect(g);
                let rw = gw.div_ceil(8) * 8;
                let ly = y - gy;
                for (o, &v) in dst[gx..gx + rw].iter_mut().zip(&groups[g].recon[c][ly * rw..(ly + 1) * rw]) {
                    *o = unpack_f16(v);
                }
            }
        };
        let orig_rows = |by0: usize, by1: usize| -> [Vec<f32>; 3] {
            let (y0, y1) = (by0 * 8, (by1 * 8).min(h));
            let ch = img.channels();
            let mut x = Xyb::from_srgb8(&img.data[y0 * w * ch..y1 * w * ch], ch, w, y1 - y0, xyb_w, (by1 - by0) * 8, 1);
            if let Some(p) = &patches {
                p.subtract_rows(&mut x.planes, xyb_w, y0, (by1 - by0) * 8);
            }
            x.planes
        };
        let scale = quantizer.scale();
        let quant: Vec<f32> = (0..bw * bh)
            .map(|i| {
                let b = acs_map[i];
                let origin = b.oy as usize * bw + b.ox as usize;
                scale * (f32::from(quantizer.raw[origin]) + 1.0)
            })
            .collect();
        let input = epf::Input {
            orig: &orig_rows,
            mask1x1: &mask1x1,
            quant: &quant,
            w: xyb_w,
            h: xyb_h,
            distance: d,
            iters: params.epf_iters,
            gaborish,
        };
        epf::choose(&input, &recon_row, opts.threads)
    });
    for g in &mut groups {
        g.recon = Default::default();
    }
    drop(mask1x1);

    // Colour-correlation maps, one entry per 64x64 tile.
    let (ctw, cth) = (w.div_ceil(64), h.div_ceil(64));
    let (mut ytox, mut ytob) = (Channel::new(ctw, cth), Channel::new(ctw, cth));
    for (g, out) in groups.iter().enumerate() {
        let (gx, gy, gw, _) = layout.group_rect(g);
        let tw = gw.div_ceil(64);
        for (t, &[kx, kb]) in out.cmap.iter().enumerate() {
            let (x, y) = (gx / 64 + t % tw, gy / 64 + t / tw);
            ytox.set(x, y, kx);
            ytob.set(x, y, kb);
        }
    }

    // LF image, quantised; B is predicted from the reconstructed Y.
    let step = quantizer.lf_step();
    let mut lf = [Channel::new(bw, bh), Channel::new(bw, bh), Channel::new(bw, bh)]; // Y, X, B
    for g in &groups {
        for &(x, y, [dx, dy, db]) in &g.dc {
            let qy = (dy / step[1]).round() as i32;
            let y_rec = qy as f32 * step[1];
            lf[0].set(x, y, qy);
            lf[1].set(x, y, (dx / step[0]).round() as i32);
            lf[2].set(x, y, ((db - y_rec) / step[2]).round() as i32);
        }
    }

    let mopts = ModularOptions::for_effort(opts.effort.min(4), bw * bh * 3).with_threads(opts.threads.div_ceil(nlf));
    let lf_sections: Vec<BitWriter> = par_map(nlf, opts.threads, |i| {
        let (x, y, lw, lh) = layout.lf_group_rect(i);
        let (lbx, lby, lbw, lbh) = (x / 8, y / 8, lw.div_ceil(8), lh.div_ceil(8));
        let mut s = BitWriter::new();
        s.write(2, 0); // extra precision
        let crops: Vec<Channel> = lf.iter().map(|c| c.crop(lbx, lby, lbw, lbh)).collect();
        let refs: Vec<&Channel> = crops.iter().collect();
        // The LF image and the HF metadata are independent streams: code the
        // LF image on a worker and append both in order.
        std::thread::scope(|sc| {
            let lf_fn = || {
                let mut b = BitWriter::new();
                write_standalone(&mut b, &refs, (1 + i) as u32, &[], &mopts);
                b
            };
            // A single-thread encode stays on the calling thread.
            let lf_job = (opts.threads > 1).then(|| sc.spawn(lf_fn));
            let mut m = BitWriter::new();
            // HF metadata: colour correlation maps, block types and quant field, EPF sharpness.
            let mut blocks = Vec::new();
            for by in lby..lby + lbh {
                for bx in lbx..lbx + lbw {
                    let b = acs_map[by * bw + bx];
                    if (b.ox as usize, b.oy as usize) == (bx, by) {
                        blocks.push((b.acs as i32, i32::from(quantizer.raw[by * bw + bx])));
                    }
                }
            }
            let nb = blocks.len();
            m.write(ceil_log2((lbw * lbh) as u64), (nb - 1) as u64);
            let cw = lw.div_ceil(64);
            let chh = lh.div_ceil(64);
            let mut info = Channel::new(nb, 2);
            for (k, &(t, m)) in blocks.iter().enumerate() {
                info.set(k, 0, t);
                info.set(k, 1, m);
            }
            let mut sharp = Channel::new(lbw, lbh);
            match &epf_map {
                Some(map) => {
                    for by in 0..lbh {
                        for bx in 0..lbw {
                            sharp.set(bx, by, i32::from(map[(lby + by) * bw + lbx + bx]));
                        }
                    }
                }
                None => sharp.data.fill(epf_sharpness(d, gaborish)),
            }
            let meta = [ytox.crop(x / 64, y / 64, cw, chh), ytob.crop(x / 64, y / 64, cw, chh), info, sharp];
            let refs: Vec<&Channel> = meta.iter().collect();
            write_standalone(&mut m, &refs, (1 + 2 * nlf + i) as u32, &[], &mopts);
            let lf_bits = match lf_job {
                Some(job) => job.join().expect("LF worker panicked"),
                None => lf_fn(),
            };
            s.append(&lf_bits);
            s.append(&m);
        });
        s
    });

    drop(lf);
    drop(acs_map);

    // HF entropy code over all groups.
    let streams: Vec<&[Token]> = groups.iter().map(|g| g.tokens.as_slice()).collect();
    let code_opts = CodeOptions {
        max_clusters: if opts.effort >= 5 { 64 } else { 16 },
        tune_configs: opts.effort >= 3,
        threads: opts.threads,
    };
    let hf_code = EntropyCode::build(&streams, NUM_HF_CTX, code_opts);

    // Alpha as a modular extra channel (its tree was learned concurrently).
    let alpha = (img.has_alpha() && small).then(|| alpha_rect(0, 0, w, h));
    let alpha_tree = alpha_job.map(|job| match job {
        AlphaJob::Worker(h) => h.join().expect("alpha worker panicked"),
        AlphaJob::Inline(streams, mopts) => GlobalTree::build_owned(streams, &mopts),
    });

    let frame = FrameInfo {
        encoding: Encoding::VarDct { x_qm_scale: params.x_qm_scale, b_qm_scale: params.b_qm_scale },
        num_extra: u32::from(img.has_alpha()),
        xyb: true,
        gaborish,
        epf_iters: params.epf_iters,
        skip_lf_smoothing: false,
        patches: patches.is_some(),
        noise: noise_lut,
        reference: None,
        placement,
    };
    if let Some(p) = &patches {
        p.write_reference_frame(frame.num_extra, opts.effort, opts.threads, out);
    }
    write_frame_header(out, &frame, (w as u32, h as u32));

    let mut lf_global = BitWriter::new();
    if let Some(p) = &patches {
        p.write_dictionary(frame.num_extra, &mut lf_global);
    }
    if let Some(lut) = &noise_lut {
        for &v in lut {
            lf_global.write(10, u64::from(v));
        }
    }
    lf_global.write_bool(true); // default LF dequantisation
    lf_global.write_u32(quantizer.global_scale, [Bits(11, 1), Bits(11, 2049), Bits(12, 4097), Bits(16, 8193)]);
    lf_global.write_u32(quantizer.quant_lf, [Val(16), Bits(5, 1), Bits(8, 1), Bits(16, 1)]);
    lf_global.write_bool(true); // default HF block context
    lf_global.write_bool(true); // default LF channel correlation
    match (&alpha, &alpha_tree) {
        (None, None) => lf_global.write_bool(false), // no global tree, no channels
        (Some(a), None) => {
            lf_global.write_bool(false);
            write_standalone(&mut lf_global, &[a], 0, &[], &alpha_mopts);
        }
        (_, Some(t)) => {
            lf_global.write_bool(true);
            t.write(&mut lf_global);
            write_header(&mut lf_global, true, &[]);
            t.write_empty_stream(&mut lf_global);
        }
    }

    let mut hf_global = BitWriter::new();
    hf_global.write_bool(true); // default dequantisation matrices
    hf_global.write(ceil_log2(ng as u64), 0); // one HF preset
    hf_global.write_u32(0, [Val(0x5f), Val(0x13), Val(0), Bits(13, 0)]); // default coefficient orders
    hf_code.write_header(&mut hf_global);

    let pass_groups: Vec<BitWriter> = par_map(ng, opts.threads, |g| {
        let mut s = BitWriter::new();
        hf_code.write_tokens(&groups[g].tokens, &mut s);
        if let Some(t) = &alpha_tree {
            write_header(&mut s, true, &[]);
            t.write_stream(g, &mut s);
        }
        s
    });

    drop(groups);
    drop(alpha_tree);

    if ng == 1 {
        let mut all = lf_global;
        for s in lf_sections.iter().chain([&hf_global]).chain(pass_groups.iter()) {
            all.append(s);
        }
        write_sections(out, vec![all]);
    } else {
        let mut sections = Vec::with_capacity(2 + nlf + ng);
        sections.push(lf_global);
        sections.extend(lf_sections);
        sections.push(hf_global);
        sections.extend(pass_groups);
        write_sections(out, sections);
    }
}
