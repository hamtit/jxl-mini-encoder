//! Noise synthesis parameters.
//!
//! Heavy video noise and GIF dithering are mostly quantised away at the
//! distances animations are coded at, which leaves the decoded frames
//! visibly smoother than the source. The decoder can add pseudo-random noise
//! (seeded per displayed frame) on top of the decoded image; the encoder only
//! has to send the strength. The synthesised noise is luma-like: it moves X
//! by nothing and Y and B together.

use super::xyb::Xyb;

/// Standard deviation of the decoder's noise in Y per unit LUT value: the
/// filtered white noise has std ~1.13 and Y gets `0.22 * (sx + sy)` of it.
const Y_STD_PER_LUT: f32 = 0.44 * 1.13;

/// Fraction of the noise variance that coding at `distance` removes. Fitted
/// on video-noise GIF frames, where the share of Laplacian energy kept was
/// 0.70 / 0.84 / 0.95 at distances 3.7 / 2.4 / 1.0.
fn lost_variance(distance: f32) -> f32 {
    let kept = 1.0 - 0.08 * distance;
    (1.0 - kept * kept).clamp(0.0, 1.0)
}

/// Noise strength of the original in Y: the lower quartile of the per-block
/// RMS of the residual to the 4-neighbour mean, so that edges and texture do
/// not count as noise. Returns 0 when there are too few flat blocks.
fn estimate_sigma(xyb: &Xyb, w: usize, h: usize) -> f32 {
    let p = &xyb.planes[1];
    let s = xyb.w;
    let (bw, bh) = ((w.saturating_sub(2)) / 8, (h.saturating_sub(2)) / 8);
    if bw == 0 || bh == 0 {
        return 0.0;
    }
    let mut rms = Vec::with_capacity(bw * bh);
    for by in 0..bh {
        for bx in 0..bw {
            let mut acc = 0.0f32;
            for y in 0..8 {
                let row = (by * 8 + y + 1) * s + bx * 8 + 1;
                for x in 0..8 {
                    let i = row + x;
                    let r = p[i] - 0.25 * (p[i - 1] + p[i + 1] + p[i - s] + p[i + s]);
                    acc += r * r;
                }
            }
            rms.push((acc / 64.0).sqrt());
        }
    }
    let k = rms.len() / 4;
    let (_, q, _) = rms.select_nth_unstable_by(k, f32::total_cmp);
    // Residual std of white noise is sqrt(1 + 1/4) times its own.
    *q / 1.118
}

/// LUT (10 bit, 8 entries) for the noise to add after coding at `distance`,
/// or `None` when the content is clean or coding keeps the noise anyway.
pub fn params(xyb: &Xyb, w: usize, h: usize, distance: f32, gain: f32) -> Option<[u16; 8]> {
    if gain.is_nan() || gain <= 0.0 {
        return None;
    }
    let sigma = estimate_sigma(xyb, w, h) * (lost_variance(distance)).sqrt() * gain;
    let lut = (sigma / Y_STD_PER_LUT * 1024.0).round();
    if lut < 8.0 {
        return None;
    }
    Some([lut.min(1023.0) as u16; 8])
}
