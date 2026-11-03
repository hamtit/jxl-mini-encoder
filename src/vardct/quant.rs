//! Default quantisation weights and the quantiser parameters.

fn mult(x: f32) -> f32 {
    if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
}

fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
    if bands.len() == 1 {
        return bands[0];
    }
    let scaled = pos * (bands.len() - 1) as f32 / max;
    let i = scaled as usize;
    let frac = scaled - i as f32;
    let (a, b) = (bands[i], bands[i + 1]);
    a * (b / a).powf(frac)
}

pub fn dct_quant_weights(params: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut bands = vec![params[0]];
    for &p in &params[1..] {
        bands.push(bands.last().unwrap() * mult(p));
    }
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            let d = (dx * dx + dy * dy).sqrt();
            out.push(interpolate(d, std::f32::consts::SQRT_2 + 1e-6, &bands));
        }
    }
    out
}

/// Decoder-side reconstruction biases (default opsin inverse metadata).
pub const QUANT_BIAS: [f32; 3] = [1.0 - 0.054_650_07, 1.0 - 0.070_054_5, 1.0 - 0.049_935_1];
pub const QUANT_BIAS_NUMERATOR: f32 = 0.145;

/// Value the decoder reconstructs for a quantised coefficient (before the
/// multiplier).
#[inline]
pub fn dequant_bias(q: i32, c: usize) -> f32 {
    let qf = q as f32;
    if q.abs() <= 1 {
        qf * QUANT_BIAS[c]
    } else {
        qf - QUANT_BIAS_NUMERATOR / qf
    }
}

pub const GLOBAL_SCALE_DENOM: f32 = 65536.0;
pub const QUANT_MAX: f32 = 256.0;

/// Integer quantiser parameters and the per-block quantisation field.
pub struct Quantizer {
    pub global_scale: u32,
    pub quant_lf: u32,
    pub raw: Vec<u8>, // hf_mul - 1 per block
}

impl Quantizer {
    /// libjxl's ComputeGlobalScaleAndQuant(quant_dc, quant_median, 0)
    /// followed by SetQuantFieldRect.
    pub fn new(quant_dc: f32, quant_median: f32, field: &[f32]) -> Self {
        let scale = (GLOBAL_SCALE_DENOM * quant_median / 5.0).clamp(1.0, 32768.0);
        let mut gs = scale as i32;
        let scaled_dc = (quant_dc * 4096.0 * 1.6) as i32;
        if gs > scaled_dc {
            gs = scaled_dc.max(1);
        }
        let inv_gs = GLOBAL_SCALE_DENOM / gs as f32;
        let quant_lf = ((quant_dc * inv_gs + 0.5).min(65536.0) as u32).max(1);
        let raw = field
            .iter()
            .map(|&q| ((q * inv_gs + 0.5).clamp(1.0, QUANT_MAX) as u32 - 1) as u8)
            .collect();
        Quantizer {
            global_scale: gs as u32,
            quant_lf,
            raw,
        }
    }

    /// Scale applied (with the block's hf_mul) before rounding HF coefficients.
    pub fn scale(&self) -> f32 {
        self.global_scale as f32 / GLOBAL_SCALE_DENOM
    }

    /// LF quantisation step per channel (X, Y, B).
    pub fn lf_step(&self) -> [f32; 3] {
        let inv = GLOBAL_SCALE_DENOM / (self.global_scale as f32 * self.quant_lf as f32);
        [inv / 4096.0, inv / 512.0, inv / 256.0]
    }
}
