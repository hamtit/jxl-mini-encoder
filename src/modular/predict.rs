//! Modular predictors, the self-correcting (weighted) predictor and the MA
//! tree properties, with the edge rules of the specification.

/// Number of properties that do not depend on previous channels.
pub const NUM_BASE_PROPS: usize = 16;
/// Previous channels whose values are offered as properties.
pub const MAX_PREV_CHANNELS: usize = 2;
pub const NUM_PROPS: usize = NUM_BASE_PROPS + 4 * MAX_PREV_CHANNELS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(dead_code)] // the full set of the format's predictors
pub enum Predictor {
    Zero = 0,
    West = 1,
    North = 2,
    AvgWN = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    NorthEast = 7,
    NorthWest = 8,
    WestWest = 9,
    AvgWNW = 10,
    AvgNNW = 11,
    AvgNNE = 12,
    AvgAll = 13,
}

/// Neighbourhood of the current sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neighbors {
    pub n: i32,
    pub w: i32,
    pub nw: i32,
    pub ne: i32,
    pub nn: i32,
    pub ww: i32,
    pub nee: i32,
}

impl Neighbors {
    /// `cur` is the current row (valid up to x), `prev`/`prev2` the rows above.
    #[inline(always)]
    pub fn at(x: usize, cur: &[i32], prev: Option<&[i32]>, prev2: Option<&[i32]>, width: usize) -> Self {
        let (n, w, nw) = match (x, prev) {
            (0, None) => (0, 0, 0),
            (_, None) => (cur[x - 1], cur[x - 1], cur[x - 1]),
            (0, Some(p)) => (p[0], p[0], p[0]),
            (_, Some(p)) => (p[x], cur[x - 1], p[x - 1]),
        };
        let ne = match prev {
            Some(p) if x + 1 < width => p[x + 1],
            _ => n,
        };
        let nee = match prev {
            Some(p) if x + 2 < width => p[x + 2],
            _ => ne,
        };
        let nn = prev2.map_or(n, |p| p[x]);
        let ww = if x >= 2 { cur[x - 2] } else { w };
        Neighbors { n, w, nw, ne, nn, ww, nee }
    }
}

#[inline(always)]
pub fn grad_clamped(n: i32, w: i32, nw: i32) -> i32 {
    let (n, w, nw) = (i64::from(n), i64::from(w), i64::from(nw));
    (n + w - nw).clamp(w.min(n), w.max(n)) as i32
}

impl Predictor {
    /// Prediction for every predictor except `Weighted` (which needs `wp`).
    #[inline(always)]
    pub fn predict(self, nb: &Neighbors, wp: i64) -> i32 {
        let Neighbors { n, w, nw, ne, nn, ww, nee } = *nb;
        let avg = |a: i32, b: i32| ((i64::from(a) + i64::from(b)) / 2) as i32;
        match self {
            Predictor::Zero => 0,
            Predictor::West => w,
            Predictor::North => n,
            Predictor::AvgWN => avg(w, n),
            Predictor::Select => {
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            }
            Predictor::Gradient => grad_clamped(n, w, nw),
            Predictor::Weighted => ((wp + 3) >> 3) as i32,
            Predictor::NorthEast => ne,
            Predictor::NorthWest => nw,
            Predictor::WestWest => ww,
            Predictor::AvgWNW => avg(w, nw),
            Predictor::AvgNNW => avg(n, nw),
            Predictor::AvgNNE => avg(n, ne),
            Predictor::AvgAll => {
                let v = 6 * i64::from(n) - 2 * i64::from(nn)
                    + 7 * i64::from(w)
                    + i64::from(ww)
                    + i64::from(nee)
                    + 3 * i64::from(ne)
                    + 8;
                (v / 16) as i32
            }
        }
    }
}

const fn div_lookup() -> [u32; 65] {
    let mut out = [0u32; 65];
    let mut i = 1;
    while i <= 64 {
        out[i] = ((1 << 24) / i) as u32;
        i += 1;
    }
    out
}
const DIV_LOOKUP: [u32; 65] = div_lookup();

/// Parameters of the self-correcting predictor (the WPHeader of the Modular header). Fields are the
/// bit-stream values: `p1` and `p2` in 0..32 (5 bits), `p3` in 0..32 each, `weights` in 0..16 (4 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WpParams {
    pub p1: i64,
    pub p2: i64,
    pub p3: [i64; 5],
    pub weights: [u32; 4],
}

/// The parameters every decoder assumes when the header says "default".
pub static WP_DEFAULT: WpParams = WpParams { p1: 16, p2: 10, p3: [7, 7, 7, 0, 0], weights: [13, 12, 12, 12] };

/// Parameters for lossless photographic coding, found by coordinate descent on the total size of
/// lossless Kodak images (held-out images improve by about 0.5% in total).
pub static WP_LOSSLESS: WpParams = WpParams { p1: 0, p2: 6, p3: [2, 7, 3, 5, 1], weights: [9, 12, 12, 12] };

#[derive(Clone, Copy, Debug)]
pub struct WpPrediction {
    pub prediction: i64,
    pub max_error: i32,
    subpred: [i64; 4],
}

/// The self-correcting predictor with default parameters.
pub struct Weighted {
    params: &'static WpParams,
    width: usize,
    x: usize,
    y: usize,
    true_err_row: Vec<i32>,
    subpred_err_row: Vec<[u32; 4]>,
    true_err_w: i32,
    true_err_nw: i32,
    true_err_n: i32,
    true_err_ne: i32,
    sub_nw_ww: [u32; 4],
    sub_n_w: [u32; 4],
    sub_ne: [u32; 4],
}

impl Weighted {
    pub fn new(width: usize, params: &'static WpParams) -> Self {
        Weighted {
            params,
            width,
            x: 0,
            y: 0,
            true_err_row: vec![0; width],
            subpred_err_row: vec![[0; 4]; width],
            true_err_w: 0,
            true_err_nw: 0,
            true_err_n: 0,
            true_err_ne: 0,
            sub_nw_ww: [0; 4],
            sub_n_w: [0; 4],
            sub_ne: [0; 4],
        }
    }

    #[inline(always)]
    pub fn predict(&self, nb: &Neighbors) -> WpPrediction {
        let (te_w, te_nw, te_n, te_ne) = (
            i64::from(self.true_err_w),
            i64::from(self.true_err_nw),
            i64::from(self.true_err_n),
            i64::from(self.true_err_ne),
        );
        let WpParams { p1, p2, p3, weights } = *self.params;
        let n3 = i64::from(nb.n) << 3;
        let nw3 = i64::from(nb.nw) << 3;
        let ne3 = i64::from(nb.ne) << 3;
        let w3 = i64::from(nb.w) << 3;
        let nn3 = i64::from(nb.nn) << 3;
        let subpred = [
            w3 + ne3 - n3,
            n3 - (((te_w + te_n + te_ne) * p1) >> 5),
            w3 - (((te_w + te_n + te_nw) * p2) >> 5),
            n3 - ((te_nw * p3[0] + te_n * p3[1] + te_ne * p3[2] + (nn3 - n3) * p3[3] + (nw3 - w3) * p3[4]) >> 5),
        ];
        let mut weight = [0u32; 4];
        for i in 0..4 {
            let err_sum = self.sub_nw_ww[i].wrapping_add(self.sub_n_w[i]).wrapping_add(self.sub_ne[i]);
            let shift = ((u64::from(err_sum) + 1) >> 5).checked_ilog2().unwrap_or(0);
            weight[i] = 4 + ((weights[i] * DIV_LOOKUP[(err_sum >> shift) as usize + 1]) >> shift);
        }
        let sum: u32 = weight.iter().sum();
        let log_weight = (u64::from(sum) >> 4).ilog2();
        for w in &mut weight {
            *w >>= log_weight;
        }
        let sum: u32 = weight.iter().sum();
        let mut s = (i64::from(sum) >> 1) - 1;
        for i in 0..4 {
            s += subpred[i] * i64::from(weight[i]);
        }
        let mut prediction = (s * i64::from(DIV_LOOKUP[sum as usize])) >> 24;
        if (te_n ^ te_w) | (te_n ^ te_nw) <= 0 {
            let lo = n3.min(w3).min(ne3);
            let hi = n3.max(w3).max(ne3);
            prediction = prediction.clamp(lo, hi);
        }
        let mut max_error = te_w;
        for e in [te_n, te_nw, te_ne] {
            if e.abs() > max_error.abs() {
                max_error = e;
            }
        }
        WpPrediction { prediction, max_error: max_error as i32, subpred }
    }

    #[inline(always)]
    pub fn record(&mut self, pred: &WpPrediction, sample: i32) {
        let s8 = i64::from(sample) << 3;
        let true_err = (pred.prediction - s8) as i32;
        let mut sub = [0u32; 4];
        for i in 0..4 {
            sub[i] = ((pred.subpred[i].abs_diff(s8) + 3) >> 3) as u32;
        }
        self.true_err_row[self.x] = true_err;
        self.subpred_err_row[self.x] = sub;
        self.x += 1;
        if self.x >= self.width {
            self.y += 1;
            self.x = 0;
            self.true_err_w = 0;
            self.true_err_n = self.true_err_row[0];
            self.true_err_nw = self.true_err_n;
            self.sub_n_w = self.subpred_err_row[0];
            self.sub_nw_ww = self.sub_n_w;
            if self.width <= 1 {
                self.true_err_ne = self.true_err_n;
                self.sub_ne = self.sub_n_w;
            } else {
                self.true_err_ne = self.true_err_row[1];
                self.sub_ne = self.subpred_err_row[1];
            }
        } else {
            self.true_err_w = true_err;
            self.true_err_nw = self.true_err_n;
            self.true_err_n = self.true_err_ne;
            self.sub_nw_ww = self.sub_n_w;
            self.sub_n_w = self.sub_ne;
            for i in 0..4 {
                self.sub_n_w[i] = self.sub_n_w[i].wrapping_add(sub[i]);
            }
            if self.x + 1 >= self.width {
                self.true_err_ne = self.true_err_n;
                self.sub_ne = self.sub_n_w;
            } else if self.y != 0 {
                self.true_err_ne = self.true_err_row[self.x + 1];
                self.sub_ne = self.subpred_err_row[self.x + 1];
            }
        }
    }
}

/// Fill the 16 base properties. `prev_grad` is property 9 of the previous
/// sample in the row (0 at the row start); returns this sample's property 9.
#[inline(always)]
pub fn base_props(
    props: &mut [i32; NUM_PROPS],
    channel: i32,
    stream: i32,
    x: usize,
    y: usize,
    nb: &Neighbors,
    prev_grad: i32,
    wp_max_error: i32,
) -> i32 {
    let w_nw = nb.w.wrapping_sub(nb.nw);
    let p9 = w_nw.wrapping_add(nb.n);
    props[0] = channel;
    props[1] = stream;
    props[2] = y as i32;
    props[3] = x as i32;
    props[4] = nb.n.unsigned_abs() as i32;
    props[5] = nb.w.unsigned_abs() as i32;
    props[6] = nb.n;
    props[7] = nb.w;
    props[8] = nb.w.wrapping_sub(prev_grad);
    props[9] = p9;
    props[10] = w_nw;
    props[11] = nb.nw.wrapping_sub(nb.n);
    props[12] = nb.n.wrapping_sub(nb.ne);
    props[13] = nb.n.wrapping_sub(nb.nn);
    props[14] = nb.w.wrapping_sub(nb.ww);
    props[15] = wp_max_error;
    p9
}

/// Properties 16.. from the previous channel (k = 0 is the most recent one
/// with the same size).
#[inline(always)]
pub fn prev_channel_props(props: &mut [i32; NUM_PROPS], k: usize, ch: &[i32], width: usize, x: usize, y: usize) {
    let c = ch[y * width + x];
    let g = match (x, y) {
        (0, 0) => 0,
        (0, _) => ch[(y - 1) * width],
        (_, 0) => ch[x - 1],
        _ => grad_clamped(ch[(y - 1) * width + x], ch[y * width + x - 1], ch[(y - 1) * width + x - 1]),
    };
    let base = NUM_BASE_PROPS + 4 * k;
    props[base] = c.wrapping_abs();
    props[base + 1] = c;
    props[base + 2] = c.abs_diff(g) as i32;
    props[base + 3] = c.wrapping_sub(g);
}
