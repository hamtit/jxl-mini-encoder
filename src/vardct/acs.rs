//! Transform (AC strategy) types, their quantisation tables and coefficient
//! orders, and libjxl's block-merging search that picks them.

use super::dct::Dct;
use super::quant::dct_quant_weights;
use super::small::{self, Small};

/// The DCT transforms this encoder uses (numbered as in the bitstream).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Acs {
    Dct8 = 0,
    Identity = 1,
    Dct2x2 = 2,
    Dct4x4 = 3,
    Dct16 = 4,
    Dct32 = 5,
    /// 16 tall, 8 wide.
    Dct16x8 = 6,
    /// 8 tall, 16 wide.
    Dct8x16 = 7,
    Dct4x8 = 12,
    Dct8x4 = 13,
    Dct32x16 = 10,
    Dct16x32 = 11,
    Dct64 = 18,
    Dct64x32 = 19,
    Dct32x64 = 20,
}

pub const ALL: [Acs; 15] = [
    Acs::Dct8,
    Acs::Identity,
    Acs::Dct2x2,
    Acs::Dct4x4,
    Acs::Dct4x8,
    Acs::Dct8x4,
    Acs::Dct16,
    Acs::Dct32,
    Acs::Dct16x8,
    Acs::Dct8x16,
    Acs::Dct32x16,
    Acs::Dct16x32,
    Acs::Dct64,
    Acs::Dct64x32,
    Acs::Dct32x64,
];

impl Acs {
    /// Covered blocks (width, height).
    pub fn blocks(self) -> (usize, usize) {
        match self {
            Acs::Dct8 | Acs::Identity | Acs::Dct2x2 | Acs::Dct4x4 | Acs::Dct4x8 | Acs::Dct8x4 => {
                (1, 1)
            }
            Acs::Dct16 => (2, 2),
            Acs::Dct32 => (4, 4),
            Acs::Dct16x8 => (1, 2),
            Acs::Dct8x16 => (2, 1),
            Acs::Dct32x16 => (2, 4),
            Acs::Dct16x32 => (4, 2),
            Acs::Dct64 => (8, 8),
            Acs::Dct64x32 => (4, 8),
            Acs::Dct32x64 => (8, 4),
        }
    }

    pub fn num_blocks(self) -> usize {
        let (w, h) = self.blocks();
        w * h
    }

    pub fn order_id(self) -> usize {
        match self {
            Acs::Dct8 => 0,
            Acs::Identity | Acs::Dct2x2 | Acs::Dct4x4 | Acs::Dct4x8 | Acs::Dct8x4 => 1,
            Acs::Dct16 => 2,
            Acs::Dct32 => 3,
            Acs::Dct16x8 | Acs::Dct8x16 => 4,
            Acs::Dct32x16 | Acs::Dct16x32 => 6,
            Acs::Dct64 => 7,
            Acs::Dct64x32 | Acs::Dct32x64 => 8,
        }
    }

    /// Whether canonical (wide) coefficient coordinates are transposed in the
    /// coefficient grid.
    pub fn transposed(self) -> bool {
        let (w, h) = self.blocks();
        h >= w
    }

    /// The 8x8 transforms that are not the plain DCT.
    pub fn small(self) -> Option<Small> {
        match self {
            Acs::Identity => Some(Small::Identity),
            Acs::Dct2x2 => Some(Small::Dct2x2),
            Acs::Dct4x4 => Some(Small::Dct4x4),
            Acs::Dct4x8 => Some(Small::Dct4x8),
            Acs::Dct8x4 => Some(Small::Dct8x4),
            _ => None,
        }
    }

    fn index(self) -> usize {
        ALL.iter().position(|&a| a == self).unwrap()
    }

    fn params(self) -> [Vec<f32>; 3] {
        const SEQ_A: [f32; 7] = [
            -1.025,
            -0.78,
            -0.65012,
            -0.19041574,
            -0.20819396,
            -0.421064,
            -0.32733846,
        ];
        const SEQ_B: [f32; 7] = [
            -0.30419582,
            -0.36330363,
            -0.3566038,
            -0.34430745,
            -0.33699593,
            -0.30180866,
            -0.27321684,
        ];
        const SEQ_C: [f32; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
        let common = |a: f32, b: f32, c: f32| {
            [(a, SEQ_A), (b, SEQ_B), (c, SEQ_C)].map(|(first, seq)| {
                let mut v = vec![first];
                v.extend_from_slice(&seq);
                v
            })
        };
        match self {
            Acs::Dct8 => [
                vec![3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
                vec![560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
                vec![512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
            ],
            Acs::Dct16 => [
                vec![
                    8996.873,
                    -1.3000778,
                    -0.4942453,
                    -0.43909377,
                    -0.6350102,
                    -0.9017726,
                    -1.6162099,
                ],
                vec![
                    3191.4836,
                    -0.67424583,
                    -0.80745816,
                    -0.4492584,
                    -0.3586544,
                    -0.3132239,
                    -0.37615025,
                ],
                vec![
                    1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914,
                ],
            ],
            Acs::Dct32 => [
                vec![
                    15718.408,
                    -1.025,
                    -0.98,
                    -0.9012,
                    -0.4,
                    -0.48819396,
                    -0.421064,
                    -0.27,
                ],
                vec![
                    7305.7637,
                    -0.8041958,
                    -0.76330364,
                    -0.5566038,
                    -0.49785304,
                    -0.43699592,
                    -0.40180868,
                    -0.27321684,
                ],
                vec![
                    3803.5317,
                    -3.0607336,
                    -2.041327,
                    -2.023565,
                    -0.54953897,
                    -0.4,
                    -0.4,
                    -0.3,
                ],
            ],
            Acs::Dct16x8 | Acs::Dct8x16 => [
                vec![7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
                vec![1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
                vec![506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
            ],
            Acs::Dct32x16 | Acs::Dct16x32 => [
                vec![
                    13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6,
                ],
                vec![
                    4798.964,
                    -0.6112531,
                    -0.8377079,
                    -0.7901486,
                    -0.26927274,
                    -0.38272768,
                    -0.22924222,
                    -0.20719099,
                ],
                vec![1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5],
            ],
            Acs::Dct64 => common(23966.166, 8380.191, 4493.024),
            Acs::Dct64x32 | Acs::Dct32x64 => common(15358.898, 5597.3604, 2919.9617),
            _ => unreachable!("small transforms have their own weights"),
        }
    }
}

/// Default quantisation weights of the small 8x8 transforms, in libjxl's
/// 8x8 coefficient layout (index `y * 8 + x`).
fn small_weights(acs: Acs) -> [Vec<f32>; 3] {
    const ID: [[f32; 3]; 3] = [
        [280.0, 3160.0, 3160.0],
        [60.0, 864.0, 864.0],
        [18.0, 200.0, 200.0],
    ];
    const DCT2: [[f32; 6]; 3] = [
        [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
        [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
        [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
    ];
    const DCT4: [[f32; 4]; 3] = [
        [2200.0, 0.0, 0.0, 0.0],
        [392.0, 0.0, 0.0, 0.0],
        [112.0, -0.25, -0.25, -0.5],
    ];
    const DCT4X8: [[f32; 4]; 3] = [
        [2198.0506, -0.962_696_2, -0.761_942_5, -0.655_114_07],
        [764.365_5, -0.926_302, -0.967_523, -0.278_452_9],
        [527.107_55, -1.459_438_6, -1.450_082_1, -1.584_372_3],
    ];
    std::array::from_fn(|c| {
        let mut w = vec![0f32; 64];
        match acs {
            Acs::Identity => {
                w.fill(ID[c][0]);
                w[1] = ID[c][1];
                w[8] = ID[c][1];
                w[9] = ID[c][2];
            }
            Acs::Dct2x2 => {
                let p = &DCT2[c];
                w[0] = 2989.0; // LF slot; only used by the search's cost estimate
                w[1] = p[0];
                w[8] = p[0];
                w[9] = p[1];
                for y in 0..2 {
                    for x in 0..2 {
                        w[y * 8 + x + 2] = p[2];
                        w[(y + 2) * 8 + x] = p[2];
                        w[(y + 2) * 8 + x + 2] = p[3];
                    }
                }
                for y in 0..4 {
                    for x in 0..4 {
                        w[y * 8 + x + 4] = p[4];
                        w[(y + 4) * 8 + x] = p[4];
                        w[(y + 4) * 8 + x + 4] = p[5];
                    }
                }
            }
            Acs::Dct4x4 => {
                let w4 = dct_quant_weights(&DCT4[c], 4, 4);
                for y in 0..8 {
                    for x in 0..8 {
                        w[y * 8 + x] = w4[(y / 2) * 4 + x / 2];
                    }
                }
            }
            Acs::Dct4x8 | Acs::Dct8x4 => {
                let w4 = dct_quant_weights(&DCT4X8[c], 8, 4);
                for y in 0..8 {
                    for x in 0..8 {
                        w[y * 8 + x] = w4[(y / 2) * 8 + x];
                    }
                }
            }
            _ => unreachable!(),
        }
        w
    })
}

/// Forward transform of the varblock at `src` into `out` (the layout the
/// quantiser and tokeniser expect for `acs`).
pub fn forward_block(dct: &Dct, acs: Acs, src: &[f32], stride: usize, out: &mut [f32]) {
    match acs.small() {
        Some(k) => small::forward(k, src, stride, out),
        None => {
            let t = acs.blocks();
            dct.forward(src, stride, t.0 * 8, t.1 * 8, out)
        }
    }
}

/// Inverse of `forward_block` (`w` is the varblock width in pixels).
pub fn inverse_block(dct: &Dct, acs: Acs, coeffs: &[f32], out: &mut [f32]) {
    let (bw, bh) = acs.blocks();
    match acs.small() {
        Some(k) => small::inverse(k, coeffs, out, 8),
        None => dct.inverse(coeffs, bw * 8, bh * 8, out),
    }
}

/// Natural coefficient order for a canonical (wide) block of `bw` x `bh`
/// pixels, as canonical (x, y) positions; the LLF coefficients come first.
fn natural_order(bw: usize, bh: usize) -> Vec<(u8, u8)> {
    let y_scale = bw / bh;
    let (lbw, lbh) = (bw / 8, bh / 8);
    let mut out = Vec::with_capacity(bw * bh);
    for idx in 0..lbw * lbh {
        out.push(((idx % lbw) as u8, (idx / lbw) as u8));
    }
    for dist in 1..2 * bw {
        let margin = dist.saturating_sub(bw);
        for order in margin..dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            if x < lbw && y < lbw {
                continue;
            }
            if y % y_scale != 0 {
                continue;
            }
            out.push((x as u8, (y / y_scale) as u8));
        }
    }
    out
}

/// Per-transform tables in grid layout (`w` px wide, x horizontal).
pub struct TransformTables {
    pub w: usize,
    pub h: usize,
    /// Quantisation weights (reciprocal of the dequantisation factors).
    pub weights: [Vec<f32>; 3],
    /// Reciprocals of `weights`, for the block-size search.
    pub inv_weights: [Vec<f32>; 3],
    /// Coefficient order as grid indices `y * w + x`.
    pub order: Vec<u16>,
}

pub struct Tables {
    t: Vec<TransformTables>,
}

impl Tables {
    pub fn new() -> Self {
        let t = ALL
            .iter()
            .map(|&acs| {
                let (bw, bh) = acs.blocks();
                let (w, h) = (bw * 8, bh * 8);
                if acs.small().is_some() {
                    let weights = small_weights(acs);
                    // libjxl indexes the order as `y * 8 + x`.
                    let order = natural_order(8, 8)
                        .into_iter()
                        .map(|(ox, oy)| (oy as usize * 8 + ox as usize) as u16)
                        .collect();
                    let inv_weights = weights
                        .clone()
                        .map(|g| g.iter().map(|&w| 1.0 / w).collect());
                    return TransformTables {
                        w,
                        h,
                        weights,
                        inv_weights,
                        order,
                    };
                }
                let (cw, ch) = (w.max(h), w.min(h));
                let tr = acs.transposed();
                let params = acs.params();
                let weights = params.map(|p| {
                    let canon = dct_quant_weights(&p, cw, ch);
                    let mut g = vec![0f32; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            let (cx, cy) = if tr { (y, x) } else { (x, y) };
                            g[y * w + x] = canon[cy * cw + cx];
                        }
                    }
                    g
                });
                let order = natural_order(cw, ch)
                    .into_iter()
                    .map(|(ox, oy)| {
                        let (x, y) = if tr {
                            (oy as usize, ox as usize)
                        } else {
                            (ox as usize, oy as usize)
                        };
                        (y * w + x) as u16
                    })
                    .collect();
                let inv_weights = weights
                    .clone()
                    .map(|g| g.iter().map(|&w| 1.0 / w).collect());
                TransformTables {
                    w,
                    h,
                    weights,
                    inv_weights,
                    order,
                }
            })
            .collect();
        Tables { t }
    }

    pub fn get(&self, acs: Acs) -> &TransformTables {
        &self.t[acs.index()]
    }
}

/// libjxl's scaling between a varblock's lowest frequencies and the DCT of
/// its 8x8 block means.
const SCALE_F: [f32; 32] = [
    1.0000000000000000,
    0.9996047255830407,
    0.9984194528776054,
    0.9964458326264695,
    0.9936866130906366,
    0.9901456355893141,
    0.9858278282666936,
    0.9807391980963174,
    0.9748868211368796,
    0.9682788310563117,
    0.9609244059440204,
    0.9528337534340876,
    0.9440180941651672,
    0.9344896436056892,
    0.9242615922757944,
    0.9133480844001980,
    0.9017641950288744,
    0.8895259056651056,
    0.8766500784429904,
    0.8631544288990163,
    0.8490574973847023,
    0.8343786191696513,
    0.8191378932865928,
    0.8033561501721485,
    0.7870549181591013,
    0.7702563888779096,
    0.7529833816270532,
    0.7352593067735488,
    0.7171081282466044,
    0.6985543251889097,
    0.6796228528314652,
    0.6603391026591464,
];

/// LF (block mean) values of a varblock from its coefficients: what the
/// decoder turns back into the lowest frequencies. Returns bw*bh values in
/// raster order.
pub fn lf_from_llf(dct: &Dct, coeffs: &[f32], acs: Acs, out: &mut [f32]) {
    let (bw, bh) = acs.blocks();
    let w = bw * 8;
    if bw * bh == 1 {
        out[0] = coeffs[0];
        return;
    }
    let (lw, lh) = (bw.trailing_zeros() as usize, bh.trailing_zeros() as usize);
    let mut d = [0f32; 64];
    for y in 0..bh {
        for x in 0..bw {
            d[y * bw + x] = coeffs[y * w + x] * SCALE_F[y << (5 - lh)] * SCALE_F[x << (5 - lw)];
        }
    }
    dct.inverse(&d[..bw * bh], bw, bh, &mut out[..bw * bh]);
}

// ---------------------------------------------------------------------------
// Transform selection

/// Inputs of the search, all in block or pixel units of the padded image.
pub struct SearchInput<'a> {
    pub planes: [&'a [f32]; 3],
    pub stride: usize,
    pub bw: usize,
    pub bh: usize,
    /// Initial quantisation field per block.
    pub quant: &'a [f32],
    /// Blurred per-pixel masking.
    pub mask1x1: &'a [f32],
    /// Chroma-from-luma factors for X and B (Y is 0).
    pub cmap: [f32; 3],
    pub distance: f32,
    pub effort: u32,
}

struct Config {
    info_loss_mul: f32,
    zeros_mul: f32,
    cost_delta: f32,
}

/// Chosen transform per block: the transform and the origin of its varblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockAcs {
    pub acs: Acs,
    pub ox: u32,
    pub oy: u32,
}

pub struct Search<'a> {
    inp: &'a SearchInput<'a>,
    tables: &'a Tables,
    dct: &'a Dct,
    cfg: Config,
    /// Whether the small 8x8 transforms are tried.
    small: bool,
    /// Estimates already paid for while probing small transforms. Read-only
    /// during the parallel tile search; no scratch buffers are retained.
    probe_cache: Vec<((usize, usize), EstimateCache)>,
}

type EstimateCache = [[f32; 64]; ALL.len()];

fn ceil_log2(x: usize) -> usize {
    x.next_power_of_two().trailing_zeros() as usize
}

const SCRATCH_LEN: usize = 6 * 4096;
/// Tiles sampled when deciding whether the small transforms are worth trying.
const PROBE_TILES: usize = 12;

thread_local! {
    /// Per-thread estimate scratch, reused across tiles (it is fully
    /// overwritten before being read, so it never needs clearing).
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A tile of up to 8x8 blocks being decided.
struct Tile {
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    acs: [Acs; 64],
    /// Origin of the covering varblock, tile-relative.
    origin: [(u8, u8); 64],
    entropy: [f32; 64],
    priority: [u8; 64],
    /// Memoised estimates per (transform, tile position).
    cache: EstimateCache,
    scratch: Vec<f32>,
}

impl Tile {
    fn set(&mut self, cx: usize, cy: usize, acs: Acs) {
        let (w, h) = acs.blocks();
        for y in cy..cy + h {
            for x in cx..cx + w {
                self.acs[y * 8 + x] = acs;
                self.origin[y * 8 + x] = (cx as u8, cy as u8);
            }
        }
    }

    fn is_first(&self, x: usize, y: usize) -> bool {
        self.origin[y * 8 + x] == (x as u8, y as u8)
    }

    /// Does a varblock cross the horizontal line at row `y` between x0..x1?
    fn crosses_h(&self, x0: usize, y: usize, x1: usize) -> bool {
        if y >= self.h || y == 0 {
            return false;
        }
        (x0..x1.min(self.w)).any(|x| (self.origin[y * 8 + x].1 as usize) < y)
    }

    /// Does a varblock cross the vertical line at column `x` between y0..y1?
    fn crosses_v(&self, x: usize, y0: usize, y1: usize) -> bool {
        if x >= self.w || x == 0 {
            return false;
        }
        (y0..y1.min(self.h)).any(|y| (self.origin[y * 8 + x].0 as usize) < x)
    }

    fn set_entropy(&mut self, cx: usize, cy: usize, acs: Acs, e: f32) {
        let (w, h) = acs.blocks();
        for y in cy..cy + h {
            for x in cx..cx + w {
                self.entropy[y * 8 + x] = 0.0;
            }
        }
        self.entropy[cy * 8 + cx] = e;
    }
}

impl<'a> Search<'a> {
    pub fn new(inp: &'a SearchInput<'a>, tables: &'a Tables, dct: &'a Dct) -> Self {
        let ratio = (inp.distance + 0.137_317_43) / (1.0 + 0.137_317_43);
        let cfg = Config {
            info_loss_mul: 1.2 * ratio.powf(0.336_778_07),
            zeros_mul: 9.308_906 * ratio.powf(0.509_909_27),
            cost_delta: 10.833_273 * ratio.powf(0.367_029_4),
        };
        Search {
            inp,
            tables,
            dct,
            cfg,
            small: false,
            probe_cache: Vec::new(),
        }
    }

    /// Turn the small 8x8 transforms on or off.
    pub fn set_small(&mut self, on: bool) {
        self.small = on;
    }

    /// Relative estimated gain of trying the small 8x8 transforms, measured
    /// on a sample of tiles: `(sum DCT8 - sum best) / sum DCT8`.
    pub fn probe_small(&mut self, threads: usize) -> f32 {
        self.probe_cache.clear();
        let inp = self.inp;
        let (tx, ty) = (inp.bw.div_ceil(8), inp.bh.div_ceil(8));
        let step = (tx * ty / PROBE_TILES).max(1);
        let sampled: Vec<usize> = (0..tx * ty).step_by(step).collect();
        let results = crate::parallel::par_map(sampled.len(), threads, |i| {
            let t = sampled[i];
            let (bx, by) = ((t % tx) * 8, (t / tx) * 8);
            let mut tile = self.new_tile(bx, by);
            let (b, e) = self.select_8x8(&mut tile, true);
            let cache = tile.cache;
            self.release_tile(tile);
            ((bx, by), cache, b, e)
        });
        // Preserve the serial reduction order, independently of thread count.
        let (mut base, mut best) = (0f64, 0f64);
        for (origin, cache, b, e) in results {
            base += f64::from(b);
            best += f64::from(e);
            self.probe_cache.push((origin, cache));
        }
        if base > 0.0 {
            ((base - best) / base) as f32
        } else {
            0.0
        }
    }

    fn new_tile(&self, bx: usize, by: usize) -> Tile {
        let inp = self.inp;
        let mut tile = Tile {
            bx,
            by,
            w: (inp.bw - bx).min(8),
            h: (inp.bh - by).min(8),
            acs: [Acs::Dct8; 64],
            origin: std::array::from_fn(|i| ((i % 8) as u8, (i / 8) as u8)),
            entropy: [0.0; 64],
            priority: [0; 64],
            cache: self
                .probe_cache
                .iter()
                .find(|(origin, _)| *origin == (bx, by))
                .map(|(_, cache)| *cache)
                .unwrap_or([[f32::NAN; 64]; ALL.len()]),
            scratch: SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut())),
        };
        if tile.scratch.len() < SCRATCH_LEN {
            tile.scratch = vec![0f32; SCRATCH_LEN];
        }
        tile
    }

    fn release_tile(&self, mut tile: Tile) {
        SCRATCH.with(|c| *c.borrow_mut() = std::mem::take(&mut tile.scratch));
    }

    /// libjxl's EstimateEntropy: coefficient cost plus masked distortion.
    fn estimate(
        &self,
        acs: Acs,
        bx: usize,
        by: usize,
        entropy_mul: f32,
        scratch: &mut [f32],
    ) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.dct.has_avx2() {
            // SAFETY: AVX2 support was detected at runtime.
            return unsafe { self.estimate_avx2(acs, bx, by, entropy_mul, scratch) };
        }
        self.estimate_impl(acs, bx, by, entropy_mul, scratch)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn estimate_avx2(
        &self,
        acs: Acs,
        bx: usize,
        by: usize,
        entropy_mul: f32,
        scratch: &mut [f32],
    ) -> f32 {
        self.estimate_impl(acs, bx, by, entropy_mul, scratch)
    }

    #[inline(always)]
    fn estimate_impl(
        &self,
        acs: Acs,
        bx: usize,
        by: usize,
        entropy_mul: f32,
        scratch: &mut [f32],
    ) -> f32 {
        let inp = self.inp;
        let t = self.tables.get(acs);
        let (nbx, nby) = acs.blocks();
        let nb = nbx * nby;
        let n = t.w * t.h;
        let quant = match nb {
            1 => inp.quant[by * inp.bw + bx],
            2 => {
                let (x2, y2) = if nby == 2 { (bx, by + 1) } else { (bx + 1, by) };
                inp.quant[by * inp.bw + bx].max(inp.quant[y2 * inp.bw + x2])
            }
            _ => {
                let mut s = 0f64;
                for iy in 0..nby {
                    for ix in 0..nbx {
                        s += f64::from(inp.quant[(by + iy) * inp.bw + bx + ix]).powi(16);
                    }
                }
                ((s / nb as f64).powf(1.0 / 16.0)) as f32
            }
        };
        let origin = by * 8 * inp.stride + bx * 8;
        let (coeffs, rest) = scratch.split_at_mut(3 * 4096);
        let (err, rest) = rest.split_at_mut(4096);
        let (pix, sq) = rest.split_at_mut(4096);
        let mut cs = coeffs.chunks_exact_mut(4096);
        let coeffs: [&mut [f32]; 3] = std::array::from_fn(|_| cs.next().unwrap());
        for c in 0..3 {
            forward_block(
                self.dct,
                acs,
                &inp.planes[c][origin..],
                inp.stride,
                coeffs[c],
            );
        }
        let mut entropy = 0f32;
        let mut loss = 0f64;
        const MASK_OFFSET: [f32; 3] = [12.0, 0.0, 4.0];
        const CHANNEL_MUL: [f64; 3] = [27_984_208.0, 1.0, 1.266_770_081_387_4]; // 8.2^8, 1, 1.03^8
        for c in 0..3 {
            let mut nz = 0u32;
            let k = inp.cmap[c];
            let (wt, iw) = (&t.weights[c][..n], &t.inv_weights[c][..n]);
            let (c0, c1) = (&coeffs[c][..n], &coeffs[1][..n]);
            // Element-wise part: vectorisable.
            for i in 0..n {
                let v = (c0[i] - k * c1[i]) * wt[i] * quant;
                let r = v.round();
                err[i] = (v - r) * iw[i];
                let q = r.abs();
                sq[i] = q.sqrt();
                nz += u32::from(q != 0.0);
            }
            // Ordered sums (kept sequential so results do not depend on the
            // vector width).
            let mut acc = [0f32; 8];
            let mut chunks = sq[..n].chunks_exact(8);
            for ch in &mut chunks {
                for j in 0..8 {
                    acc[j] += ch[j];
                }
            }
            let mut sum_sqrt =
                ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
            for &v in chunks.remainder() {
                sum_sqrt += v;
            }
            inverse_block(self.dct, acs, &err[..n], pix);
            let mut lacc = [0f64; 4];
            for y in 0..t.h {
                let mrow = &inp.mask1x1[(by * 8 + y) * inp.stride + bx * 8..][..t.w];
                let prow = &pix[y * t.w..(y + 1) * t.w];
                let off = MASK_OFFSET[c];
                for (mc, pc) in mrow.chunks(4).zip(prow.chunks(4)) {
                    for j in 0..mc.len() {
                        let e = (mc[j] + off) * pc[j];
                        let e2 = e * e;
                        let e4 = f64::from(e2 * e2);
                        lacc[j] += e4 * e4;
                    }
                }
            }
            let lossc = (lacc[0] + lacc[2]) + (lacc[1] + lacc[3]);
            loss += lossc * CHANNEL_MUL[c];
            entropy += self.cfg.cost_delta * sum_sqrt;
            let nbits = ceil_log2(nz as usize + 1) + 1;
            entropy += self.cfg.zeros_mul * (ceil_log2(nbits + 17) + nbits) as f32;
            if c == 0 && nb >= 2 {
                let w = 1.0 + (nb as f32 / 8.0).min(3.0);
                entropy *= w;
                loss *= f64::from(w);
            }
        }
        let size = (nb * 64) as f64;
        let loss_scalar = ((loss / size).powf(1.0 / 8.0) * size / f64::from(quant)) as f32;
        entropy * entropy_mul + self.cfg.info_loss_mul * loss_scalar
    }

    /// `estimate` for a tile-relative position, memoised.
    fn est(&self, tile: &mut Tile, acs: Acs, cx: usize, cy: usize, mul: f32) -> f32 {
        let slot = &mut tile.cache[acs.index()][cy * 8 + cx];
        if slot.is_nan() {
            *slot = self.estimate(acs, tile.bx + cx, tile.by + cy, mul, &mut tile.scratch);
        }
        *slot
    }

    /// libjxl's FindBestFirstLevelDivisionForSquare.
    fn divide_square(
        &self,
        tile: &mut Tile,
        blocks: usize,
        cx: usize,
        cy: usize,
        mul_split: f32,
        mul_square: f32,
    ) {
        let half = blocks / 2;
        let (vsplit, hsplit, square) = match blocks {
            2 => (Acs::Dct16x8, Acs::Dct8x16, Acs::Dct16),
            4 => (Acs::Dct32x16, Acs::Dct16x32, Acs::Dct32),
            _ => (Acs::Dct64x32, Acs::Dct32x64, Acs::Dct64),
        };
        if tile.crosses_h(cx, cy, cx + blocks)
            || tile.crosses_h(cx, cy + blocks, cx + blocks)
            || tile.crosses_v(cx, cy, cy + blocks)
            || tile.crosses_v(cx + blocks, cy, cy + blocks)
        {
            return;
        }
        let allow_v = !tile.crosses_v(cx + half, cy, cy + blocks);
        let allow_h = !tile.crosses_h(cx, cy + half, cx + blocks);
        let mut q = [[0f32; 2]; 2];
        for dy in 0..blocks {
            for dx in 0..blocks {
                q[dy / half][dx / half] += tile.entropy[(cy + dy) * 8 + cx + dx];
            }
        }
        let max = f32::MAX;
        let (mut v_left, mut v_right, mut h_top, mut h_bottom) = (max, max, max, max);
        if allow_v {
            if tile.acs[cy * 8 + cx] != vsplit {
                v_left = self.est(tile, vsplit, cx, cy, mul_split);
            }
            if tile.acs[cy * 8 + cx + half] != vsplit {
                v_right = self.est(tile, vsplit, cx + half, cy, mul_split);
            }
        }
        if allow_h {
            if tile.acs[cy * 8 + cx] != hsplit {
                h_top = self.est(tile, hsplit, cx, cy, mul_split);
            }
            if tile.acs[(cy + half) * 8 + cx] != hsplit {
                h_bottom = self.est(tile, hsplit, cx, cy + half, mul_split);
            }
        }
        let e_square = self.est(tile, square, cx, cy, mul_square);
        let cost_v = v_left.min(q[0][0] + q[1][0]) + v_right.min(q[0][1] + q[1][1]);
        let cost_h = h_top.min(q[0][0] + q[0][1]) + h_bottom.min(q[1][0] + q[1][1]);
        if e_square < cost_v && e_square < cost_h {
            tile.set(cx, cy, square);
            tile.set_entropy(cx, cy, square, e_square);
        } else if cost_v < cost_h {
            if v_left < q[0][0] + q[1][0] {
                tile.set(cx, cy, vsplit);
                tile.set_entropy(cx, cy, vsplit, v_left);
            }
            if v_right < q[0][1] + q[1][1] {
                tile.set(cx + half, cy, vsplit);
                tile.set_entropy(cx + half, cy, vsplit, v_right);
            }
        } else {
            if h_top < q[0][0] + q[0][1] {
                tile.set(cx, cy, hsplit);
                tile.set_entropy(cx, cy, hsplit, h_top);
            }
            if h_bottom < q[1][0] + q[1][1] {
                tile.set(cx, cy + half, hsplit);
                tile.set_entropy(cx, cy + half, hsplit, h_bottom);
            }
        }
    }

    /// libjxl's TryMergeAcs (for tile borders where a square does not fit).
    fn try_merge(&self, tile: &mut Tile, acs: Acs, cx: usize, cy: usize, mul: f32, priority: u8) {
        let (w, h) = acs.blocks();
        let mut current = 0f32;
        for y in cy..cy + h {
            for x in cx..cx + w {
                if tile.priority[y * 8 + x] >= priority {
                    return;
                }
                current += tile.entropy[y * 8 + x];
            }
        }
        let cand = self.est(tile, acs, cx, cy, mul);
        if cand >= current {
            return;
        }
        for y in cy..cy + h {
            for x in cx..cx + w {
                tile.entropy[y * 8 + x] = 0.0;
                tile.priority[y * 8 + x] = priority;
            }
        }
        tile.set(cx, cy, acs);
        tile.entropy[cy * 8 + cx] = cand;
    }

    /// libjxl's FindBest8x8Transform for every block of the tile. Returns the
    /// summed estimate of plain DCT8 and of the chosen transforms.
    fn select_8x8(&self, tile: &mut Tile, small: bool) -> (f32, f32) {
        let inp = self.inp;
        let d = inp.distance;
        let mul8 = 1.0 - 0.4 / (d + 1.4);
        // libjxl's FindBest8x8Transform: (transform, entropy multiplier,
        // lowest effort that tries it). Multipliers are relative to DCT8's 0.8.
        const TRY8: [(Acs, f32, u32); 6] = [
            (Acs::Dct8, 0.8, 0),
            (Acs::Dct4x4, 1.08, 5),
            (Acs::Dct2x2, 0.95, 5),
            (Acs::Dct4x8, 0.859_316_4, 6),
            (Acs::Dct8x4, 0.859_316_4, 6),
            (Acs::Identity, 1.042_754_2, 5),
        ];
        let (mut sum_base, mut sum_best) = (0f32, 0f32);
        for y in 0..tile.h {
            for x in 0..tile.w {
                let mut best = (f32::MAX, Acs::Dct8);
                for &(acs, m, min_effort) in &TRY8 {
                    if inp.effort < min_effort || (acs != Acs::Dct8 && !small) {
                        continue;
                    }
                    let mut mul = m / TRY8[0].1;
                    // libjxl also favours Identity at high quality; with
                    // SSIMULACRA2 as the yardstick that only cost size, so
                    // only DCT2x2 gets the bonus.
                    if acs == Acs::Dct2x2 && d < 5.0 {
                        mul -= 0.4 * ((5.0 - d) / 5.0).powi(2);
                    }
                    if !matches!(acs, Acs::Dct8 | Acs::Dct2x2 | Acs::Identity) && d > 4.0 {
                        let k = if d < 12.0 { 8.0 / (d - 4.0) } else { 1.0 };
                        mul += 0.5 * k;
                    }
                    let e = self.est(tile, acs, x, y, mul);
                    if e < best.0 {
                        best = (e, acs);
                    }
                }
                sum_base += self.est(tile, Acs::Dct8, x, y, 1.0);
                sum_best += best.0;
                tile.set(x, y, best.1);
                tile.entropy[y * 8 + x] = best.0 * mul8;
            }
        }
        (sum_base, sum_best)
    }

    /// Decide the transforms of the 8x8-block tile at (bx, by); `out` is
    /// tile-local with row stride `out_stride`, origins are absolute.
    pub fn tile(&self, bx: usize, by: usize, out: &mut [BlockAcs], out_stride: usize) {
        let mut tile = self.new_tile(bx, by);
        let (w, h) = (tile.w, tile.h);
        let inp = self.inp;
        self.select_8x8(&mut tile, self.small);
        const MUL_16X8: f32 = 1.21;
        const MUL_16X16: f32 = 1.34;
        const MUL_16X32: f32 = 1.49;
        const MUL_32X32: f32 = 1.48;
        const MUL_64: f32 = 2.25;
        // 16x8 / 8x16 / 16x16 on aligned 2x2 squares, rectangles at the edges.
        for cy in (0..h).step_by(2) {
            for cx in (0..w).step_by(2) {
                if cy + 1 < h && cx + 1 < w {
                    self.divide_square(&mut tile, 2, cx, cy, MUL_16X8, MUL_16X16);
                }
            }
        }
        for cy in (0..h.saturating_sub(1)).step_by(2) {
            for cx in 0..w {
                if !(cx + 1 < w) && tile.is_first(cx, cy) {
                    self.try_merge(&mut tile, Acs::Dct16x8, cx, cy, MUL_16X8, 2);
                }
            }
        }
        for cy in 0..h {
            for cx in (0..w.saturating_sub(1)).step_by(2) {
                if !(cy + 1 < h) && tile.is_first(cx, cy) {
                    self.try_merge(&mut tile, Acs::Dct8x16, cx, cy, MUL_16X8, 2);
                }
            }
        }
        // 32x16 / 16x32 / 32x32 on aligned 4x4 squares.
        for cy in (0..h).step_by(4) {
            for cx in (0..w).step_by(4) {
                if cy + 3 < h && cx + 3 < w {
                    self.divide_square(&mut tile, 4, cx, cy, MUL_16X32, MUL_32X32);
                }
            }
        }
        // 64x32 / 32x64 / 64x64 on a full tile.
        if w == 8 && h == 8 {
            self.divide_square(&mut tile, 8, 0, 0, MUL_64, MUL_64);
        }
        if inp.effort >= 8 {
            for cy in 0..h.saturating_sub(1) {
                for cx in 0..w.saturating_sub(1) {
                    if (cy | cx) % 2 != 0 {
                        self.divide_square(&mut tile, 2, cx, cy, MUL_16X8, MUL_16X16);
                    }
                }
            }
            let step = if inp.effort >= 9 { 1 } else { 2 };
            for cy in (0..h.saturating_sub(3)).step_by(step) {
                for cx in (0..w.saturating_sub(3)).step_by(step) {
                    if (cy | cx) % 4 != 0 {
                        self.divide_square(&mut tile, 4, cx, cy, MUL_16X32, MUL_32X32);
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let (ox, oy) = tile.origin[y * 8 + x];
                out[y * out_stride + x] = BlockAcs {
                    acs: tile.acs[y * 8 + x],
                    ox: (bx + ox as usize) as u32,
                    oy: (by + oy as usize) as u32,
                };
            }
        }
        SCRATCH.with(|c| *c.borrow_mut() = std::mem::take(&mut tile.scratch));
    }
}

/// libjxl's AdjustQuantField: a varblock uses (roughly) the maximum quant of
/// the blocks it covers.
pub fn adjust_quant_field(acs: &[BlockAcs], bw: usize, distance: f32, field: &mut [f32]) {
    let mut mixer = 1.0f32;
    if distance > 1.541_38 {
        mixer = (1.0 - (distance - 1.541_38) * 0.563_91).max(0.0);
    }
    for (i, b) in acs.iter().enumerate() {
        let (x, y) = (i % bw, i / bw);
        if (b.ox as usize, b.oy as usize) != (x, y) || b.acs == Acs::Dct8 {
            continue;
        }
        let (w, h) = b.acs.blocks();
        let mut max = field[i];
        let mut mean = 0f32;
        for iy in 0..h {
            for ix in 0..w {
                let v = field[(y + iy) * bw + x + ix];
                mean += v;
                max = max.max(v);
            }
        }
        mean /= (w * h) as f32;
        if w * h >= 4 {
            max = max * mixer + (1.0 - mixer) * mean;
        }
        for iy in 0..h {
            for ix in 0..w {
                field[(y + iy) * bw + x + ix] = max;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_cache_does_not_change_tile_choices() {
        let (bw, bh) = (25usize, 49usize);
        let stride = bw * 8;
        let planes: [Vec<f32>; 3] = std::array::from_fn(|c| {
            (0..stride * bh * 8)
                .map(|i| {
                    let x = i % stride;
                    let y = i / stride;
                    ((x / 13 + y / 17 + c) % 7) as f32 * 0.03
                        + ((i * 37 + c * 11) % 101) as f32 * 0.0001
                })
                .collect()
        });
        let quant = vec![1.0; bw * bh];
        let mask = vec![0.5; stride * bh * 8];
        let tables = Tables::new();
        let dct = Dct::new();
        for effort in [5, 7, 9] {
            for distance in [0.3, 1.0, 8.0] {
                let input = SearchInput {
                    planes: [&planes[0], &planes[1], &planes[2]],
                    stride,
                    bw,
                    bh,
                    quant: &quant,
                    mask1x1: &mask,
                    cmap: [0.0, 0.0, 1.0],
                    distance,
                    effort,
                };
                let mut cached = Search::new(&input, &tables, &dct);
                let mut plain = Search::new(&input, &tables, &dct);
                let mut serial = Search::new(&input, &tables, &dct);
                assert_eq!(cached.probe_small(4), serial.probe_small(1));
                assert!(cached.probe_cache.len() <= 2 * PROBE_TILES);
                for small in [false, true] {
                    cached.set_small(small);
                    plain.set_small(small);
                    for by in (0..bh).step_by(8) {
                        for bx in (0..bw).step_by(8) {
                            let init = BlockAcs {
                                acs: Acs::Dct8,
                                ox: 0,
                                oy: 0,
                            };
                            let mut expected = [init; 64];
                            let mut actual = expected;
                            plain.tile(bx, by, &mut expected, 8);
                            cached.tile(bx, by, &mut actual, 8);
                            assert_eq!(
                                actual, expected,
                                "e{effort} d{distance} small={small} ({bx},{by})"
                            );
                        }
                    }
                }
            }
        }
    }
}
