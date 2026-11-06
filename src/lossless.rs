//! Lossless encoding: one modular frame in the original colour space.

use crate::bit_writer::BitWriter;
use crate::entropy::{HybridUint, pack_signed};
use crate::headers::{
    Encoding, FrameInfo, FramePlacement, GROUP_DIM, Layout, write_frame_header, write_sections,
};
use crate::modular::predict::grad_clamped;
use crate::modular::{
    Channel, GlobalTree, ModularOptions, Transform, forward_rct, write_header, write_standalone,
};
use crate::parallel::par_map;
use crate::{EncoderOptions, ImageRef};

/// Running statistics for the rough cost of a channel under the gradient
/// predictor (entropy of tokens).
struct GradientCost {
    hist: [u32; 64],
    extra: u64,
    n: u32,
}

impl GradientCost {
    fn new() -> Self {
        GradientCost {
            hist: [0; 64],
            extra: 0,
            n: 0,
        }
    }

    /// Add row `y` of `ch` (rows `y - 1` and `y` must be present).
    fn add_row(&mut self, ch: &Channel, y: usize) {
        for x in 1..ch.w {
            let p = grad_clamped(ch.get(x, y - 1), ch.get(x - 1, y), ch.get(x - 1, y - 1));
            let (t, e, _) = HybridUint::DEFAULT.encode(pack_signed(ch.get(x, y) - p));
            self.hist[(t as usize).min(63)] += 1;
            self.extra += u64::from(e);
            self.n += 1;
        }
    }

    fn cost(&self) -> f64 {
        let t = f64::from(self.n.max(1));
        self.hist
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| f64::from(c) * (t / f64::from(c)).log2())
            .sum::<f64>()
            + self.extra as f64
    }
}

/// The `nc` channels of the pixel rectangle (x, y, w, h).
fn load_rect(img: &ImageRef, nc: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<Channel> {
    let stride = img.channels();
    let width = img.width as usize;
    (0..nc)
        .map(|c| {
            let mut data = Vec::with_capacity(w * h);
            for row in y..y + h {
                let base = (row * width + x) * stride + c;
                data.extend(
                    img.data[base..base + (w - 1) * stride + 1]
                        .iter()
                        .step_by(stride)
                        .map(|&v| i32::from(v)),
                );
            }
            Channel { w, h, data }
        })
        .collect()
}

/// Pick the reversible colour transform that makes the colour channels
/// cheapest to code. Only the sampled rows are looked at, so no full-size
/// copy of the image is needed.
fn choose_rct(img: &ImageRef, effort: u32) -> Option<Transform> {
    const YCOCG: (u32, u32) = (0, 6);
    if effort < 4 {
        return Some(Transform::Rct {
            begin_c: 0,
            permutation: YCOCG.0,
            kind: YCOCG.1,
        });
    }
    let candidates = &[YCOCG, (1, 3), (1, 5), (0, 0), (1, 2)];
    let (w, h) = (img.width as usize, img.height as usize);
    let step = ((h * w) / (1 << 18)).max(1);
    let mut best: Option<(f64, (u32, u32))> = None;
    for &(perm, kind) in candidates {
        let mut acc = [
            GradientCost::new(),
            GradientCost::new(),
            GradientCost::new(),
        ];
        for y in (1..h).step_by(step) {
            // Two rows are enough: the predictor looks at y - 1 and y only.
            let mut rows = load_rect(img, 3, 0, y - 1, w, 2);
            if (perm, kind) != (0, 0) {
                forward_rct(&mut rows, perm, kind);
            }
            for (a, ch) in acc.iter_mut().zip(&rows) {
                a.add_row(ch, 1);
            }
        }
        let cost: f64 = acc.iter().map(GradientCost::cost).sum();
        if best.is_none_or(|b| cost < b.0) {
            best = Some((cost, (perm, kind)));
        }
    }
    let (_, (perm, kind)) = best?;
    if (perm, kind) == (0, 0) {
        return None;
    }
    Some(Transform::Rct {
        begin_c: 0,
        permutation: perm,
        kind,
    })
}

/// Channels of a rectangle with the colour transform applied.
fn load_group(
    img: &ImageRef,
    (x, y, w, h): (usize, usize, usize, usize),
    transform: Option<Transform>,
) -> Vec<Channel> {
    let mut ch = load_rect(img, img.channels(), x, y, w, h);
    if let Some(Transform::Rct {
        permutation, kind, ..
    }) = transform
    {
        forward_rct(&mut ch[..3], permutation, kind);
    }
    ch
}

/// An exact palette of at most 256 colours (RGB or RGBA tuples).
struct Palette {
    /// Colours in index order, each `channels` bytes.
    colors: Vec<[u8; 4]>,
    /// Open-addressing table of (packed colour + 1, index); size is a power of two.
    table: Vec<(u64, u16)>,
}

const PALETTE_TABLE: usize = 1024;

#[inline]
fn pack_px(px: &[u8]) -> u32 {
    let a = if px.len() == 4 { px[3] } else { 0 };
    u32::from_le_bytes([px[0], px[1], px[2], a])
}

#[inline]
fn slot(key: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> 22) as usize & (PALETTE_TABLE - 1)
}

impl Palette {
    /// The palette of the image, if it has at most 256 distinct colours.
    fn of(img: &ImageRef) -> Option<Palette> {
        let ch = img.channels();
        let mut keys: Vec<u32> = Vec::new();
        let mut seen = vec![u64::MAX; PALETTE_TABLE];
        let mut last = None;
        for px in img.data.chunks_exact(ch) {
            let key = pack_px(px);
            if last == Some(key) {
                continue;
            }
            last = Some(key);
            let mut i = slot(key);
            loop {
                if seen[i] == u64::MAX {
                    if keys.len() == 256 {
                        return None;
                    }
                    seen[i] = u64::from(key);
                    keys.push(key);
                    break;
                }
                if seen[i] == u64::from(key) {
                    break;
                }
                i = (i + 1) & (PALETTE_TABLE - 1);
            }
        }
        // Order by luma so that smooth ramps stay smooth in index space.
        keys.sort_by_key(|&k| {
            let [r, g, b, a] = k.to_le_bytes();
            (
                u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114,
                a,
                k,
            )
        });
        let mut table = vec![(0u64, 0u16); PALETTE_TABLE];
        let mut colors = Vec::with_capacity(keys.len());
        for (idx, &k) in keys.iter().enumerate() {
            let mut i = slot(k);
            while table[i].0 != 0 {
                i = (i + 1) & (PALETTE_TABLE - 1);
            }
            table[i] = (u64::from(k) + 1, idx as u16);
            colors.push(k.to_le_bytes());
        }
        Some(Palette { colors, table })
    }

    fn index(&self, key: u32) -> i32 {
        let mut i = slot(key);
        loop {
            let (k, idx) = self.table[i];
            if k == u64::from(key) + 1 {
                return i32::from(idx);
            }
            i = (i + 1) & (PALETTE_TABLE - 1);
        }
    }

    /// The meta channel: one row per colour component.
    fn meta(&self, nc: usize) -> Channel {
        let n = self.colors.len();
        let mut m = Channel::new(n, nc);
        for (i, c) in self.colors.iter().enumerate() {
            for k in 0..nc {
                m.set(i, k, i32::from(c[k]));
            }
        }
        m
    }

    fn transform(&self, nc: usize) -> Transform {
        Transform::Palette {
            begin_c: 0,
            num_c: nc as u32,
            nb_colors: self.colors.len() as u32,
        }
    }
}

/// The index channel of the rectangle (x, y, w, h).
fn load_indices(
    img: &ImageRef,
    pal: &Palette,
    (x, y, w, h): (usize, usize, usize, usize),
) -> Channel {
    let ch = img.channels();
    let width = img.width as usize;
    let mut out = Channel::new(w, h);
    for row in 0..h {
        let base = ((y + row) * width + x) * ch;
        let src = &img.data[base..base + w * ch];
        let dst = &mut out.data[row * w..(row + 1) * w];
        let mut last: Option<(u32, i32)> = None;
        for (d, px) in dst.iter_mut().zip(src.chunks_exact(ch)) {
            let key = pack_px(px);
            *d = match last {
                Some((k, i)) if k == key => i,
                _ => {
                    let i = pal.index(key);
                    last = Some((key, i));
                    i
                }
            };
        }
    }
    out
}

pub fn encode(
    img: &ImageRef,
    opts: &EncoderOptions,
    placement: FramePlacement,
    out: &mut BitWriter,
) {
    let (w, h) = (img.width as usize, img.height as usize);
    let nc = img.channels();
    let effort = opts.effort;
    let palette = Palette::of(img);
    let rct = if palette.is_some() {
        None
    } else {
        choose_rct(img, effort)
    };
    let transforms: Vec<Transform> = match &palette {
        Some(p) => vec![p.transform(nc)],
        None => rct.into_iter().collect(),
    };
    let mopts = ModularOptions::for_effort(effort, w * h * nc).with_threads(opts.threads);
    let mopts = if palette.is_some() { mopts.for_palette(effort) } else { mopts };
    let frame = FrameInfo {
        encoding: Encoding::Modular,
        num_extra: (nc - 3) as u32,
        xyb: false,
        gaborish: false,
        epf_iters: 0,
        skip_lf_smoothing: false,
        patches: false,
        noise: None,
        reference: None,
        placement,
    };
    write_frame_header(out, &frame, (w as u32, h as u32));
    let layout = Layout::new(w, h);
    if w <= GROUP_DIM && h <= GROUP_DIM {
        let channels = match &palette {
            Some(p) => vec![p.meta(nc), load_indices(img, p, (0, 0, w, h))],
            None => load_group(img, (0, 0, w, h), rct),
        };
        let refs: Vec<&Channel> = channels.iter().collect();
        let mut s = BitWriter::new();
        s.write_bool(true); // default LF dequantisation
        s.write_bool(false); // no global tree
        write_standalone(&mut s, &refs, 0, &transforms, &mopts);
        write_sections(out, vec![s]);
        return;
    }
    let ng = layout.num_groups();
    let nlf = layout.num_lf_groups();
    let crops: Vec<Vec<Channel>> = par_map(ng, opts.threads, |g| {
        match &palette {
            Some(p) => vec![load_indices(img, p, layout.group_rect(g))],
            None => load_group(img, layout.group_rect(g), rct),
        }
    });
    // The palette is a small meta channel, coded in the global stream (id 0).
    let mut streams: Vec<(u32, Vec<Channel>)> = Vec::new();
    if let Some(p) = &palette {
        streams.push((0, vec![p.meta(nc)]));
    }
    let first_group = streams.len();
    streams.extend(
        crops
            .into_iter()
            .enumerate()
            .map(|(g, c)| ((1 + 3 * nlf + 17 + g) as u32, c)),
    );
    let global = GlobalTree::build_owned(streams, &mopts);
    let mut sections = Vec::with_capacity(2 + nlf + ng);
    let mut lf_global = BitWriter::new();
    lf_global.write_bool(true); // default LF dequantisation
    lf_global.write_bool(true); // global tree
    global.write(&mut lf_global);
    write_header(&mut lf_global, true, &transforms);
    if palette.is_some() {
        global.write_stream(0, &mut lf_global);
    } else {
        // No channel is small enough to be coded here; an empty stream follows.
        global.write_empty_stream(&mut lf_global);
    }
    sections.push(lf_global);
    sections.extend((0..nlf).map(|_| BitWriter::new()));
    sections.push(BitWriter::new()); // HF global
    sections.extend(par_map(ng, opts.threads, |g| {
        let mut s = BitWriter::new();
        write_header(&mut s, true, &[]);
        global.write_stream(first_group + g, &mut s);
        s
    }));
    write_sections(out, sections);
}
