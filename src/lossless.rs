//! Lossless encoding: one modular frame in the original colour space.

use crate::bit_writer::BitWriter;
use crate::entropy::{HybridUint, pack_signed};
use crate::headers::{Encoding, FrameInfo, GROUP_DIM, Layout, write_frame_header, write_sections};
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

pub fn encode(img: &ImageRef, opts: &EncoderOptions, out: &mut BitWriter) {
    let (w, h) = (img.width as usize, img.height as usize);
    let nc = img.channels();
    let effort = opts.effort;
    let rct = choose_rct(img, effort);
    let transforms: Vec<Transform> = rct.into_iter().collect();
    let mopts = ModularOptions::for_effort(effort, w * h * nc).with_threads(opts.threads);
    let frame = FrameInfo {
        encoding: Encoding::Modular,
        num_extra: (nc - 3) as u32,
        xyb: false,
        gaborish: false,
        epf_iters: 0,
        skip_lf_smoothing: false,
        patches: false,
        reference: None,
    };
    write_frame_header(out, &frame);
    let layout = Layout::new(w, h);
    if w <= GROUP_DIM && h <= GROUP_DIM {
        let channels = load_group(img, (0, 0, w, h), rct);
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
        load_group(img, layout.group_rect(g), rct)
    });
    let streams: Vec<(u32, Vec<Channel>)> = crops
        .into_iter()
        .enumerate()
        .map(|(g, c)| ((1 + 3 * nlf + 17 + g) as u32, c))
        .collect();
    let global = GlobalTree::build_owned(streams, &mopts);
    let mut sections = Vec::with_capacity(2 + nlf + ng);
    let mut lf_global = BitWriter::new();
    lf_global.write_bool(true); // default LF dequantisation
    lf_global.write_bool(true); // global tree
    global.write(&mut lf_global);
    write_header(&mut lf_global, true, &transforms);
    // No channel is small enough to be coded here; an empty stream follows.
    global.write_empty_stream(&mut lf_global);
    sections.push(lf_global);
    sections.extend((0..nlf).map(|_| BitWriter::new()));
    sections.push(BitWriter::new()); // HF global
    sections.extend(par_map(ng, opts.threads, |g| {
        let mut s = BitWriter::new();
        write_header(&mut s, true, &[]);
        global.write_stream(g, &mut s);
        s
    }));
    write_sections(out, sections);
}
