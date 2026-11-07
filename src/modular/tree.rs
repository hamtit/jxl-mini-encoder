//! Meta-adaptive (MA) context trees: tokenisation of channels with a tree,
//! tree serialisation and a greedy entropy-driven tree learner.

use super::Channel;
use super::predict::{
    MAX_PREV_CHANNELS, NUM_PROPS, Neighbors, Predictor, Weighted, base_props, prev_channel_props,
};
use crate::bit_writer::BitWriter;
use crate::entropy::{CodeOptions, HybridUint, Token, pack_signed, write_stream};
use crate::parallel::{par_map, par_tasks};

#[derive(Clone, Copy, Debug)]
pub enum Node {
    /// `prop > val` goes to `left`, otherwise to `right`.
    Split {
        prop: u8,
        val: i32,
        left: u32,
        right: u32,
    },
    Leaf {
        pred: Predictor,
        ctx: u32,
    },
}

/// A learned tree in breadth-first serialisation order, which also numbers
/// the leaf contexts. Tokenisation-only specialisations may have a different
/// node order but retain the learned tree's context numbers.
#[derive(Clone, Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    num_leaves: u32,
    uses_wp: bool,
    max_prev: usize,
}

/// Recursive form used while building.
enum Build {
    Split {
        prop: u8,
        val: i32,
        /// Predictor of the node if it ends up as a leaf.
        pred: Predictor,
        left: Box<Build>,
        right: Box<Build>,
    },
    Leaf(Predictor),
}

impl Tree {
    pub fn single(pred: Predictor) -> Self {
        Tree::from_build(Build::Leaf(pred))
    }

    fn from_build(root: Build) -> Self {
        let mut nodes = Vec::new();
        let mut queue = std::collections::VecDeque::from([root]);
        let mut next_child = 1u32;
        let mut num_leaves = 0;
        let mut uses_wp = false;
        let mut max_prev = 0;
        while let Some(b) = queue.pop_front() {
            match b {
                Build::Split {
                    prop,
                    val,
                    left,
                    right,
                    ..
                } => {
                    nodes.push(Node::Split {
                        prop,
                        val,
                        left: next_child,
                        right: next_child + 1,
                    });
                    next_child += 2;
                    uses_wp |= prop == 15;
                    if prop as usize >= 16 {
                        max_prev = max_prev.max((prop as usize - 16) / 4 + 1);
                    }
                    queue.push_back(*left);
                    queue.push_back(*right);
                }
                Build::Leaf(pred) => {
                    uses_wp |= pred == Predictor::Weighted;
                    nodes.push(Node::Leaf {
                        pred,
                        ctx: num_leaves,
                    });
                    num_leaves += 1;
                }
            }
        }
        Tree {
            nodes,
            num_leaves,
            uses_wp,
            max_prev,
        }
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves as usize
    }

    #[inline(always)]
    fn leaf(&self, props: &[i32; NUM_PROPS]) -> (Predictor, u32) {
        let mut i = 0usize;
        loop {
            match self.nodes[i] {
                Node::Split {
                    prop,
                    val,
                    left,
                    right,
                } => {
                    i = if props[prop as usize] > val {
                        left
                    } else {
                        right
                    } as usize;
                }
                Node::Leaf { pred, ctx } => return (pred, ctx),
            }
        }
    }

    /// Resolve properties that are constant for a whole channel. In particular,
    /// a weighted predictor in another channel must not make this one maintain
    /// weighted-predictor state. Leaf context numbers remain those of the
    /// original, serialised tree.
    fn specialize(&self, channel: i32, stream: i32) -> std::borrow::Cow<'_, Self> {
        if !self
            .nodes
            .iter()
            .any(|n| matches!(n, Node::Split { prop: 0 | 1, .. }))
        {
            return std::borrow::Cow::Borrowed(self);
        }
        fn copy(tree: &Tree, i: usize, channel: i32, stream: i32, out: &mut Vec<Node>) -> u32 {
            if let Node::Split {
                prop: 0 | 1,
                val,
                left,
                right,
            } = tree.nodes[i]
            {
                let prop = match tree.nodes[i] {
                    Node::Split { prop: 0, .. } => channel,
                    _ => stream,
                };
                return copy(
                    tree,
                    if prop > val { left } else { right } as usize,
                    channel,
                    stream,
                    out,
                );
            }
            let index = out.len() as u32;
            out.push(tree.nodes[i]);
            if let Node::Split {
                prop,
                val,
                left,
                right,
            } = tree.nodes[i]
            {
                let left = copy(tree, left as usize, channel, stream, out);
                let right = copy(tree, right as usize, channel, stream, out);
                out[index as usize] = Node::Split {
                    prop,
                    val,
                    left,
                    right,
                };
            }
            index
        }
        let mut nodes = Vec::new();
        copy(self, 0, channel, stream, &mut nodes);
        let uses_wp = nodes.iter().any(|n| {
            matches!(
                n,
                Node::Split { prop: 15, .. }
                    | Node::Leaf {
                        pred: Predictor::Weighted,
                        ..
                    }
            )
        });
        let max_prev = nodes
            .iter()
            .filter_map(|n| match n {
                Node::Split { prop, .. } if *prop >= 16 => Some((*prop as usize - 16) / 4 + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        std::borrow::Cow::Owned(Self {
            nodes,
            num_leaves: self.num_leaves,
            uses_wp,
            max_prev,
        })
    }

    /// Serialise the tree with its own entropy code.
    pub fn write(&self, w: &mut BitWriter) {
        let mut tokens = Vec::with_capacity(self.nodes.len() * 2);
        for n in &self.nodes {
            match *n {
                Node::Split { prop, val, .. } => {
                    tokens.push(Token::new(1, u32::from(prop) + 1));
                    tokens.push(Token::new(0, pack_signed(val)));
                }
                Node::Leaf { pred, .. } => {
                    tokens.push(Token::new(1, 0));
                    tokens.push(Token::new(2, pred as u32));
                    tokens.push(Token::new(3, 0)); // offset
                    tokens.push(Token::new(4, 0)); // multiplier log
                    tokens.push(Token::new(5, 0)); // multiplier bits
                }
            }
        }
        write_stream(
            &tokens,
            6,
            CodeOptions {
                max_clusters: 6,
                tune_configs: true,
                threads: 1,
            },
            w,
        );
    }

    /// Tokenise a list of channels forming one modular stream.
    /// `tokenize` with the channels of one stream coded concurrently.
    pub fn tokenize_par(&self, channels: &[&Channel], stream: u32, threads: usize) -> Vec<Token> {
        let mut out = Vec::new();
        if threads <= 1 || channels.len() < 2 {
            self.tokenize(channels, stream, &mut out);
            return out;
        }
        let parts = par_map(channels.len(), threads, |i| {
            let mut t = Vec::new();
            self.tokenize_one(channels, i, stream, &mut t);
            t
        });
        for p in parts {
            out.extend(p);
        }
        out
    }

    pub fn tokenize(&self, channels: &[&Channel], stream: u32, out: &mut Vec<Token>) {
        for i in 0..channels.len() {
            self.tokenize_one(channels, i, stream, out);
        }
    }

    fn tokenize_one(&self, channels: &[&Channel], i: usize, stream: u32, out: &mut Vec<Token>) {
        let ch = channels[i];
        if ch.w == 0 || ch.h == 0 {
            return;
        }
        let tree = self.specialize(i as i32, stream as i32);
        let prev = previous_same_size(channels, i, tree.max_prev);
        visit(
            ch,
            i as i32,
            stream as i32,
            &prev,
            tree.uses_wp,
            |props, nb, wp, v| {
                let (pred, ctx) = tree.leaf(props);
                let p = pred.predict(nb, wp);
                out.push(Token::new(ctx, pack_signed(v.wrapping_sub(p))));
            },
        );
    }
}

/// Earlier channels with the same dimensions, most recent first.
fn previous_same_size<'a>(channels: &[&'a Channel], i: usize, max_prev: usize) -> Vec<&'a Channel> {
    let ch = channels[i];
    channels[..i]
        .iter()
        .rev()
        .filter(|c| c.w == ch.w && c.h == ch.h)
        .take(max_prev)
        .copied()
        .collect()
}

/// Walk a channel in raster order, computing properties and (optionally) the
/// weighted prediction for every sample.
fn visit(
    ch: &Channel,
    index: i32,
    stream: i32,
    prev: &[&Channel],
    use_wp: bool,
    mut f: impl FnMut(&[i32; NUM_PROPS], &Neighbors, i64, i32),
) {
    let w = ch.w;
    let mut wp = use_wp.then(|| Weighted::new(w));
    let mut props = [0i32; NUM_PROPS];
    for y in 0..ch.h {
        let cur = &ch.data[y * w..(y + 1) * w];
        let above = (y > 0).then(|| &ch.data[(y - 1) * w..y * w]);
        let above2 = (y > 1).then(|| &ch.data[(y - 2) * w..(y - 1) * w]);
        let mut prev_grad = 0;
        for x in 0..w {
            let nb = Neighbors::at(x, cur, above, above2, w);
            let wpp = wp.as_ref().map(|wp| wp.predict(&nb));
            let max_err = wpp.as_ref().map_or(0, |p| p.max_error);
            prev_grad = base_props(&mut props, index, stream, x, y, &nb, prev_grad, max_err);
            for (k, p) in prev.iter().enumerate() {
                prev_channel_props(&mut props, k, &p.data, w, x, y);
            }
            let v = cur[x];
            f(&props, &nb, wpp.as_ref().map_or(0, |p| p.prediction), v);
            if let (Some(wp), Some(p)) = (wp.as_mut(), wpp.as_ref()) {
                wp.record(p, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(prop: u8, val: i32, left: Build, right: Build) -> Build {
        Build::Split {
            prop,
            val,
            pred: Predictor::Gradient,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    #[test]
    fn specialized_tokens_match_full_tree() {
        let tree = Tree::from_build(split(
            0,
            1,
            split(
                1,
                2,
                split(
                    19,
                    0,
                    Build::Leaf(Predictor::Gradient),
                    Build::Leaf(Predictor::AvgAll),
                ),
                Build::Leaf(Predictor::North),
            ),
            split(
                15,
                0,
                Build::Leaf(Predictor::Weighted),
                split(
                    23,
                    0,
                    Build::Leaf(Predictor::Weighted),
                    Build::Leaf(Predictor::Gradient),
                ),
            ),
        ));
        assert!(tree.uses_wp);
        assert!(!tree.specialize(2, 3).uses_wp);
        assert_eq!(tree.specialize(2, 0).max_prev, 0);
        for (w, h) in [(1, 1), (1, 19), (19, 1), (17, 13)] {
            let channels: Vec<Channel> = (0..4)
                .map(|c| Channel {
                    w,
                    h,
                    data: (0..w * h)
                        .map(|i| ((i * 37 + c * 71) % 511) as i32 - 255)
                        .collect(),
                })
                .collect();
            let refs: Vec<_> = channels.iter().collect();
            for stream in [0, 3, 91] {
                let mut expected = Vec::new();
                for (i, ch) in channels.iter().enumerate() {
                    let prev = previous_same_size(&refs, i, tree.max_prev);
                    visit(
                        ch,
                        i as i32,
                        stream as i32,
                        &prev,
                        tree.uses_wp,
                        |props, nb, wp, v| {
                            let (pred, ctx) = tree.leaf(props);
                            expected.push(Token::new(
                                ctx,
                                pack_signed(v.wrapping_sub(pred.predict(nb, wp))),
                            ));
                        },
                    );
                }
                assert_eq!(tree.tokenize_par(&refs, stream, 1), expected);
                assert_eq!(tree.tokenize_par(&refs, stream, 3), expected);
            }
        }
    }

    #[test]
    fn constant_planes_use_gradient_tree() {
        let channels: Vec<_> = [-91, 0, 255]
            .into_iter()
            .map(|v| Channel {
                w: 17,
                h: 13,
                data: vec![v; 17 * 13],
            })
            .collect();
        let refs: Vec<_> = channels.iter().collect();
        let opts = super::super::ModularOptions::for_effort(7, 17 * 13 * 3);
        let tree = learn(&[(0, refs.clone())], opts.learn.as_ref().unwrap());
        assert_eq!(tree.num_leaves(), 1);
        assert!(!tree.uses_wp);
        let tokens = tree.tokenize_par(&refs, 0, 2);
        for (c, plane) in tokens.chunks(17 * 13).enumerate() {
            assert_eq!(plane[0].value(), pack_signed(channels[c].data[0]));
            assert!(plane[1..].iter().all(|t| t.value() == 0));
        }
    }
}

// ---------------------------------------------------------------------------
// Tree learning

/// Learning parameters.
#[derive(Clone, Debug)]
pub struct LearnOptions {
    pub predictors: Vec<Predictor>,
    /// Properties that may be split on.
    pub props: Vec<u8>,
    /// Upper bound on the number of samples used for learning.
    pub max_samples: usize,
    /// Bits a split must save to be kept.
    pub split_cost: f64,
    pub max_leaves: usize,
    /// Worker threads for sampling and split search.
    pub threads: usize,
}

const MAX_TOK: usize = 64;
const MAX_BUCKETS: usize = 32;
/// Smallest node whose split search is spread over threads.
const PAR_SPLIT_MIN: usize = 8192;
/// Smallest pair of children built concurrently.
const PAR_SUBTREE_MIN: usize = 16384;

/// One sampled pixel: its properties and, per candidate predictor, the token
/// and raw-bit count of its residual.
#[derive(Clone, Copy)]
struct Sample<const N: usize> {
    // Only the properties selected by LearnOptions, in the same order.
    props: [i32; N],
    tok: [u8; 4],
    extra: [u8; 4],
}

impl<const N: usize> Default for Sample<N> {
    fn default() -> Self {
        Self {
            props: [0; N],
            tok: [0; 4],
            extra: [0; 4],
        }
    }
}

fn residual_token(r: i32) -> (u8, u8) {
    let (t, n, _) = HybridUint::DEFAULT.encode(pack_signed(r));
    ((t as usize).min(MAX_TOK - 1) as u8, n as u8)
}

/// `c * log2(c)` for small counts.
const XLOGX_N: usize = 1 << 16;

fn xlogx_table() -> &'static [f32] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..XLOGX_N)
            .map(|c| {
                if c == 0 {
                    0.0
                } else {
                    (c as f64 * (c as f64).log2()) as f32
                }
            })
            .collect()
    })
}

/// `c * log2(c)`, tabulated for small counts.
#[inline]
fn xlogx_t(t: &[f32], c: u32) -> f64 {
    match t.get(c as usize) {
        Some(&v) => f64::from(v),
        None => f64::from(c) * f64::from(c).log2(),
    }
}

#[inline]
fn xlogx(c: u32) -> f64 {
    xlogx_t(xlogx_table(), c)
}

#[derive(Clone)]
struct Hist {
    counts: [u32; MAX_TOK],
    total: u32,
    extra: u64,
}

impl Hist {
    fn new() -> Self {
        Hist {
            counts: [0; MAX_TOK],
            total: 0,
            extra: 0,
        }
    }
    #[inline]
    fn add(&mut self, tok: u8, extra: u8) {
        self.counts[tok as usize] += 1;
        self.total += 1;
        self.extra += u64::from(extra);
    }
    fn cost(&self) -> f64 {
        self.cost_sub(None, MAX_TOK)
    }
    /// Cost of `self - other`, looking only at the first `ntok` tokens
    /// (all later counts are zero and contribute nothing).
    #[inline]
    fn cost_sub(&self, other: Option<&Hist>, ntok: usize) -> f64 {
        let (total, extra) = match other {
            Some(o) => (self.total - o.total, self.extra - o.extra),
            None => (self.total, self.extra),
        };
        if total == 0 {
            return 0.0;
        }
        let mut bits = extra as f64 + xlogx(total);
        match other {
            Some(o) => {
                for (a, b) in self.counts[..ntok].iter().zip(&o.counts[..ntok]) {
                    bits -= xlogx(a - b);
                }
            }
            None => {
                for &c in &self.counts[..ntok] {
                    bits -= xlogx(c);
                }
            }
        }
        bits
    }
}

/// Learn a tree for the given streams (each a list of channels).
pub fn learn(streams: &[(u32, Vec<&Channel>)], opts: &LearnOptions) -> Tree {
    // Constant planes (notably opaque alpha) need only their first sample and
    // zero gradient residuals thereafter. Sampling and weighted-predictor
    // training cannot improve those residuals, but still visit every pixel.
    if streams.iter().all(|(_, channels)| {
        channels.iter().all(|ch| {
            ch.data
                .first()
                .is_none_or(|first| ch.data.iter().all(|v| v == first))
        })
    }) {
        return Tree::single(Predictor::Gradient);
    }
    // Bound each sample's stride by the number of properties actually used.
    // Keep a few size classes rather than specializing for every effort.
    match opts.props.len() {
        0..=6 => learn_sized::<6>(streams, opts),
        7..=9 => learn_sized::<9>(streams, opts),
        10..=13 => learn_sized::<13>(streams, opts),
        14..=18 => learn_sized::<18>(streams, opts),
        _ => learn_sized::<NUM_PROPS>(streams, opts),
    }
}

fn learn_sized<const N: usize>(streams: &[(u32, Vec<&Channel>)], opts: &LearnOptions) -> Tree {
    debug_assert!(opts.props.len() <= N);
    let npred = opts.predictors.len().min(4);
    let total_px: usize = streams
        .iter()
        .flat_map(|(_, c)| c.iter())
        .map(|c| c.w * c.h)
        .sum();
    if total_px == 0 {
        return Tree::single(opts.predictors[0]);
    }
    let stride = total_px.div_ceil(opts.max_samples.max(1)).max(1);
    let need_wp = opts.predictors.contains(&Predictor::Weighted) || opts.props.contains(&15);
    let uses_prev = opts.props.iter().any(|&p| p >= 16);
    // One unit per (stream, channel); `offset` is the number of pixels
    // before it, so every unit samples exactly the pixels a sequential walk
    // would have.
    let mut units = Vec::new();
    let mut offset = 0usize;
    for (si, (_, chans)) in streams.iter().enumerate() {
        for (i, ch) in chans.iter().enumerate() {
            if ch.w == 0 || ch.h == 0 {
                continue;
            }
            units.push((si, i, offset));
            offset += ch.w * ch.h;
        }
    }
    // The number of samples each unit yields is known up front, so units
    // write straight into their own slice of one buffer.
    let counts: Vec<usize> = units
        .iter()
        .map(|&(si, i, start)| {
            let n = streams[si].1[i].w * streams[si].1[i].h;
            (start + n) / stride - start / stride
        })
        .collect();
    let mut samples = vec![Sample::<N>::default(); counts.iter().sum()];
    let mut rest = samples.as_mut_slice();
    let mut tasks = Vec::with_capacity(units.len());
    for (&unit, &c) in units.iter().zip(&counts) {
        let (head, tail) = rest.split_at_mut(c);
        tasks.push((unit, head));
        rest = tail;
    }
    par_tasks(tasks, opts.threads, |((si, i, start), out)| {
        let (stream, chans) = &streams[si];
        let ch = chans[i];
        let mut counter = start;
        let mut k = 0;
        let prev = previous_same_size(chans, i, if uses_prev { MAX_PREV_CHANNELS } else { 0 });
        visit(
            ch,
            i as i32,
            *stream as i32,
            &prev,
            need_wp,
            |props, nb, wp, v| {
                counter += 1;
                if !counter.is_multiple_of(stride) {
                    return;
                }
                let s = &mut out[k];
                k += 1;
                for (dst, &p) in s.props.iter_mut().zip(&opts.props) {
                    *dst = props[p as usize];
                }
                for (kk, p) in opts.predictors.iter().take(npred).enumerate() {
                    let (t, e) = residual_token(v.wrapping_sub(p.predict(nb, wp)));
                    s.tok[kk] = t;
                    s.extra[kk] = e;
                }
            },
        );
        debug_assert_eq!(k, out.len());
    });
    let scale = stride as f64;
    // Candidate thresholds per property, from quantiles.
    let thresholds: Vec<Vec<i32>> = (0..NUM_PROPS)
        .map(|p| {
            let Some(prop_index) = opts.props.iter().position(|&prop| prop as usize == p) else {
                return Vec::new();
            };
            let step = (samples.len() / 4096).max(1);
            let mut vals: Vec<i32> = samples
                .iter()
                .step_by(step)
                .map(|s| s.props[prop_index])
                .collect();
            vals.sort_unstable();
            vals.dedup();
            if vals.len() <= 1 {
                return Vec::new();
            }
            // Split values must leave something on both sides: drop the max.
            vals.pop();
            if vals.len() < MAX_BUCKETS {
                return vals;
            }
            let n = vals.len();
            let mut out: Vec<i32> = (1..MAX_BUCKETS)
                .map(|k| vals[k * n / MAX_BUCKETS])
                .collect();
            out.dedup();
            out
        })
        .collect();
    let mut idx: Vec<u32> = (0..samples.len() as u32).collect();
    let leaves = 1usize;
    let ntok = samples
        .iter()
        .flat_map(|s| s.tok[..npred].iter())
        .map(|&t| t as usize + 1)
        .max()
        .unwrap_or(1);
    // Bucket of every sample for every property, computed once: the same
    // `partition_point` the split search would otherwise redo at each node.
    // Row-major, so the counting pass touches one cache line per sample.
    let active: Vec<usize> = (0..opts.props.len())
        .filter(|&pi| !thresholds[opts.props[pi] as usize].is_empty())
        .collect();
    let bk: Vec<[u8; N]> = samples
        .iter()
        .map(|s| {
            let mut row = [0u8; N];
            for &pi in &active {
                let th = &thresholds[opts.props[pi] as usize];
                row[pi] = th.partition_point(|&t| t < s.props[pi]) as u8;
            }
            row
        })
        .collect();
    let lay = StatsLayout::new(opts, &thresholds, ntok, npred, active);
    let ctx = Learner {
        samples: &samples,
        bk: &bk,
        ntok,
        thresholds: &thresholds,
        opts,
        npred,
        scale,
        lay,
    };
    let (root, _) = ctx.build(&mut idx, None, leaves, 0, opts.threads.max(1));
    Tree::from_build(root)
}

/// Smallest node whose children's statistics are derived by subtraction.
const SUB_MIN: usize = 2048;

/// Where each property's bucket statistics live in a [`Stats`]. Section
/// `props.len()` is the whole node (one bucket).
struct StatsLayout {
    /// Properties (indices into `opts.props`) that have split candidates.
    active: Vec<usize>,
    /// Number of buckets per section (0 when the property has no thresholds).
    nb: Vec<usize>,
    cnt: Vec<usize>,
    tot: Vec<usize>,
    ext: Vec<usize>,
    cnt_len: usize,
    tot_len: usize,
    ext_len: usize,
}

impl StatsLayout {
    fn new(
        opts: &LearnOptions,
        thresholds: &[Vec<i32>],
        ntok: usize,
        np: usize,
        active: Vec<usize>,
    ) -> Self {
        let mut l = StatsLayout {
            active,
            nb: Vec::new(),
            cnt: Vec::new(),
            tot: Vec::new(),
            ext: Vec::new(),
            cnt_len: 0,
            tot_len: 0,
            ext_len: 0,
        };
        for pi in 0..=opts.props.len() {
            let nb = if pi == opts.props.len() {
                1
            } else {
                let th = &thresholds[opts.props[pi] as usize];
                if th.is_empty() { 0 } else { th.len() + 1 }
            };
            l.nb.push(nb);
            l.cnt.push(l.cnt_len);
            l.tot.push(l.tot_len);
            l.ext.push(l.ext_len);
            l.cnt_len += nb * np * ntok;
            l.tot_len += nb;
            l.ext_len += nb * np;
        }
        l
    }
}

/// Token histograms of a node, per property bucket and predictor:
/// `counts[cnt[p] + (b * np + k) * ntok + token]`, `btotal[tot[p] + b]`,
/// `bextra[ext[p] + b * np + k]`.
struct Stats {
    counts: Vec<u32>,
    btotal: Vec<u32>,
    bextra: Vec<u64>,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        for (a, b) in self.counts.iter_mut().zip(&o.counts) {
            *a += b;
        }
        for (a, b) in self.btotal.iter_mut().zip(&o.btotal) {
            *a += b;
        }
        for (a, b) in self.bextra.iter_mut().zip(&o.bextra) {
            *a += b;
        }
    }

    /// `self - o`, where `o` was counted over a subset of `self`'s samples.
    fn minus(mut self, o: &Stats) -> Stats {
        for (a, b) in self.counts.iter_mut().zip(&o.counts) {
            *a -= b;
        }
        for (a, b) in self.btotal.iter_mut().zip(&o.btotal) {
            *a -= b;
        }
        for (a, b) in self.bextra.iter_mut().zip(&o.bextra) {
            *a -= b;
        }
        self
    }
}

struct Learner<'a, const N: usize> {
    samples: &'a [Sample<N>],
    /// `bk[sample][prop_index]`: threshold bucket of each sample.
    bk: &'a [[u8; N]],
    /// Number of token values in use (counts beyond are zero).
    ntok: usize,
    thresholds: &'a [Vec<i32>],
    opts: &'a LearnOptions,
    npred: usize,
    scale: f64,
    lay: StatsLayout,
}

impl<const N: usize> Learner<'_, N> {
    fn empty_stats(&self) -> Stats {
        Stats {
            counts: vec![0; self.lay.cnt_len],
            btotal: vec![0; self.lay.tot_len],
            bextra: vec![0; self.lay.ext_len],
        }
    }

    /// Statistics of the samples in `idx`, one pass over them for all properties.
    fn count_seq(&self, idx: &[u32]) -> Stats {
        let (np, ntok) = (self.npred, self.ntok);
        let lay = &self.lay;
        let mut st = self.empty_stats();
        let whole = lay.nb.len() - 1;
        for &i in idx {
            let s = &self.samples[i as usize];
            let row = &self.bk[i as usize];
            for &p in &lay.active {
                let b = usize::from(row[p]);
                st.btotal[lay.tot[p] + b] += 1;
                let base = lay.cnt[p] + b * np * ntok;
                let eb = lay.ext[p] + b * np;
                for k in 0..np {
                    st.counts[base + k * ntok + usize::from(s.tok[k])] += 1;
                    st.bextra[eb + k] += u64::from(s.extra[k]);
                }
            }
            st.btotal[lay.tot[whole]] += 1;
            for k in 0..np {
                st.counts[lay.cnt[whole] + k * ntok + usize::from(s.tok[k])] += 1;
                st.bextra[lay.ext[whole] + k] += u64::from(s.extra[k]);
            }
        }
        st
    }

    fn count(&self, idx: &[u32], threads: usize) -> Stats {
        if threads <= 1 || idx.len() < PAR_SPLIT_MIN {
            return self.count_seq(idx);
        }
        let chunk = idx.len().div_ceil(threads);
        let parts = par_map(idx.len().div_ceil(chunk), threads, |c| {
            self.count_seq(&idx[c * chunk..((c + 1) * chunk).min(idx.len())])
        });
        let mut it = parts.into_iter();
        let mut total = it.next().unwrap();
        for p in it {
            total.add(&p);
        }
        total
    }

    /// Per-predictor token histograms of a node, straight from its samples.
    fn node_hists(&self, idx: &[u32]) -> Vec<Hist> {
        let mut hists = vec![Hist::new(); self.npred];
        for &i in idx.iter() {
            let s = &self.samples[i as usize];
            for (k, h) in hists.iter_mut().enumerate() {
                h.add(s.tok[k], s.extra[k]);
            }
        }
        hists
    }

    fn hists_of(&self, st: &Stats) -> Vec<Hist> {
        let (np, ntok) = (self.npred, self.ntok);
        let whole = self.lay.nb.len() - 1;
        (0..np)
            .map(|k| {
                let mut h = Hist::new();
                let base = self.lay.cnt[whole] + k * ntok;
                h.counts[..ntok].copy_from_slice(&st.counts[base..base + ntok]);
                h.total = st.btotal[self.lay.tot[whole]];
                h.extra = st.bextra[self.lay.ext[whole] + k];
                h
            })
            .collect()
    }

    /// Best split of property `prop_index` from the node statistics:
    /// (cost, prop, value).
    fn best_split_for(
        &self,
        st: &Stats,
        prop_index: usize,
        prop: u8,
        hists: &[Hist],
    ) -> Option<(f64, u8, i32)> {
        let (np, ntok) = (self.npred, self.ntok);
        let th = &self.thresholds[prop as usize];
        if th.is_empty() {
            return None;
        }
        let lay = &self.lay;
        let counts = &st.counts[lay.cnt[prop_index]..];
        let btotal = &st.btotal[lay.tot[prop_index]..];
        let bextra = &st.bextra[lay.ext[prop_index]..];
        let mut best: Option<(f64, u8, i32)> = None;
        // right = buckets[..=t], left = buckets[t+1..]
        let mut rcounts = vec![0u32; np * ntok];
        let mut rtotal = 0u32;
        let mut rextra = vec![0u64; np];
        let all = hists[0].total;
        let table = xlogx_table();
        for (t, &val) in th.iter().enumerate() {
            if btotal[t] == 0 {
                // Nothing moved to the right side: same split as before.
                continue;
            }
            rtotal += btotal[t];
            for k in 0..np {
                let src = &counts[(t * np + k) * ntok..(t * np + k + 1) * ntok];
                for (a, b) in rcounts[k * ntok..(k + 1) * ntok].iter_mut().zip(src) {
                    *a += b;
                }
                rextra[k] += bextra[t * np + k];
            }
            if rtotal == all {
                continue;
            }
            let mut rc = f64::INFINITY;
            let mut lc = f64::INFINITY;
            for k in 0..np {
                let r = &rcounts[k * ntok..(k + 1) * ntok];
                let mut rbits = rextra[k] as f64 + xlogx_t(table, rtotal);
                for &c in r {
                    rbits -= xlogx_t(table, c);
                }
                rc = rc.min(rbits);
                let h = &hists[k];
                let mut lbits = (h.extra - rextra[k]) as f64 + xlogx_t(table, h.total - rtotal);
                for (a, b) in h.counts[..ntok].iter().zip(r) {
                    lbits -= xlogx_t(table, a - b);
                }
                lc = lc.min(lbits);
            }
            let c = lc + rc;
            if best.is_none_or(|b| c < b.0) {
                best = Some((c, prop, val));
            }
        }
        best
    }

    /// Build the subtree for `idx`. `leaves` is the number of leaves so far
    /// (depth-first), and the number after this subtree is returned. `stats`
    /// are the node's statistics when the parent already derived them.
    fn build(
        &self,
        idx: &mut [u32],
        stats: Option<Stats>,
        leaves: usize,
        depth: usize,
        threads: usize,
    ) -> (Build, usize) {
        let leaf = idx.len() < 64 || leaves >= self.opts.max_leaves || depth > 40;
        let (st, hists) = match stats {
            Some(st) => {
                let h = self.hists_of(&st);
                (Some(st), h)
            }
            None if leaf => (None, self.node_hists(idx)),
            None => {
                let st = self.count(idx, threads);
                let h = self.hists_of(&st);
                (Some(st), h)
            }
        };
        // Best predictor for this node.
        let (best_k, node_cost) = hists
            .iter()
            .map(Hist::cost)
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let pred = self.opts.predictors[best_k];
        if leaf {
            return (Build::Leaf(pred), leaves);
        }
        let st = st.expect("statistics of a split candidate");
        // Find the best split; each side picks its own best predictor.
        let mut best: Option<(f64, u8, i32)> = None;
        for (k, &prop) in self.opts.props.iter().enumerate() {
            if let Some(c) = self.best_split_for(&st, k, prop, &hists) {
                if best.is_none_or(|b| c.0 < b.0) {
                    best = Some(c);
                }
            }
        }
        let Some((cost, prop, val)) = best else {
            return (Build::Leaf(pred), leaves);
        };
        let gain = (node_cost - cost) * self.scale;
        if gain < self.opts.split_cost {
            return (Build::Leaf(pred), leaves);
        }
        // Partition: left (prop > val) first.
        let prop_index = self.opts.props.iter().position(|&p| p == prop).unwrap();
        let mut k = 0;
        for j in 0..idx.len() {
            if self.samples[idx[j] as usize].props[prop_index] > val {
                idx.swap(j, k);
                k += 1;
            }
        }
        let leaves = leaves + 1;
        let (l, r) = idx.split_at_mut(k);
        // Count the smaller side only; the larger one is the parent minus it.
        let (lst, rst) = if l.len() + r.len() >= SUB_MIN {
            if l.len() <= r.len() {
                let ls = self.count(l, threads);
                let rs = st.minus(&ls);
                (Some(ls), Some(rs))
            } else {
                let rs = self.count(r, threads);
                let ls = st.minus(&rs);
                (Some(ls), Some(rs))
            }
        } else {
            drop(st);
            (None, None)
        };
        let (left, right, after) = if threads > 1 && l.len().min(r.len()) >= PAR_SUBTREE_MIN {
            // Build both sides at once. The right side is built as if the
            // left used no leaf budget (so it may be too large) and is then
            // trimmed to what a depth-first walk would have allowed.
            let (tl, tr) = (threads.div_ceil(2), threads / 2);
            let ((left, after_l), (spec, _)) = std::thread::scope(|sc| {
                let h = sc.spawn(|| self.build(r, rst, leaves, depth + 1, tr));
                let a = self.build(l, lst, leaves, depth + 1, tl);
                (a, h.join().expect("worker panicked"))
            });
            let mut counter = after_l;
            let right = self.trim(spec, &mut counter);
            (left, right, counter)
        } else {
            let (left, after_l) = self.build(l, lst, leaves, depth + 1, threads);
            let (right, after_r) = self.build(r, rst, after_l, depth + 1, threads);
            (left, right, after_r)
        };
        (
            Build::Split {
                prop,
                val,
                pred,
                left: Box::new(left),
                right: Box::new(right),
            },
            after,
        )
    }

    /// Re-apply the leaf limit to a subtree that was built with too much
    /// budget: nodes reached after `counter` hits the limit become leaves.
    fn trim(&self, b: Build, counter: &mut usize) -> Build {
        match b {
            Build::Leaf(p) => Build::Leaf(p),
            Build::Split {
                prop,
                val,
                pred,
                left,
                right,
            } => {
                if *counter >= self.opts.max_leaves {
                    return Build::Leaf(pred);
                }
                *counter += 1;
                let left = self.trim(*left, counter);
                let right = self.trim(*right, counter);
                Build::Split {
                    prop,
                    val,
                    pred,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
        }
    }
}
