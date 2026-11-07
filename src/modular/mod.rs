//! Modular image coding: channels, transforms and modular streams.

pub mod predict;
pub mod tree;

use crate::bit_writer::BitWriter;
use crate::bit_writer::U32Dist::{Bits, Val};
use crate::entropy::{CodeOptions, EntropyCode, Token};
use crate::parallel::par_tasks;
use predict::Predictor;
use tree::{LearnOptions, Tree};

/// One channel of integer samples.
#[derive(Clone, Debug, Default)]
pub struct Channel {
    pub w: usize,
    pub h: usize,
    pub data: Vec<i32>,
}

impl Channel {
    pub fn new(w: usize, h: usize) -> Self {
        Channel { w, h, data: vec![0; w * h] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> i32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: i32) {
        self.data[y * self.w + x] = v;
    }

    /// Copy of the rectangle (x, y, w, h).
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Channel {
        let mut out = Channel::new(w, h);
        for row in 0..h {
            let src = (y + row) * self.w + x;
            out.data[row * w..(row + 1) * w].copy_from_slice(&self.data[src..src + w]);
        }
        out
    }
}

/// Modular transforms this encoder produces.
#[derive(Clone, Copy, Debug)]
pub enum Transform {
    /// Reversible colour transform on channels begin_c..begin_c+3.
    Rct { begin_c: u32, permutation: u32, kind: u32 },
    /// Palette of `nb_colors` colours (no delta entries) for `num_c` channels
    /// starting at `begin_c`; the channels become one index channel and a
    /// meta channel holding the palette is put in front.
    Palette { begin_c: u32, num_c: u32, nb_colors: u32 },
}

impl Transform {
    fn write(&self, w: &mut BitWriter) {
        match *self {
            Transform::Rct { begin_c, permutation, kind } => {
                w.write(2, 0);
                w.write_u32(begin_c, [Bits(3, 0), Bits(6, 8), Bits(10, 72), Bits(13, 1096)]);
                w.write_u32(permutation * 7 + kind, [Val(6), Bits(2, 0), Bits(4, 2), Bits(6, 10)]);
            }
            Transform::Palette { begin_c, num_c, nb_colors } => {
                w.write(2, 1);
                w.write_u32(begin_c, [Bits(3, 0), Bits(6, 8), Bits(10, 72), Bits(13, 1096)]);
                w.write_u32(num_c, [Val(1), Val(3), Val(4), Bits(13, 1)]);
                w.write_u32(nb_colors, [Bits(8, 0), Bits(10, 256), Bits(12, 1280), Bits(16, 5376)]);
                w.write_u32(0, [Val(0), Bits(8, 1), Bits(10, 257), Bits(16, 1281)]); // nb_deltas
                w.write(4, 0); // d_pred (unused without deltas)
            }
        }
    }
}

/// Forward RCT on three channels (inverse of the decoder's transform).
pub fn forward_rct(ch: &mut [Channel], permutation: u32, kind: u32) {
    let n = ch[0].data.len();
    // The decoder permutes after the inverse transform; undo that first.
    let order: [usize; 3] = match permutation {
        0 => [0, 1, 2],
        1 => [1, 2, 0],
        2 => [2, 0, 1],
        3 => [0, 2, 1],
        4 => [1, 0, 2],
        5 => [2, 1, 0],
        _ => unreachable!(),
    };
    // Each output depends only on the three inputs at the same pixel.
    // Read them before overwriting any channel, without copying the planes.
    for i in 0..n {
        let d = ch[order[0]].data[i];
        let e = ch[order[1]].data[i];
        let f = ch[order[2]].data[i];
        let (a, b, c);
        if kind == 6 {
            b = d.wrapping_sub(f);
            let tmp = f.wrapping_add(b >> 1);
            c = e.wrapping_sub(tmp);
            a = tmp.wrapping_add(c >> 1);
        } else {
            a = d;
            c = if kind & 1 != 0 { f.wrapping_sub(a) } else { f };
            b = match kind >> 1 {
                1 => e.wrapping_sub(a),
                2 => e.wrapping_sub(a.wrapping_add(f) >> 1),
                _ => e,
            };
        }
        ch[0].data[i] = a;
        ch[1].data[i] = b;
        ch[2].data[i] = c;
    }
}

/// Write a modular sub-bitstream header.
pub fn write_header(w: &mut BitWriter, use_global_tree: bool, transforms: &[Transform]) {
    w.write_bool(use_global_tree);
    w.write_bool(true); // default weighted-predictor parameters
    w.write_u32(transforms.len() as u32, [Val(0), Val(1), Bits(4, 2), Bits(8, 18)]);
    for t in transforms {
        t.write(w);
    }
}

/// Effort-dependent choices for modular coding.
#[derive(Clone, Debug)]
pub struct ModularOptions {
    pub learn: Option<LearnOptions>,
    pub fixed_predictor: Predictor,
    pub code: CodeOptions,
}

impl ModularOptions {
    pub fn for_effort(effort: u32, total_px: usize) -> Self {
        let learn = (effort >= 2).then(|| {
            // libjxl's order of usefulness for non-squeezed images.
            const ORDER: [u8; 15] = [0, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8];
            let n = match effort {
                0..=2 => 3,
                3 => 4,
                4 => 5,
                5..=7 => 7,
                8 => 10,
                _ => 15,
            };
            let mut props: Vec<u8> = ORDER[..n].to_vec();
            if effort >= 8 {
                props.push(1);
            }
            // Gradient residual of the previous channels.
            props.extend([19, 23]);
            let predictors = match effort {
                0..=2 => vec![Predictor::Gradient],
                3..=8 => vec![Predictor::Weighted, Predictor::Gradient],
                _ => vec![Predictor::Weighted, Predictor::Gradient, Predictor::AvgAll],
            };
            LearnOptions {
                predictors,
                props,
                max_samples: match effort {
                    0..=3 => 1 << 16,
                    4..=6 => 1 << 17,
                    7 => 1 << 18,
                    _ => 1 << 20,
                },
                split_cost: 150.0,
                max_leaves: (total_px / 512).clamp(16, 4096),
                threads: 1,
            }
        });
        ModularOptions {
            learn,
            fixed_predictor: Predictor::Gradient,
            code: CodeOptions {
                max_clusters: if effort >= 5 { 128 } else { 32 },
                tune_configs: effort >= 3,
                threads: 1,
            },
        }
    }

    /// Settings for palette index channels, where neighbouring indices are
    /// categories rather than intensities: constant and neighbour predictors,
    /// and splits on the neighbouring indices.
    pub fn for_palette(mut self, effort: u32) -> Self {
        if let Some(l) = &mut self.learn {
            l.predictors = vec![Predictor::Zero, Predictor::West, Predictor::North];
            if effort >= 8 {
                l.predictors.push(Predictor::Gradient);
            }
            // W, N, then the gradient-like differences and the channel.
            l.props = vec![7, 6, 10, 11, 12, 0];
            if effort >= 5 {
                l.props.extend([13, 14]);
            }
            if effort >= 7 {
                l.props.extend([2, 3]);
            }
            // Index residuals are cheap to split on: a smaller threshold than
            // for intensities pays off (measured on GIF animations).
            l.split_cost = 80.0;
        }
        self
    }

    /// Do not split on properties of more than one previous channel.
    ///
    /// jxl-oxide fails to decode the (XYB, mostly empty) patch reference
    /// frame when its tree uses these; libjxl decodes it either way.
    pub fn nearest_channel_only(mut self) -> Self {
        if let Some(learn) = &mut self.learn {
            learn.props.retain(|&p| p < 20);
        }
        self
    }

    /// Use up to `threads` worker threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        let threads = threads.max(1);
        self.code.threads = threads;
        if let Some(l) = &mut self.learn {
            l.threads = threads;
        }
        self
    }

    fn tree(&self, streams: &[(u32, Vec<&Channel>)]) -> Tree {
        match &self.learn {
            Some(opts) => tree::learn(streams, opts),
            None => Tree::single(self.fixed_predictor),
        }
    }
}

/// Write a self-contained modular stream (header, local tree, code and
/// data) for channels that are all decoded in one go.
pub fn write_standalone(
    w: &mut BitWriter,
    channels: &[&Channel],
    stream: u32,
    transforms: &[Transform],
    opts: &ModularOptions,
) {
    write_header(w, false, transforms);
    let streams = [(stream, channels.to_vec())];
    let tree = opts.tree(&streams);
    tree.write(w);
    let tokens = tree.tokenize_par(channels, stream, opts.code.threads);
    let code = EntropyCode::build(&[&tokens], tree.num_leaves(), opts.code);
    code.write_header(w);
    code.write_tokens(&tokens, w);
}

/// A tree and entropy code shared by several streams (the "global tree").
pub struct GlobalTree {
    tree: Tree,
    code: EntropyCode,
    tokens: Vec<Vec<Token>>,
}

impl GlobalTree {
    /// Learn a tree and code for all given streams and tokenise them. The
    /// channels of each stream are freed as soon as that stream is tokenised.
    pub fn build_owned(streams: Vec<(u32, Vec<Channel>)>, opts: &ModularOptions) -> Self {
        let tree = {
            let views: Vec<(u32, Vec<&Channel>)> = streams.iter().map(|(id, c)| (*id, c.iter().collect())).collect();
            opts.tree(&views)
        };
        let mut tokens: Vec<Vec<Token>> = (0..streams.len()).map(|_| Vec::new()).collect();
        let tasks = streams.into_iter().zip(tokens.iter_mut()).collect();
        par_tasks(tasks, opts.code.threads, |((id, channels), out)| {
            // One token per sample. Exact reservation avoids the spare capacity
            // from geometric growth while finished streams remain in memory.
            out.reserve_exact(channels.iter().map(|c| c.data.len()).sum());
            let refs: Vec<&Channel> = channels.iter().collect();
            tree.tokenize(&refs, id, out);
        });
        Self::finish(tree, tokens, opts)
    }

    fn finish(tree: Tree, tokens: Vec<Vec<Token>>, opts: &ModularOptions) -> Self {
        let refs: Vec<&[Token]> = tokens.iter().map(Vec::as_slice).collect();
        let code = EntropyCode::build(&refs, tree.num_leaves(), opts.code);
        GlobalTree { tree, code, tokens }
    }

    /// The tree and code description (follows the `has_tree` flag).
    pub fn write(&self, w: &mut BitWriter) {
        self.tree.write(w);
        self.code.write_header(w);
    }

    /// An ANS stream without symbols.
    pub fn write_empty_stream(&self, w: &mut BitWriter) {
        self.code.write_tokens(&[], w);
    }

    /// Data of the i-th stream given to `build`.
    pub fn write_stream(&self, i: usize, w: &mut BitWriter) {
        self.code.write_tokens(&self.tokens[i], w);
    }
}
