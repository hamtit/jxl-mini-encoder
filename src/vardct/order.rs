//! Custom coefficient orders (libjxl's ComputeCoeffOrder): per transform
//! order class and channel, the HF positions are sorted by how often they are
//! non-zero, so that the zero tail of a block (which is not coded) starts
//! earlier and the coded positions share better-predicted contexts.
//!
//! The groups are first tokenised in the natural order. The statistics are
//! then read back from those tokens and every group is re-tokenised in the
//! new order, so no coefficient plane has to be kept.

use super::acs::{ALL, Acs, BlockAcs};
use super::{coeff_ctx_base, emit_coeffs};
use crate::bit_writer::BitWriter;
use crate::bit_writer::U32Dist::{Bits, Val};
use crate::entropy::{CodeOptions, HybridUint, Token, write_stream};
use crate::headers::Layout;
use crate::parallel::par_map;

/// Number of order classes in the format (`kStrategyOrder` values).
const NUM_ORDS: usize = 13;
/// Contexts of the permutation stream.
const PERM_CTX: usize = 8;

/// Coefficients per block of an order class (0 if no transform here uses it).
fn ord_size() -> [usize; NUM_ORDS] {
    let mut s = [0; NUM_ORDS];
    for acs in ALL {
        s[acs.order_id()] = 64 * acs.num_blocks();
    }
    s
}

/// For every order class and channel (X, Y, B), the natural index of each
/// position of the order; `None` keeps the natural order.
pub struct CoeffOrders {
    perm: Vec<Option<[Vec<u16>; 3]>>,
}

impl CoeffOrders {
    pub fn natural() -> Self {
        CoeffOrders { perm: vec![None; NUM_ORDS] }
    }

    pub fn is_natural(&self) -> bool {
        self.perm.iter().all(Option::is_none)
    }

    fn used(&self) -> u32 {
        self.perm.iter().enumerate().filter(|(_, p)| p.is_some()).map(|(o, _)| 1 << o).sum()
    }

    /// Writes the used-orders field and the permutations (HfPass).
    pub fn write(&self, threads: usize, w: &mut BitWriter) {
        let used = self.used();
        w.write_u32(used, [Val(0x5f), Val(0x13), Val(0), Bits(13, 0)]);
        if used == 0 {
            return;
        }
        let size = ord_size();
        let mut tokens = Vec::new();
        for (o, p) in self.perm.iter().enumerate() {
            let Some(p) = p else { continue };
            let llf = size[o] / 64;
            for perm in p {
                tokenize_permutation(perm, llf, &mut tokens);
            }
        }
        let opts = CodeOptions { max_clusters: PERM_CTX, threads, ..CodeOptions::default() };
        write_stream(&tokens, PERM_CTX, opts, w);
    }
}

fn perm_ctx(v: u32) -> u32 {
    // Token of the (0, 0, 0) hybrid-uint configuration, capped.
    let t = if v == 0 { 0 } else { 32 - v.leading_zeros() };
    t.min(PERM_CTX as u32 - 1)
}

/// libjxl's TokenizePermutation: the Lehmer code of `perm`, without its
/// first `skip` (identity) entries and trailing zeros.
fn tokenize_permutation(perm: &[u16], skip: usize, out: &mut Vec<Token>) {
    let n = perm.len();
    let mut fenwick = vec![0u32; n + 1];
    let mut lehmer = vec![0u32; n];
    for (l, &s) in lehmer.iter_mut().zip(perm) {
        let s = usize::from(s);
        let mut used_below = 0;
        let mut i = s + 1;
        while i != 0 {
            used_below += fenwick[i];
            i &= i - 1;
        }
        *l = s as u32 - used_below;
        let mut i = s + 1;
        while i <= n {
            fenwick[i] += 1;
            i += i.isolate_lowest_one();
        }
    }
    let mut end = n;
    while end > skip && lehmer[end - 1] == 0 {
        end -= 1;
    }
    out.push(Token::new(perm_ctx(n as u32), (end - skip) as u32));
    let mut last = 0;
    for &l in &lehmer[skip..end] {
        out.push(Token::new(perm_ctx(last), l));
        last = l;
    }
}

/// Origins of the varblocks of group `g`, in coding order.
fn group_blocks<'a>(layout: &Layout, acs: &'a [BlockAcs], bw: usize, g: usize) -> impl Iterator<Item = Acs> + 'a {
    let (gx, gy, gw, gh) = layout.group_rect(g);
    let (bx0, by0) = (gx / 8, gy / 8);
    let (gbw, gbh) = (gw.div_ceil(8), gh.div_ceil(8));
    (0..gbh).flat_map(move |by| (0..gbw).map(move |bx| (bx0 + bx, by0 + by))).filter_map(move |(x, y)| {
        let b = acs[y * bw + x];
        ((b.ox as usize, b.oy as usize) == (x, y)).then_some(b.acs)
    })
}

/// Channels in token order (Y, X, B).
const CHANNELS: [usize; 3] = [1, 0, 2];

/// Walks the tokens of one group: for every block and channel, `f` gets the
/// transform, the channel, the non-zero count token (whose context also
/// identifies the block context) and the coefficient tokens (natural order
/// after the LLF positions).
fn walk(tokens: &[Token], blocks: impl Iterator<Item = Acs>, mut f: impl FnMut(Acs, usize, Token, &[Token])) {
    let mut i = 0;
    for acs in blocks {
        for c in CHANNELS {
            let nz_tok = tokens[i];
            i += 1;
            let mut remaining = nz_tok.value();
            let start = i;
            while remaining > 0 {
                if tokens[i].value() != 0 {
                    remaining -= 1;
                }
                i += 1;
            }
            f(acs, c, nz_tok, &tokens[start..i]);
        }
    }
    debug_assert_eq!(i, tokens.len());
}

/// Counts non-zero coefficients per order class, channel and natural index
/// and derives the orders.
pub fn compute(
    tokens: &[&[Token]],
    layout: &Layout,
    acs: &[BlockAcs],
    bw: usize,
    num_ctx: u32,
    threads: usize,
) -> CoeffOrders {
    let size = ord_size();
    let mut offset = [0usize; NUM_ORDS + 1];
    for o in 0..NUM_ORDS {
        offset[o + 1] = offset[o] + 3 * size[o];
    }
    let total = offset[NUM_ORDS];
    let ng = tokens.len();
    let chunks = threads.clamp(1, ng);
    let per = ng.div_ceil(chunks);
    let parts = par_map(ng.div_ceil(per), threads, |k| {
        let mut nonzero = vec![0u32; total];
        let mut blocks = [0u32; NUM_ORDS];
        for g in k * per..((k + 1) * per).min(ng) {
            walk(tokens[g], group_blocks(layout, acs, bw, g), |a, c, _, coeffs| {
                let o = a.order_id();
                let nb = a.num_blocks();
                if c == 0 {
                    blocks[o] += 1;
                }
                let base = offset[o] + c * size[o] + nb;
                for (k, t) in coeffs.iter().enumerate() {
                    nonzero[base + k] += u32::from(t.value() != 0);
                }
            });
        }
        (nonzero, blocks)
    });
    let mut nonzero = vec![0u32; total];
    let mut blocks = [0u32; NUM_ORDS];
    for (nzp, bp) in parts {
        nonzero.iter_mut().zip(&nzp).for_each(|(a, b)| *a += b);
        blocks.iter_mut().zip(&bp).for_each(|(a, b)| *a += b);
    }
    // Candidate orders per class and channel: the natural one, and sorts by
    // the non-zero rate quantised to a few levels (cheaper permutations that
    // ignore insignificant differences) or not at all.
    struct Slot {
        o: usize,
        c: usize,
        perm: Vec<u16>,
    }
    let mut slots: Vec<Slot> = Vec::new();
    let mut slot_range = [[(0usize, 0usize); 3]; NUM_ORDS];
    for o in 0..NUM_ORDS {
        if blocks[o] == 0 {
            continue;
        }
        let n = size[o];
        let llf = n / 64;
        let b = f64::from(blocks[o]);
        for c in 0..3 {
            let nzc = &nonzero[offset[o] + c * n..offset[o] + (c + 1) * n];
            let first = slots.len();
            slots.push(Slot { o, c, perm: (0..n as u16).collect() });
            for levels in LEVELS {
                let mut p: Vec<u16> = (0..n as u16).collect();
                let key = |j: u16| {
                    let v = f64::from(nzc[usize::from(j)]);
                    if levels == 0.0 { v as u64 } else { (v / b * levels) as u64 }
                };
                // Stable: equally frequent positions keep their natural order.
                p[llf..].sort_by_key(|&j| std::cmp::Reverse(key(j)));
                if slots[first..].iter().all(|s| s.perm != p) {
                    slots.push(Slot { o, c, perm: p });
                }
            }
            slot_range[o][c] = (first, slots.len());
        }
    }
    if slots.iter().all(|s| s.perm.iter().enumerate().all(|(i, &j)| usize::from(j) == i)) {
        return CoeffOrders::natural();
    }
    // Histograms of the coefficient tokens under every candidate, from an
    // evenly spaced sample of groups (the decision only needs the rough size
    // of the gain); the costs are then scaled up to all blocks of the class.
    let ns = slots.len();
    let stride = (ng / EST_GROUPS).max(1);
    let sample: Vec<usize> = (0..ng).step_by(stride).collect();
    // Within the sampled groups, every `block_stride`-th varblock, so that
    // about EST_POSITIONS coefficient positions are looked at per candidate.
    let positions: usize = sample
        .iter()
        .map(|&g| {
            let (_, _, gw, gh) = layout.group_rect(g);
            3 * gw.div_ceil(8) * gh.div_ceil(8) * 64
        })
        .sum();
    let block_stride = positions.div_ceil(EST_POSITIONS).max(1);
    // At most eight partial histograms are alive at a time.
    let per = sample.len().div_ceil(threads.clamp(1, sample.len()).min(8));
    let parts = par_map(sample.len().div_ceil(per), threads, |k| {
        let mut hist = vec![0u32; ns * ZERO_DENSITY_CTX * EST_SYMS];
        let mut sampled = [0u32; NUM_ORDS];
        let mut natural = vec![0u32; 64 * 64];
        for &g in &sample[k * per..((k + 1) * per).min(sample.len())] {
            // Per group, so that the sample does not depend on the threads.
            let mut seen = 0usize;
            walk(tokens[g], group_blocks(layout, acs, bw, g), |a, c, nz_tok, coeffs| {
                let o = a.order_id();
                if c == 0 {
                    seen += 1;
                }
                if !seen.is_multiple_of(block_stride) {
                    return;
                }
                if c == 0 {
                    sampled[o] += 1;
                }
                if coeffs.is_empty() {
                    return;
                }
                let nb = a.num_blocks();
                let natural = &mut natural[..size[o]];
                natural.fill(0);
                for (k, t) in coeffs.iter().enumerate() {
                    natural[nb + k] = t.value();
                }
                let base = coeff_ctx_base(num_ctx, nz_tok.ctx() % num_ctx);
                let log = nb.trailing_zeros();
                let (s0, s1) = slot_range[o][c];
                for (si, slot) in slots.iter().enumerate().take(s1).skip(s0) {
                    let perm = &slot.perm[nb..];
                    let h = &mut hist[si * ZERO_DENSITY_CTX * EST_SYMS..(si + 1) * ZERO_DENSITY_CTX * EST_SYMS];
                    emit_coeffs(
                        base,
                        nz_tok.value(),
                        log,
                        perm.len(),
                        |idx| natural[usize::from(perm[idx])],
                        |t| {
                            // Values up to 15 are their own hybrid-uint token;
                            // larger ones are rare and pooled.
                            h[(t.ctx() - base) as usize * EST_SYMS + (t.value() as usize).min(EST_SYMS - 1)] += 1;
                        },
                    );
                }
            });
        }
        (hist, sampled)
    });
    let mut hist = vec![0u32; ns * ZERO_DENSITY_CTX * EST_SYMS];
    let mut sampled = [0u32; NUM_ORDS];
    for (h, sm) in parts {
        hist.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
        sampled.iter_mut().zip(&sm).for_each(|(a, b)| *a += b);
    }
    let coef_bits: Vec<f64> = hist
        .chunks(ZERO_DENSITY_CTX * EST_SYMS)
        .zip(&slots)
        .map(|(h, slot)| {
            let scale = f64::from(blocks[slot.o]) / f64::from(sampled[slot.o].max(1));
            h.chunks(EST_SYMS).map(entropy_bits).sum::<f64>() * scale
        })
        .collect();
    let mut out = CoeffOrders::natural();
    let mut total_gain = 0.0;
    for o in 0..NUM_ORDS {
        if blocks[o] == 0 {
            continue;
        }
        let mut chosen: [Option<usize>; 3] = [None; 3];
        for c in 0..3 {
            let (s0, s1) = slot_range[o][c];
            let mut best = (coef_bits[s0], s0);
            for si in s0 + 1..s1 {
                let cost = coef_bits[si] + perm_bits(&slots[si].perm, size[o] / 64);
                if cost < best.0 {
                    best = (cost, si);
                }
            }
            if best.1 != s0 {
                total_gain += coef_bits[s0] - best.0;
                chosen[c] = Some(best.1);
            }
        }
        if chosen.iter().any(Option::is_some) {
            debug_assert!(slots[slot_range[o][0].0].o == o && slots[slot_range[o][0].0].c == 0);
            out.perm[o] = Some(std::array::from_fn(|c| slots[chosen[c].unwrap_or(slot_range[o][c].0)].perm.clone()));
        }
    }
    // The permutation stream has a fixed cost (its code description).
    if total_gain < STREAM_OVERHEAD_BITS {
        return CoeffOrders::natural();
    }
    out
}

/// Quantisation levels of the non-zero rate for the candidate orders (0: exact counts).
const LEVELS: [f64; 5] = [8.0, 16.0, 32.0, 64.0, 0.0];
/// Coefficient contexts per block context.
const ZERO_DENSITY_CTX: usize = 458;
/// Symbols tracked by the estimate (larger ones are pooled).
const EST_SYMS: usize = 17;
/// Groups sampled for the estimate.
const EST_GROUPS: usize = 16;
/// Coefficient positions (of those groups) sampled per candidate order.
const EST_POSITIONS: usize = 1 << 19;
/// Rough cost of the permutation stream's code description.
const STREAM_OVERHEAD_BITS: f64 = 400.0;

/// Shannon cost of a histogram, in bits.
fn entropy_bits(h: &[u32]) -> f64 {
    let n: u32 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let ln = f64::from(n).log2();
    h.iter().filter(|&&c| c > 0).map(|&c| f64::from(c) * (ln - f64::from(c).log2())).sum()
}

/// Estimated cost of coding `perm` (its own adaptive-entropy cost plus raw
/// bits and a small charge per used symbol for the distribution).
fn perm_bits(perm: &[u16], skip: usize) -> f64 {
    let mut tokens = Vec::new();
    tokenize_permutation(perm, skip, &mut tokens);
    let mut hist = vec![0u32; PERM_CTX * 64];
    let mut raw = 0u32;
    for t in &tokens {
        let (sym, nbits, _) = HybridUint::DEFAULT.encode(t.value());
        hist[t.ctx() as usize * 64 + (sym as usize).min(63)] += 1;
        raw += nbits;
    }
    let used = hist.iter().filter(|&&c| c > 0).count();
    hist.chunks(64).map(entropy_bits).sum::<f64>() + f64::from(raw) + 6.0 * used as f64
}

/// Re-tokenises one group (coded in the natural order) in `orders`.
pub fn retokenize(
    tokens: &[Token],
    layout: &Layout,
    acs: &[BlockAcs],
    bw: usize,
    g: usize,
    num_ctx: u32,
    orders: &CoeffOrders,
) -> Vec<Token> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut natural = vec![0u32; 64 * 64];
    walk(tokens, group_blocks(layout, acs, bw, g), |a, c, nz_tok, coeffs| {
        out.push(nz_tok);
        let nb = a.num_blocks();
        if coeffs.is_empty() {
            return;
        }
        let Some(perm) = &orders.perm[a.order_id()] else {
            out.extend_from_slice(coeffs);
            return;
        };
        let perm = &perm[c][nb..];
        let natural = &mut natural[..perm.len() + nb];
        natural.fill(0);
        for (k, t) in coeffs.iter().enumerate() {
            natural[nb + k] = t.value();
        }
        let base = coeff_ctx_base(num_ctx, nz_tok.ctx() % num_ctx);
        let log = nb.trailing_zeros();
        emit_coeffs(base, nz_tok.value(), log, perm.len(), |idx| natural[usize::from(perm[idx])], |t| out.push(t));
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inverse of `tokenize_permutation`, as the decoder reads it.
    fn decode(tokens: &[Token], size: usize, skip: usize) -> Vec<u16> {
        assert_eq!(tokens[0].ctx(), perm_ctx(size as u32));
        let end = skip + tokens[0].value() as usize;
        let mut lehmer = vec![0u32; size];
        let mut last = 0;
        for (i, t) in tokens[1..].iter().enumerate() {
            assert_eq!(t.ctx(), perm_ctx(last));
            lehmer[skip + i] = t.value();
            last = t.value();
        }
        assert_eq!(tokens.len() - 1, end - skip);
        let mut unused: Vec<u16> = (0..size as u16).collect();
        lehmer.iter().map(|&l| unused.remove(l as usize)).collect()
    }

    #[test]
    fn permutations_round_trip() {
        let mut seed = 12345u32;
        for &(size, skip) in &[(64usize, 1usize), (256, 4), (1024, 16), (4096, 64)] {
            for shuffle in [0, 3, size / 2, size] {
                let mut p: Vec<u16> = (0..size as u16).collect();
                for _ in 0..shuffle {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let a = skip + (seed >> 8) as usize % (size - skip);
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let b = skip + (seed >> 8) as usize % (size - skip);
                    p.swap(a, b);
                }
                let mut tokens = Vec::new();
                tokenize_permutation(&p, skip, &mut tokens);
                assert_eq!(decode(&tokens, size, skip), p, "size {size} shuffle {shuffle}");
            }
        }
    }

    #[test]
    fn identity_costs_one_token() {
        let mut tokens = Vec::new();
        tokenize_permutation(&(0..64).collect::<Vec<u16>>(), 1, &mut tokens);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value(), 0);
    }
}
