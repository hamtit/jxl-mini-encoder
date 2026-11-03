//! Entropy coding: hybrid-uint tokenisation, context clustering, ANS
//! distributions and the interleaved ANS bitstream.

mod ans;

use crate::bit_writer::{BitWriter, bits_for};
use crate::parallel::par_map;
use ans::{AnsDist, estimate_cost};

/// A value to be entropy coded in a given context.
///
/// Packed to 6 bytes (instead of 8): a frame holds one token per sample, so
/// this is a quarter less memory for the token buffers. Fields are read by
/// value only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, packed)]
pub struct Token {
    ctx: u16,
    value: u32,
}

impl Token {
    #[inline]
    pub fn new(ctx: u32, value: u32) -> Self {
        debug_assert!(ctx <= u32::from(u16::MAX));
        Token {
            ctx: ctx as u16,
            value,
        }
    }

    #[inline]
    pub fn ctx(self) -> u32 {
        u32::from(self.ctx)
    }

    #[inline]
    pub fn value(self) -> u32 {
        self.value
    }
}

/// Map a signed integer to the unsigned value JPEG XL codes.
#[inline]
pub fn pack_signed(v: i32) -> u32 {
    if v >= 0 {
        (v as u32) << 1
    } else {
        ((-(v as i64)) as u32) * 2 - 1
    }
}

/// Hybrid integer configuration: values below `2^split_exp` are coded as
/// their own token, larger ones as a token holding the exponent plus the top
/// `msb` and bottom `lsb` bits, the rest are raw bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridUint {
    split_exp: u32,
    msb: u32,
    lsb: u32,
}

impl HybridUint {
    pub const DEFAULT: HybridUint = HybridUint::new(4, 2, 0);

    pub const fn new(split_exp: u32, msb: u32, lsb: u32) -> Self {
        HybridUint {
            split_exp,
            msb,
            lsb,
        }
    }

    /// (token, number of raw bits, raw bits)
    #[inline]
    pub fn encode(self, v: u32) -> (u32, u32, u32) {
        let split = 1u32 << self.split_exp;
        if v < split {
            return (v, 0, 0);
        }
        let n = 31 - v.leading_zeros();
        let m = v - (1 << n);
        let token = split
            + ((n - self.split_exp) << (self.msb + self.lsb))
            + ((m >> (n - self.msb)) << self.lsb)
            + (m & ((1 << self.lsb) - 1));
        let nbits = n - self.msb - self.lsb;
        let bits = (m >> self.lsb) & ((1u32 << nbits).wrapping_sub(1));
        (token, nbits, bits)
    }

    fn write(self, w: &mut BitWriter, log_alpha: u32) {
        w.write(bits_for(log_alpha), u64::from(self.split_exp));
        if self.split_exp != log_alpha {
            w.write(bits_for(self.split_exp), u64::from(self.msb));
            w.write(bits_for(self.split_exp - self.msb), u64::from(self.lsb));
        }
    }
}

const CANDIDATE_CONFIGS: [HybridUint; 14] = [
    HybridUint::new(4, 2, 0),
    HybridUint::new(4, 1, 0),
    HybridUint::new(4, 0, 0),
    HybridUint::new(4, 1, 1),
    HybridUint::new(3, 1, 0),
    HybridUint::new(3, 2, 0),
    HybridUint::new(5, 2, 0),
    HybridUint::new(5, 1, 0),
    HybridUint::new(5, 2, 1),
    HybridUint::new(6, 2, 0),
    HybridUint::new(2, 0, 0),
    HybridUint::new(1, 0, 0),
    HybridUint::new(0, 0, 0),
    HybridUint::new(2, 1, 0),
];

/// How hard to try when building the code.
#[derive(Clone, Copy, Debug)]
pub struct CodeOptions {
    pub max_clusters: usize,
    /// Try several hybrid-uint configurations per cluster.
    pub tune_configs: bool,
    /// Worker threads for the histogram and tuning passes.
    pub threads: usize,
}

impl Default for CodeOptions {
    fn default() -> Self {
        CodeOptions {
            max_clusters: 64,
            tune_configs: true,
            threads: 1,
        }
    }
}

#[derive(Debug, Clone)]
struct Cluster {
    config: HybridUint,
    dist: AnsDist,
}

/// A complete entropy code: context map, per-cluster hybrid-uint configs and
/// ANS distributions.
#[derive(Debug, Clone)]
pub struct EntropyCode {
    context_map: Vec<u8>,
    clusters: Vec<Cluster>,
    log_alpha: u32,
}

fn histogram(tokens: impl Iterator<Item = u32>, hist: &mut Vec<u32>) {
    for t in tokens {
        let t = t as usize;
        if t >= hist.len() {
            hist.resize(t + 1, 0);
        }
        hist[t] += 1;
    }
}

fn add_into(dst: &mut Vec<u32>, src: &[u32]) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Cut the token streams into at most about `threads` pieces of similar size.
fn split_pieces<'a>(streams: &[&'a [Token]], threads: usize) -> Vec<&'a [Token]> {
    let total: usize = streams.iter().map(|s| s.len()).sum();
    if threads <= 1 {
        return streams.to_vec();
    }
    let max = total.div_ceil(threads * 2).max(1 << 14);
    let mut out = Vec::new();
    for s in streams {
        out.extend(s.chunks(max));
    }
    out
}

/// Count repeated values once, so configuration tuning need not encode every
/// sample fourteen times. Bound the temporary storage to 4 MiB; unusual large
/// values or many wide alphabets fall back to visiting the original tokens.
fn value_histograms(
    pieces: &[&[Token]],
    context_map: &[u8],
    num_clusters: usize,
) -> Option<Vec<Vec<u32>>> {
    let mut hists = vec![Vec::new(); num_clusters];
    let mut entries = 0;
    for piece in pieces {
        for &t in *piece {
            let value = t.value() as usize;
            if value >= 1 << 16 {
                return None;
            }
            let hist = &mut hists[context_map[t.ctx() as usize] as usize];
            if value >= hist.len() {
                let len = (value + 1).next_power_of_two().max(4);
                entries += len - hist.len();
                if entries > 1 << 20 {
                    return None;
                }
                hist.resize(len, 0);
            }
            hist[value] += 1;
        }
    }
    // Scanning sparse alphabets must not cost more than the original pass.
    let samples: usize = pieces.iter().map(|p| p.len()).sum();
    (entries <= samples).then_some(hists)
}

fn visit_values(
    pieces: &[&[Token]],
    context_map: &[u8],
    hists: Option<&[Vec<u32>]>,
    mut visit: impl FnMut(usize, u32, u32),
) {
    if let Some(hists) = hists {
        for (c, hist) in hists.iter().enumerate() {
            for (value, &count) in hist.iter().enumerate() {
                if count != 0 {
                    visit(c, value as u32, count);
                }
            }
        }
    } else {
        for piece in pieces {
            for &t in *piece {
                visit(context_map[t.ctx() as usize] as usize, t.value(), 1);
            }
        }
    }
}

fn merged_cost(a: &[u32], b: &[u32]) -> f64 {
    let mut m = a.to_vec();
    add_into(&mut m, b);
    estimate_cost(&m)
}

/// Group contexts with similar statistics. Returns the context map.
fn cluster(hists: &[Vec<u32>], max_clusters: usize) -> (Vec<u8>, usize) {
    let max_clusters = max_clusters.clamp(1, 255);
    let used: Vec<usize> = (0..hists.len())
        .filter(|&i| hists[i].iter().any(|&c| c > 0))
        .collect();
    if used.is_empty() {
        return (vec![0; hists.len()], 1);
    }
    let costs: Vec<f64> = hists.iter().map(|h| estimate_cost(h)).collect();
    // Seed clusters: farthest-first among the used contexts.
    let mut cluster_hist: Vec<Vec<u32>> = Vec::new();
    let mut assign = vec![0usize; hists.len()];
    if used.len() <= max_clusters {
        for (k, &i) in used.iter().enumerate() {
            cluster_hist.push(hists[i].clone());
            assign[i] = k;
        }
    } else {
        let first = *used
            .iter()
            .max_by(|&&a, &&b| costs[a].total_cmp(&costs[b]))
            .unwrap();
        let mut seeds = vec![first];
        let mut dist: Vec<f64> = used
            .iter()
            .map(|&i| merged_cost(&hists[i], &hists[first]) - costs[i] - costs[first])
            .collect();
        while seeds.len() < max_clusters {
            let (k, &d) = dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            if d <= 0.0 {
                break;
            }
            let s = used[k];
            seeds.push(s);
            for (j, &i) in used.iter().enumerate() {
                let dd = merged_cost(&hists[i], &hists[s]) - costs[i] - costs[s];
                if dd < dist[j] {
                    dist[j] = dd;
                }
            }
        }
        cluster_hist = vec![Vec::new(); seeds.len()];
        for &i in &used {
            // Evaluate each candidate once. min_by's comparator would
            // recompute the current best histogram for every challenger.
            let best = (0..seeds.len())
                .map(|k| {
                    (
                        k,
                        merged_cost(&hists[i], &hists[seeds[k]]) - costs[seeds[k]],
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            assign[i] = best;
            add_into(&mut cluster_hist[best], &hists[i]);
        }
    }
    // Greedy pairwise merging while it saves bits.
    let n = cluster_hist.len();
    let mut alive = vec![true; n];
    let mut ccost: Vec<f64> = cluster_hist.iter().map(|h| estimate_cost(h)).collect();
    let mut delta = vec![vec![f64::INFINITY; n]; n];
    for a in 0..n {
        for b in a + 1..n {
            delta[a][b] = merged_cost(&cluster_hist[a], &cluster_hist[b]) - ccost[a] - ccost[b];
        }
    }
    let mut remap: Vec<usize> = (0..n).collect();
    loop {
        let mut best = (f64::INFINITY, 0, 0);
        for a in 0..n {
            if !alive[a] {
                continue;
            }
            for b in a + 1..n {
                if alive[b] && delta[a][b] < best.0 {
                    best = (delta[a][b], a, b);
                }
            }
        }
        let (d, a, b) = best;
        if d.is_nan() || d >= 0.0 {
            break;
        }
        let hb = std::mem::take(&mut cluster_hist[b]);
        add_into(&mut cluster_hist[a], &hb);
        alive[b] = false;
        for r in remap.iter_mut() {
            if *r == b {
                *r = a;
            }
        }
        ccost[a] = estimate_cost(&cluster_hist[a]);
        for c in 0..n {
            if c != a && alive[c] {
                let v = merged_cost(&cluster_hist[a], &cluster_hist[c]) - ccost[a] - ccost[c];
                let (x, y) = if a < c { (a, c) } else { (c, a) };
                delta[x][y] = v;
            }
        }
    }
    // Renumber in order of first use so the map has no holes.
    let mut new_id = vec![usize::MAX; n];
    let mut next = 0;
    let mut map = vec![0u8; hists.len()];
    for i in 0..hists.len() {
        if !hists[i].iter().any(|&c| c > 0) {
            continue;
        }
        let c = remap[assign[i]];
        if new_id[c] == usize::MAX {
            new_id[c] = next;
            next += 1;
        }
        map[i] = new_id[c] as u8;
    }
    (map, next.max(1))
}

impl EntropyCode {
    /// Build a code for `num_ctx` contexts from every token that will be
    /// written with it (possibly split over several streams).
    pub fn build(streams: &[&[Token]], num_ctx: usize, opts: CodeOptions) -> Self {
        let num_ctx = num_ctx.max(1);
        assert!(
            num_ctx <= 1 << 16,
            "too many contexts for 16-bit token contexts"
        );
        let pieces = split_pieces(streams, opts.threads);
        let mut hists: Vec<Vec<u32>> = vec![Vec::new(); num_ctx];
        let per_worker = pieces.len().div_ceil(opts.threads.max(1)).max(1);
        let groups: Vec<&[&[Token]]> = pieces.chunks(per_worker).collect();
        for part in par_map(groups.len(), opts.threads, |g| {
            let mut hists: Vec<Vec<u32>> = vec![Vec::new(); num_ctx];
            for piece in groups[g] {
                for t in piece.iter() {
                    let (tok, _, _) = HybridUint::DEFAULT.encode(t.value());
                    let h = &mut hists[t.ctx() as usize];
                    let tok = tok as usize;
                    if tok >= h.len() {
                        h.resize(tok + 1, 0);
                    }
                    h[tok] += 1;
                }
            }
            hists
        }) {
            for (d, s) in hists.iter_mut().zip(&part) {
                add_into(d, s);
            }
        }
        let (context_map, num_clusters) = if num_ctx == 1 {
            (vec![0], 1)
        } else {
            cluster(&hists, opts.max_clusters)
        };
        // Pick a hybrid-uint config per cluster.
        let mut configs = vec![HybridUint::DEFAULT; num_clusters];
        if opts.tune_configs {
            let values = value_histograms(&pieces, &context_map, num_clusters);
            // Integer histogram counts and raw-bit totals remain exact, so
            // aggregating repeated values preserves configuration selection.
            let costs: Vec<(Vec<Option<f64>>, Vec<bool>)> =
                par_map(CANDIDATE_CONFIGS.len(), opts.threads, |ci| {
                    let cfg = CANDIDATE_CONFIGS[ci];
                    let mut hist: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
                    let mut extra = vec![0u64; num_clusters];
                    let mut used = vec![false; num_clusters];
                    let mut bad = vec![false; num_clusters];
                    visit_values(
                        &pieces,
                        &context_map,
                        values.as_deref(),
                        |c, value, count| {
                            used[c] = true;
                            let (tok, n, _) = cfg.encode(value);
                            if tok >= 256 {
                                bad[c] = true;
                                return;
                            }
                            extra[c] += u64::from(n) * u64::from(count);
                            let h = &mut hist[c];
                            if tok as usize >= h.len() {
                                h.resize(tok as usize + 1, 0);
                            }
                            h[tok as usize] += count;
                        },
                    );
                    let cost = (0..num_clusters)
                        .map(|c| (!bad[c]).then(|| estimate_cost(&hist[c]) + extra[c] as f64))
                        .collect();
                    (cost, used)
                });
            for c in 0..num_clusters {
                if !costs[0].1[c] {
                    continue;
                }
                let mut best = (f64::INFINITY, HybridUint::DEFAULT);
                for (ci, &cfg) in CANDIDATE_CONFIGS.iter().enumerate() {
                    if let Some(cost) = costs[ci].0[c]
                        && cost < best.0
                    {
                        best = (cost, cfg);
                    }
                }
                configs[c] = best.1;
            }
        }
        // Final histograms.
        let mut chists: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
        for part in par_map(groups.len(), opts.threads, |g| {
            let mut chists: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
            for piece in groups[g] {
                for t in piece.iter() {
                    let c = context_map[t.ctx() as usize] as usize;
                    let (tok, _, _) = configs[c].encode(t.value());
                    histogram(std::iter::once(tok), &mut chists[c]);
                }
            }
            chists
        }) {
            for (d, s) in chists.iter_mut().zip(&part) {
                add_into(d, s);
            }
        }
        let max_tok = chists.iter().map(|h| h.len()).max().unwrap_or(1);
        let log_alpha = (5..=8)
            .find(|&l| (1usize << l) >= max_tok)
            .expect("token alphabet too large");
        let clusters = chists
            .iter()
            .zip(configs)
            .map(|(h, config)| Cluster {
                config,
                dist: AnsDist::new(h, log_alpha),
            })
            .collect();
        EntropyCode {
            context_map,
            clusters,
            log_alpha,
        }
    }

    /// Write the code description (LZ77 flag, context map, configs and
    /// distributions).
    pub fn write_header(&self, w: &mut BitWriter) {
        w.write_bool(false); // no LZ77
        self.write_context_map(w);
        w.write_bool(false); // ANS, not prefix codes
        w.write(2, u64::from(self.log_alpha - 5));
        for c in &self.clusters {
            c.config.write(w, self.log_alpha);
        }
        for c in &self.clusters {
            c.dist.write(w);
        }
    }

    fn write_context_map(&self, w: &mut BitWriter) {
        let n = self.context_map.len();
        if n == 1 {
            return;
        }
        let max = *self.context_map.iter().max().unwrap() as u32;
        let nbits = bits_for(max);
        let simple_bits = if nbits <= 3 {
            Some(3 + n * nbits as usize)
        } else {
            None
        };
        // Entropy-coded alternative.
        let tokens: Vec<Token> = self
            .context_map
            .iter()
            .map(|&c| Token::new(0, u32::from(c)))
            .collect();
        let code = EntropyCode::build(
            &[&tokens],
            1,
            CodeOptions {
                max_clusters: 1,
                tune_configs: true,
                threads: 1,
            },
        );
        let mut coded = BitWriter::new();
        coded.write_bool(false); // no MTF
        code.write_header(&mut coded);
        code.write_tokens(&tokens, &mut coded);
        match simple_bits {
            Some(bits) if bits <= coded.bits_written() => {
                w.write_bool(true);
                w.write(2, u64::from(nbits));
                for &c in &self.context_map {
                    w.write(nbits, u64::from(c));
                }
            }
            _ => {
                w.write_bool(false);
                w.append(&coded);
            }
        }
    }

    /// Write an ANS stream of tokens (initial state followed by interleaved
    /// renormalisation words and raw bits).
    pub fn write_tokens(&self, tokens: &[Token], w: &mut BitWriter) {
        // The reverse pass only needs the symbol, and the forward pass only
        // needs the raw bits. Compute each directly instead of retaining a
        // intermediate tuple for every token.
        let mut refill: Vec<Option<u16>> = vec![None; tokens.len()];
        let mut state: u32 = 0x130000;
        for (i, &t) in tokens.iter().enumerate().rev() {
            let c = &self.clusters[self.context_map[t.ctx() as usize] as usize];
            let (s, _, _) = c.config.encode(t.value());
            let dist = &c.dist;
            let freq = u32::from(dist.freq[s as usize]);
            debug_assert!(freq > 0, "symbol {s} has zero frequency");
            if (state >> 20) >= freq {
                refill[i] = Some(state as u16);
                state >>= 16;
            }
            state = ((state / freq) << 12) + dist.slot(s as usize, state % freq);
        }
        w.write(32, u64::from(state));
        for (&t, r) in tokens.iter().zip(&refill) {
            let c = &self.clusters[self.context_map[t.ctx() as usize] as usize];
            let (_, n, b) = c.config.encode(t.value());
            if let Some(r) = r {
                w.write(16, u64::from(*r));
            }
            w.write(n, u64::from(b));
        }
    }
}

/// Code a single stream with its own code: header followed by the tokens.
pub fn write_stream(tokens: &[Token], num_ctx: usize, opts: CodeOptions, w: &mut BitWriter) {
    let code = EntropyCode::build(&[tokens], num_ctx, opts);
    code.write_header(w);
    code.write_tokens(tokens, w);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregated_tuning_matches_individual_values() {
        let tokens: Vec<_> = (0..100_000)
            .map(|i| Token::new(i % 3, (i * 17) % 513))
            .chain([Token::new(2, 65_535)])
            .collect();
        let pieces = [&tokens[..123], &tokens[123..]];
        let map = [0, 1, 0]; // Include merging and an unused cluster.
        let hists = value_histograms(&pieces, &map, 3).unwrap();
        for cfg in CANDIDATE_CONFIGS {
            let evaluate = |values: Option<&[Vec<u32>]>| {
                let mut hist = [[0u32; 256]; 3];
                let mut extra = [0u64; 3];
                let mut bad = [false; 3];
                let mut used = [false; 3];
                visit_values(&pieces, &map, values, |c, value, count| {
                    used[c] = true;
                    let (tok, n, _) = cfg.encode(value);
                    if tok >= 256 {
                        bad[c] = true;
                    } else {
                        hist[c][tok as usize] += count;
                        extra[c] += u64::from(n) * u64::from(count);
                    }
                });
                (hist, extra, bad, used)
            };
            assert_eq!(evaluate(Some(&hists)), evaluate(None), "{cfg:?}");
        }
        assert!(value_histograms(&[&[]], &map, 3).is_some());
        assert!(value_histograms(&[&[Token::new(0, 4095)]], &map, 3).is_none());
        assert!(value_histograms(&[&[Token::new(0, 65_536)]], &map, 3).is_none());
        let wide: Vec<_> = (0..17).map(|c| Token::new(c, 65_535)).collect();
        let map: Vec<_> = (0..17).collect();
        assert!(value_histograms(&[&wide], &map, 17).is_none());
    }

    /// Retain the former materialized algorithm as a byte-for-byte reference.
    fn write_tokens_reference(code: &EntropyCode, tokens: &[Token]) -> Vec<u8> {
        let coded: Vec<_> = tokens
            .iter()
            .map(|t| {
                let c = code.context_map[t.ctx() as usize] as usize;
                let (symbol, n, bits) = code.clusters[c].config.encode(t.value());
                (c, symbol, n, bits)
            })
            .collect();
        let mut refill = vec![None; coded.len()];
        let mut state = 0x130000u32;
        for (i, &(c, symbol, _, _)) in coded.iter().enumerate().rev() {
            let dist = &code.clusters[c].dist;
            let freq = u32::from(dist.freq[symbol as usize]);
            if (state >> 20) >= freq {
                refill[i] = Some(state as u16);
                state >>= 16;
            }
            state = ((state / freq) << 12) + dist.slot(symbol as usize, state % freq);
        }
        let mut w = BitWriter::new();
        w.write(32, u64::from(state));
        for (&(_, _, n, bits), refill) in coded.iter().zip(refill) {
            if let Some(word) = refill {
                w.write(16, u64::from(word));
            }
            w.write(n, u64::from(bits));
        }
        w.finish()
    }

    #[test]
    fn token_output_matches_materialized_reference() {
        let mut state = 12345u32;
        let mixed: Vec<_> = (0..20_000)
            .map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                Token::new(i % 3, state >> (i % 25 + 1))
            })
            .collect();
        for tokens in [vec![], vec![Token::new(0, 0); 4096], mixed] {
            for tune_configs in [false, true] {
                let code = EntropyCode::build(
                    &[&tokens],
                    3,
                    CodeOptions {
                        tune_configs,
                        ..CodeOptions::default()
                    },
                );
                let expected = write_tokens_reference(&code, &tokens);
                let mut w = BitWriter::new();
                code.write_tokens(&tokens, &mut w);
                assert_eq!(w.finish(), expected);
            }
        }
    }

    #[test]
    fn hybrid_uint_roundtrip() {
        for cfg in CANDIDATE_CONFIGS {
            for v in (0..5000u32).chain([u32::MAX >> 1, 123_456_789]) {
                let (tok, n, bits) = cfg.encode(v);
                // decode as in the spec
                let split = 1 << cfg.split_exp;
                let dec = if tok < split {
                    tok
                } else {
                    let nn = cfg.split_exp - (cfg.msb + cfg.lsb)
                        + ((tok - split) >> (cfg.msb + cfg.lsb));
                    assert_eq!(nn, n);
                    let low = tok & ((1 << cfg.lsb) - 1);
                    let t = (tok >> cfg.lsb) & ((1 << cfg.msb) - 1) | (1 << cfg.msb);
                    ((((t as u64) << nn) | bits as u64) << cfg.lsb | low as u64) as u32
                };
                assert_eq!(dec, v, "{cfg:?}");
            }
        }
    }

    #[test]
    fn pack() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-2), 3);
    }
}
