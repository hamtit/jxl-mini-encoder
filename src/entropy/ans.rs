//! ANS distributions: normalisation to 12-bit precision, their bitstream
//! encoding, and the alias table that maps (symbol, offset) to ANS slots
//! exactly as a decoder rebuilds it.

use crate::bit_writer::BitWriter;

pub const LOG_TAB: u32 = 12;
pub const TAB: u32 = 1 << LOG_TAB;

/// A normalised distribution (frequencies summing to 4096) plus the reverse
/// alias table the encoder needs.
#[derive(Debug, Clone)]
pub struct AnsDist {
    pub freq: Vec<u16>,
    /// Precomputed Barrett reciprocal `ceil(2^63 / freq[s])` for branchless
    /// exact division `state / freq[s]` via `(u128(state) * u128(rcp)) >> 63`.
    rcp_freq: Vec<u64>,
    /// `slots[start[s] + r]` is the ANS slot of the r-th occurrence of symbol s.
    slots: Vec<u16>,
    start: Vec<u32>,
    encoding: DistEncoding,
}

#[derive(Debug, Clone)]
enum DistEncoding {
    /// Only one symbol is used (or none).
    Single(usize),
    /// Two symbols with a directly represented 12-bit probability.
    Binary(usize, usize),
    /// General encoding with the given shift.
    General(u32),
}

impl AnsDist {
    /// Build a distribution from token counts. `counts.len()` must not exceed
    /// `1 << log_alpha`.
    pub fn new(counts: &[u32], log_alpha: u32) -> Self {
        let table_size = 1usize << log_alpha;
        assert!(counts.len() <= table_size);
        let used: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
        let (freq, encoding) = if used.len() <= 1 {
            let s = used.first().copied().unwrap_or(0);
            let mut f = vec![0u16; table_size];
            f[s] = TAB as u16;
            (f, DistEncoding::Single(s))
        } else {
            let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
            let mut best: Option<(f64, [u16; 256], usize, u32)> = None;
            let mut buf = [0u16; 256];
            for shift in 0..=LOG_TAB {
                let Some(len) = normalize(counts, shift, &mut buf) else {
                    continue;
                };
                let f = &buf[..len];
                let cost = header_bits(f, shift) as f64 + data_bits(counts, f, total);
                if best.as_ref().is_none_or(|b| cost < b.0) {
                    best = Some((cost, buf, len, shift));
                }
            }
            let (cost, best_buf, best_len, shift) = best.expect("no valid ANS normalisation");
            let mut f = vec![0u16; table_size];
            f[..best_len].copy_from_slice(&best_buf[..best_len]);
            let mut encoding = DistEncoding::General(shift);
            if let [a, b] = used[..] {
                let mut binary = vec![0u16; table_size];
                binary[a] =
                    ((u64::from(counts[a]) * u64::from(TAB) + total / 2) / total).clamp(1, u64::from(TAB - 1)) as u16;
                binary[b] = TAB as u16 - binary[a];
                let bits = 2 + u8_bits(a as u32) + u8_bits(b as u32) + LOG_TAB;
                if (bits as f64 + data_bits(counts, &binary, total)) < cost {
                    f = binary;
                    encoding = DistEncoding::Binary(a, b);
                }
            }
            (f, encoding)
        };
        let rcp_freq: Vec<u64> =
            freq.iter().map(|&f| if f == 0 { 0 } else { (1u64 << 63).div_ceil(u64::from(f)) }).collect();
        let (slots, start) = reverse_alias_table(&freq, log_alpha);
        AnsDist { freq, rcp_freq, slots, start, encoding }
    }

    /// Exact `(state / freq[s], state % freq[s])` using precomputed reciprocal
    /// multiplication (valid for all `state: u32` and `1 <= freq[s] <= 4096`).
    #[inline]
    pub fn div_mod(&self, s: usize, state: u32) -> (u32, u32) {
        let freq = u32::from(self.freq[s]);
        let q = ((u128::from(state) * u128::from(self.rcp_freq[s])) >> 63) as u32;
        let r = state - q * freq;
        debug_assert_eq!(q, state / freq);
        debug_assert_eq!(r, state % freq);
        (q, r)
    }

    /// ANS slot for the `r`-th (0-based, `r < freq[s]`) state of symbol `s`.
    #[inline]
    pub fn slot(&self, s: usize, r: u32) -> u32 {
        u32::from(self.slots[self.start[s] as usize + r as usize])
    }

    pub fn write(&self, w: &mut BitWriter) {
        match self.encoding {
            DistEncoding::Single(s) => {
                w.write_bool(true); // simple
                w.write_bool(false); // one symbol
                write_u8(w, s as u32);
            }
            DistEncoding::Binary(a, b) => {
                w.write_bool(true); // simple
                w.write_bool(true); // two symbols
                write_u8(w, a as u32);
                write_u8(w, b as u32);
                w.write(LOG_TAB, u64::from(self.freq[a]));
            }
            DistEncoding::General(shift) => write_general(w, &self.freq, shift),
        }
    }
}

/// Approximate cost in bits of coding `counts` with its own distribution,
/// including the distribution header. Used for clustering decisions.
pub fn estimate_cost(counts: &[u32]) -> f64 {
    estimate_cost_counts(counts.iter().copied())
}

/// Cost a prospective merge without allocating a temporary histogram.
pub fn estimate_merged_cost(a: &[u32], b: &[u32]) -> f64 {
    estimate_cost_counts(
        (0..a.len().max(b.len())).map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0)),
    )
}

/// Cost of `a` with the counts of `b` (a part of it) removed.
pub fn estimate_removed_cost(a: &[u32], b: &[u32]) -> f64 {
    estimate_cost_counts((0..a.len()).map(|i| a[i] - b.get(i).copied().unwrap_or(0)))
}

const XLOGX_N: usize = 4096;

/// Histogram clustering repeatedly evaluates small integer counts. Tabulating
/// c*log2(c) avoids a transcendental call per occupied bin, and the entropy is
/// then xlogx(total) - sum(xlogx(count)). Use f64 to keep tie decisions stable.
fn xlogx_table() -> &'static [f64; XLOGX_N] {
    static TABLE: std::sync::OnceLock<[f64; XLOGX_N]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| if i == 0 { 0.0 } else { i as f64 * (i as f64).log2() }))
}

#[inline]
fn xlogx_with(table: &[f64; XLOGX_N], count: u64) -> f64 {
    if count < XLOGX_N as u64 { table[count as usize] } else { count as f64 * (count as f64).log2() }
}

fn estimate_cost_counts(counts: impl ExactSizeIterator<Item = u32> + Clone) -> f64 {
    let alphabet = counts.len();
    let total: u64 = counts.clone().map(u64::from).sum();
    if total == 0 {
        return 0.0;
    }
    let table = xlogx_table();
    let mut bits = xlogx_with(table, total);
    let mut nonzero = 0;
    for c in counts {
        if c > 0 {
            nonzero += 1;
            bits -= xlogx_with(table, u64::from(c));
        }
    }
    if nonzero <= 1 {
        return 12.0;
    }
    // Roughly: shift + alphabet size + ~4 bits of log-count and ~4 bits of
    // mantissa per used symbol, 5 bits per unused one.
    let header = 12.0 + 8.0 * nonzero as f64 + 5.0 * (alphabet - nonzero) as f64;
    bits + header
}

fn write_u8(w: &mut BitWriter, v: u32) {
    if v == 0 {
        w.write_bool(false);
    } else {
        let n = 31 - v.leading_zeros();
        w.write_bool(true);
        w.write(3, u64::from(n));
        w.write(n, u64::from(v - (1 << n)));
    }
}

fn u8_bits(v: u32) -> u32 {
    if v == 0 { 1 } else { 4 + (31 - v.leading_zeros()) }
}

/// Number of mantissa bits stored for a frequency with `log = floor(log2 f)`.
fn bitcount(shift: u32, log: u32) -> u32 {
    let b = shift as i32 - ((LOG_TAB as i32 - log as i32) >> 1);
    b.clamp(0, log as i32) as u32
}

/// Round `v >= 1` to the nearest value representable with this shift.
fn representable(v: u32, shift: u32) -> u32 {
    if v <= 1 {
        return 1;
    }
    let log = 31 - v.leading_zeros();
    let drop = log - bitcount(shift, log);
    if drop == 0 {
        return v;
    }
    let step = 1 << drop;
    ((v + step / 2) >> drop) << drop
}

/// Log-count code of a frequency: 0 for 0, else floor(log2 f) + 1.
fn code_of(f: u16) -> u32 {
    if f == 0 { 0 } else { 32 - u32::from(f).leading_zeros() }
}

/// Normalise counts to 4096 for a given shift into `buf`. The most frequent
/// symbol (the first one with the largest log-count after rounding) absorbs
/// the rounding error; it must stay the strict first maximum or the decoder
/// would pick a different one, so on failure the others are scaled down and
/// retried. Returns the active length `alphabet.max(3)`.
fn normalize(counts: &[u32], shift: u32, buf: &mut [u16; 256]) -> Option<usize> {
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    let alphabet = counts.iter().rposition(|&c| c > 0)? + 1;
    let out_len = alphabet.max(3);
    let omit = (0..alphabet).max_by_key(|&i| (counts[i], std::cmp::Reverse(i)))?;
    let mut scale = 1.0f64;
    for _ in 0..64 {
        buf[..out_len].fill(0);
        let mut sum = 0u32;
        for i in 0..alphabet {
            if counts[i] == 0 || i == omit {
                continue;
            }
            let target = (f64::from(counts[i]) * f64::from(TAB) / total as f64 * scale).round().max(1.0) as u32;
            let v = representable(target, shift).min(TAB - 1);
            buf[i] = v as u16;
            sum += v;
        }
        if sum < TAB {
            let rest = TAB - sum;
            buf[omit] = rest as u16;
            let oc = code_of(buf[omit]);
            let ok = (0..alphabet).all(|i| {
                let c = code_of(buf[i]);
                i == omit || (i < omit && c < oc) || (i > omit && c <= oc)
            });
            if ok && oc <= 12 {
                return Some(out_len);
            }
        }
        scale *= 0.97;
    }
    None
}

fn bits_per_symbol_table() -> &'static [f64; TAB as usize + 1] {
    static TABLE: std::sync::OnceLock<[f64; TAB as usize + 1]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|f| if f == 0 { 0.0 } else { f64::from(LOG_TAB) - (f as f64).log2() }))
}

fn data_bits(counts: &[u32], f: &[u16], _total: u64) -> f64 {
    let lut = bits_per_symbol_table();
    counts.iter().zip(f).filter(|(c, _)| **c > 0).map(|(&c, &f)| f64::from(c) * lut[f as usize]).sum()
}

/// Prefix code for the log-counts 0..=12 and 13 (repeat previous symbol RLE):
/// (3-bit value, extra bits, number of extra bits).
fn logcount_code(code: u32) -> (u64, u64, u32) {
    match code {
        10 => (0, 0, 0),
        4 => (1, 0b1, 1),
        0 => (1, 0b10, 2),
        11 => (1, 0b100, 3),
        13 => (1, 0b1000, 4),
        12 => (1, 0b0000, 4),
        7 => (2, 0, 0),
        1 => (3, 1, 1),
        3 => (3, 0, 1),
        6 => (4, 0, 0),
        8 => (5, 0, 0),
        9 => (6, 0, 0),
        2 => (7, 1, 1),
        5 => (7, 0, 1),
        _ => unreachable!(),
    }
}

fn shift_bits(shift: u32) -> u32 {
    // len unary (up to 3) + len bits
    let len = if shift == 0 { 0 } else { 31 - (shift + 1).leading_zeros() };
    len + (len < 3) as u32 + len
}

/// Check if a zero-run RLE (code 13) can be emitted starting at `i`, returning
/// the run length (>= 4) if so. Code 13 repeats `prev` (which is 0 at `i == 0`
/// or whenever `f[i - 1] == 0`), and is disallowed at `omit + 1`. A run at
/// `i == 0` is well defined: libjxl and jxl-oxide both start `prev` at 0, and
/// the roundtrip tests exercise it.
#[inline]
fn zero_rle_run(f: &[u16], i: usize, alphabet: usize, omit: usize) -> Option<usize> {
    if f[i] != 0 || i == omit + 1 || (i > 0 && f[i - 1] != 0) {
        return None;
    }
    let mut run = 1;
    while i + run < alphabet && f[i + run] == 0 {
        run += 1;
    }
    (run >= 4).then_some(run)
}

fn header_bits(f: &[u16], shift: u32) -> usize {
    let alphabet = (f.iter().rposition(|&v| v > 0).unwrap_or(0) + 1).max(3);
    let omit = omit_pos(f, alphabet);
    let mut bits = 2 + shift_bits(shift) + u8_bits(alphabet as u32 - 3);
    let mut i = 0;
    while i < alphabet {
        if let Some(run) = zero_rle_run(f, i, alphabet, omit) {
            let (_, _, n) = logcount_code(13);
            bits += 3 + n + u8_bits((run - 4) as u32);
            i += run;
            continue;
        }
        let code = code_of(f[i]);
        let (_, _, n) = logcount_code(code);
        bits += 3 + n;
        if code > 1 && i != omit {
            bits += bitcount(shift, code - 1);
        }
        i += 1;
    }
    bits as usize
}

fn omit_pos(f: &[u16], alphabet: usize) -> usize {
    let mut best = 0;
    for i in 0..alphabet {
        if code_of(f[i]) > code_of(f[best]) {
            best = i;
        }
    }
    best
}

fn write_general(w: &mut BitWriter, f: &[u16], shift: u32) {
    let alphabet = (f.iter().rposition(|&v| v > 0).unwrap_or(0) + 1).max(3);
    w.write_bool(false); // not simple
    w.write_bool(false); // not flat
    let len = if shift == 0 { 0 } else { 31 - (shift + 1).leading_zeros() };
    for _ in 0..len {
        w.write_bool(true);
    }
    if len < 3 {
        w.write_bool(false);
    }
    w.write(len, u64::from(shift + 1 - (1 << len)));
    write_u8(w, alphabet as u32 - 3);
    let omit = omit_pos(f, alphabet);
    let mut i = 0;
    while i < alphabet {
        if let Some(run) = zero_rle_run(f, i, alphabet, omit) {
            let (v3, extra, n) = logcount_code(13);
            w.write(3, v3);
            w.write(n, extra);
            write_u8(w, (run - 4) as u32);
            i += run;
            continue;
        }
        let (v3, extra, n) = logcount_code(code_of(f[i]));
        w.write(3, v3);
        w.write(n, extra);
        i += 1;
    }
    for (i, &v) in f[..alphabet].iter().enumerate() {
        let code = code_of(v);
        if code <= 1 || i == omit {
            continue;
        }
        let log = code - 1;
        let bc = bitcount(shift, log);
        let mant = (u32::from(v) - (1 << log)) >> (log - bc);
        debug_assert_eq!((mant << (log - bc)) + (1 << log), u32::from(v));
        w.write(bc, u64::from(mant));
    }
}

#[cfg(test)]
#[path = "ans_tests.rs"]
mod tests;

/// Rebuild the decoder's alias table and invert it.
fn reverse_alias_table(freq: &[u16], log_alpha: u32) -> (Vec<u16>, Vec<u32>) {
    let table_size = freq.len();
    let log_bucket = LOG_TAB - log_alpha;
    let bucket_size = 1u32 << log_bucket;
    // (symbol, offset) for every slot
    let mut map = vec![(0u16, 0u16); TAB as usize];
    if let Some(single) = freq.iter().position(|&f| u32::from(f) == TAB) {
        for (idx, m) in map.iter_mut().enumerate() {
            *m = (single as u16, idx as u16);
        }
    } else {
        // cutoff, alias symbol, alias offset (working values)
        let mut cutoff: Vec<u32> = freq.iter().map(|&f| u32::from(f)).collect();
        let mut alias_sym: Vec<u32> = (0..table_size as u32).collect();
        let mut alias_off = vec![0u32; table_size];
        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (i, &c) in cutoff.iter().enumerate() {
            match c.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(i),
                std::cmp::Ordering::Greater => overfull.push(i),
                std::cmp::Ordering::Equal => {}
            }
        }
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - cutoff[u];
            cutoff[o] -= by;
            alias_sym[u] = o as u32;
            alias_off[u] = cutoff[o];
            match cutoff[o].cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Greater => overfull.push(o),
                std::cmp::Ordering::Equal => {}
            }
        }
        for i in 0..table_size {
            for pos in 0..bucket_size {
                let idx = (i as u32) * bucket_size + pos;
                let entry = if cutoff[i] == bucket_size || pos < cutoff[i] {
                    (i as u16, pos as u16)
                } else {
                    (alias_sym[i] as u16, (alias_off[i] + pos - cutoff[i]) as u16)
                };
                map[idx as usize] = entry;
            }
        }
    }
    let mut start = vec![0u32; table_size + 1];
    for i in 0..table_size {
        start[i + 1] = start[i] + u32::from(freq[i]);
    }
    let mut slots = vec![0u16; TAB as usize];
    for (idx, &(s, off)) in map.iter().enumerate() {
        debug_assert!(off < freq[s as usize]);
        slots[(start[s as usize] + u32::from(off)) as usize] = idx as u16;
    }
    start.pop();
    (slots, start)
}
