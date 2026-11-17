//! HF block context map (libjxl's FindBestBlockEntropyModel): the block
//! contexts of the coefficient tokens are chosen per transform order class
//! and, on large images, per range of the block's quantiser, with the
//! classes merged down to a size-dependent number of contexts.

use super::acs::BlockAcs;
use crate::bit_writer::BitWriter;
use crate::bit_writer::U32Dist::Bits;
use crate::entropy::write_context_map;

const NUM_ORDS: usize = 13;

/// libjxl's default map: Y, X, B rows over the 13 order classes.
const DEFAULT_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

pub struct BlockCtxMap {
    /// Quantiser thresholds: segment = number of thresholds below the
    /// block's multiplier (1..=256).
    qft: Vec<u32>,
    /// (channel row, order class, segment) -> block context.
    map: Vec<u8>,
    pub num: u32,
    default: bool,
}

impl BlockCtxMap {
    pub fn default_map() -> Self {
        BlockCtxMap { qft: Vec::new(), map: DEFAULT_MAP.to_vec(), num: 15, default: true }
    }

    /// Block context of channel row `ci` (0: Y, 1: X, 2: B).
    #[inline]
    pub fn ctx(&self, ci: usize, ord: usize, qf: u32) -> u32 {
        let seg = self.qft.iter().filter(|&&t| qf > t).count();
        u32::from(self.map[(ci * NUM_ORDS + ord) * (self.qft.len() + 1) + seg])
    }

    pub fn write(&self, w: &mut BitWriter) {
        w.write_bool(self.default);
        if self.default {
            return;
        }
        for _ in 0..3 {
            w.write(4, 0); // no LF thresholds
        }
        w.write(4, self.qft.len() as u64);
        for &t in &self.qft {
            w.write_u32(t - 1, [Bits(2, 0), Bits(3, 4), Bits(5, 12), Bits(8, 44)]);
        }
        write_context_map(&self.map, w);
    }

    /// libjxl's choice from the quantiser of every block (`raw`: multiplier
    /// - 1, before the per-block adjustment) and the transforms.
    pub fn choose(raw: &[u8], acs: &[BlockAcs], bw: usize, distance: f32) -> Self {
        let tot = raw.len();
        // No need to change the context model for small images.
        let size_for_ctx_model = 1024.0 * f64::from(distance);
        if (tot as f64) < size_for_ctx_model || std::env::var_os("MINI_NO_BLOCK_CTX").is_some() {
            return Self::default_map();
        }
        let mut qf_counts = [0usize; 256];
        let mut qf_ord_counts = vec![[0usize; 256]; NUM_ORDS];
        // Every 8x8 cell counts, with the quantiser of its varblock.
        for b in acs {
            let origin = b.oy as usize * bw + b.ox as usize;
            let qf = usize::from(raw[origin]);
            qf_counts[qf] += 1;
            qf_ord_counts[b.acs.order_id()][qf] += 1;
        }
        // Splitting by the quantiser only pays off on large images.
        let size_for_qf_split = 8192.0 * f64::from(distance);
        let num_qf_segments = if (tot as f64) < size_for_qf_split { 1 } else { 2 };
        let mut qft: Vec<u32> = Vec::new();
        let mut cumsum = 0;
        let mut next = 1;
        let mut cut = tot * next / num_qf_segments;
        for j in 0..256u32 {
            cumsum += qf_counts[j as usize];
            if cumsum > cut {
                if j != 0 {
                    qft.push(j);
                }
                while cumsum > cut {
                    next += 1;
                    cut = tot * next / num_qf_segments;
                }
            } else if next > num_qf_segments {
                break;
            }
        }
        let nqf = qft.len() + 1;
        let mut counts = vec![0usize; NUM_ORDS * nqf];
        let mut pos = 0;
        for j in 0..256 {
            if pos < qft.len() && j == qft[pos] as usize {
                pos += 1;
            }
            for (i, c) in qf_ord_counts.iter().enumerate() {
                counts[pos + i * nqf] += c[j];
            }
        }
        // Repeatedly merge the least frequent class into the next one. libjxl
        // keeps about twice as many contexts (tot / size / 2 and / 3); with
        // this encoder's histogram clustering fewer are smaller on photos.
        let mut remap: Vec<usize> = (0..NUM_ORDS * nqf).collect();
        let mut clusters = remap.clone();
        let r = tot as f64 / size_for_ctx_model;
        let nb_clusters = ((r / 4.0) as usize).clamp(2, 9);
        let nb_chroma = ((r / 6.0) as usize).clamp(1, 5);
        while clusters.len() > nb_clusters {
            // Stable, so that ties merge deterministically.
            clusters.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
            let last = clusters[clusters.len() - 1];
            let prev = clusters[clusters.len() - 2];
            counts[prev] += counts[last];
            counts[last] = 0;
            remap[last] = prev;
            clusters.pop();
        }
        for i in 0..remap.len() {
            while remap[remap[i]] != remap[i] {
                remap[i] = remap[remap[i]];
            }
        }
        let mut relabel = vec![usize::MAX; remap.len()];
        let mut num = 0;
        for r in remap.iter_mut() {
            if relabel[*r] == usize::MAX {
                relabel[*r] = num;
                num += 1;
            }
            *r = relabel[*r];
        }
        // X and B share the contexts of the largest luma clusters.
        let n = remap.len();
        let mut map: Vec<u8> = remap.iter().map(|&r| r as u8).collect();
        for i in n..3 * n {
            map.push((num + remap[i % n].min(nb_chroma - 1)) as u8);
        }
        let num = u32::from(*map.iter().max().unwrap()) + 1;
        BlockCtxMap { qft, map, num, default: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct::acs::Acs;

    #[test]
    fn maps_stay_within_format_limits() {
        for &(bw, bh, d) in
            &[(16usize, 16usize, 1.0f32), (96, 64, 2.2), (250, 167, 1.0), (500, 340, 0.5), (500, 340, 6.0)]
        {
            let acs: Vec<BlockAcs> = (0..bw * bh)
                .map(|i| {
                    let a = [Acs::Dct8, Acs::Dct4x4, Acs::Identity][i % 3];
                    BlockAcs { acs: a, ox: (i % bw) as u32, oy: (i / bw) as u32 }
                })
                .collect();
            let raw: Vec<u8> = (0..bw * bh).map(|i| (i * 7 % 61) as u8).collect();
            let m = BlockCtxMap::choose(&raw, &acs, bw, d);
            let nqf = m.qft.len() + 1;
            assert_eq!(m.map.len(), 3 * NUM_ORDS * nqf);
            assert!(m.num as usize <= 16 && nqf <= 64);
            assert_eq!(u32::from(*m.map.iter().max().unwrap()) + 1, m.num);
            assert!(m.qft.windows(2).all(|w| w[0] < w[1]) && m.qft.iter().all(|&t| (1..256).contains(&t)));
            for qf in 1..=256 {
                for ci in 0..3 {
                    assert!(m.ctx(ci, 0, qf) < m.num);
                }
            }
            let mut w = BitWriter::new();
            m.write(&mut w);
        }
    }
}
