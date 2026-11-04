use super::*;

fn reference_cost(counts: &[u32]) -> f64 {
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return 0.0;
    }
    let mut nonzero = 0;
    let mut bits = 0.0;
    for &c in counts {
        if c != 0 {
            nonzero += 1;
            bits += f64::from(c) * (total as f64 / f64::from(c)).log2();
        }
    }
    if nonzero <= 1 {
        return 12.0;
    }
    bits + 12.0 + 8.0 * nonzero as f64 + 5.0 * (counts.len() - nonzero) as f64
}

#[test]
fn cached_cost_matches_reference_and_materialized_merges() {
    let values = [0, 1, 2, 7, 4095, 4096, 65536, u32::MAX / 4];
    for n in 0..=256 {
        let a: Vec<_> = (0..n).map(|i| values[(i * 7 + n) % values.len()]).collect();
        let b: Vec<_> = (0..n / 2).map(|i| values[(i * 3 + n) % values.len()]).collect();
        for counts in [&a, &b] {
            let expected = reference_cost(counts);
            assert!((estimate_cost(counts) - expected).abs() <= 1e-9 * expected.max(1.0));
        }
        let mut merged = a.clone();
        for (dst, &v) in merged.iter_mut().zip(&b) {
            *dst += v;
        }
        assert_eq!(estimate_merged_cost(&a, &b), estimate_cost(&merged));
        assert_eq!(estimate_merged_cost(&b, &a), estimate_cost(&merged));
    }
    assert_eq!(estimate_cost(&[0, 0]), 0.0);
    assert_eq!(estimate_cost(&[0, 7000]), 12.0);
}

#[test]
fn sparse_binary_histograms_use_simple_header() {
    for log_alpha in 5..=8 {
        let last = (1usize << log_alpha) - 1;
        for (a, b) in [(1, 1), (1, 4095), (4095, 1), (1, 1_000_000)] {
            let mut counts = vec![0; last + 1];
            counts[0] = a;
            counts[last] = b;
            let dist = AnsDist::new(&counts, log_alpha);
            assert!(matches!(dist.encoding, DistEncoding::Binary(0, s) if s == last));
            assert_eq!(dist.freq.iter().map(|&v| u32::from(v)).sum::<u32>(), TAB);
            assert!(dist.freq[0] > 0 && dist.freq[last] > 0);
            let mut w = BitWriter::new();
            dist.write(&mut w);
            assert_eq!(w.bits_written(), (2 + u8_bits(0) + u8_bits(last as u32) + LOG_TAB) as usize);
            assert_eq!(w.finish()[0] & 3, 3);
        }
    }
}
