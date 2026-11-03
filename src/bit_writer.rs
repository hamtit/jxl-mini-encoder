//! LSB-first bit writer, the bit order JPEG XL uses everywhere.

/// One of the four alternatives of a `U32` field.
#[derive(Clone, Copy, Debug)]
pub enum U32Dist {
    /// A constant value (no extra bits).
    Val(u32),
    /// `offset + u(bits)`.
    Bits(u32, u32),
}

use U32Dist::{Bits, Val};

#[derive(Default, Debug, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            ..Self::default()
        }
    }

    /// Write the low `n` bits of `v` (n <= 32).
    #[inline]
    pub fn write(&mut self, n: u32, v: u64) {
        debug_assert!(n <= 32);
        debug_assert!(
            n == 32 || v < (1u64 << n),
            "value {v} does not fit in {n} bits"
        );
        self.acc |= v << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    #[inline]
    pub fn write_bool(&mut self, b: bool) {
        self.write(1, b as u64);
    }

    pub fn bits_written(&self) -> usize {
        self.buf.len() * 8 + self.nbits as usize
    }

    pub fn zero_pad_to_byte(&mut self) {
        if self.nbits > 0 {
            self.write(8 - self.nbits, 0);
        }
    }

    /// Append another writer's bits.
    pub fn append(&mut self, other: &BitWriter) {
        if self.nbits == 0 {
            self.buf.extend_from_slice(&other.buf);
        } else {
            for &b in &other.buf {
                self.write(8, b as u64);
            }
        }
        if other.nbits > 0 {
            self.write(other.nbits, other.acc);
        }
    }

    /// Append whole bytes; the writer must be byte aligned.
    pub fn append_bytes(&mut self, bytes: &[u8]) {
        assert_eq!(self.nbits, 0, "append_bytes on an unaligned writer");
        self.buf.extend_from_slice(bytes);
    }

    /// Byte-aligned contents (the final partial byte is zero-padded).
    pub fn finish(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        self.buf
    }

    /// Write `v` with the cheapest of the four distributions that can hold it.
    pub fn write_u32(&mut self, v: u32, d: [U32Dist; 4]) {
        let mut best: Option<(u32, usize)> = None;
        for (i, dist) in d.iter().enumerate() {
            let cost = match *dist {
                Val(c) if c == v => 0,
                Bits(n, off) if v >= off && u64::from(v - off) < (1u64 << n) => n,
                _ => continue,
            };
            if best.is_none_or(|(c, _)| cost < c) {
                best = Some((cost, i));
            }
        }
        let (_, sel) = best.unwrap_or_else(|| panic!("{v} is not representable in {d:?}"));
        self.write(2, sel as u64);
        if let Bits(n, off) = d[sel] {
            self.write(n, u64::from(v - off));
        }
    }

    pub fn write_u64(&mut self, v: u64) {
        match v {
            0 => self.write(2, 0),
            1..=16 => {
                self.write(2, 1);
                self.write(4, v - 1);
            }
            17..=272 => {
                self.write(2, 2);
                self.write(8, v - 17);
            }
            _ => {
                self.write(2, 3);
                self.write(12, v & 0xfff);
                let mut v = v >> 12;
                let mut shift = 12;
                while v > 0 {
                    self.write_bool(true);
                    if shift == 60 {
                        self.write(4, v);
                        return;
                    }
                    self.write(8, v & 0xff);
                    v >>= 8;
                    shift += 8;
                }
                self.write_bool(false);
            }
        }
    }

    /// IEEE half float.
    pub fn write_f16(&mut self, v: f32) {
        self.write(16, u64::from(f32_to_f16(v)));
    }
}

/// `v` as the decoder reads it back from `write_f16`.
pub fn round_to_f16(v: f32) -> f32 {
    let h = f32_to_f16(v);
    if h & 0x7fff == 0 {
        return 0.0;
    }
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f) - 15;
    let mant = 1.0 + f32::from(h & 0x3ff) / 1024.0;
    sign * mant * 2f32.powi(exp)
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if v == 0.0 {
        return sign;
    }
    assert!((1..31).contains(&exp), "f16 value out of range: {v}");
    // Round to nearest.
    let m = (mant + 0x1000) >> 13;
    if m == 0x400 {
        return sign | (((exp + 1) as u16) << 10);
    }
    sign | ((exp as u16) << 10) | m as u16
}

/// `ceil(log2(x + 1))`: the number of bits needed to store values up to `x`.
pub fn bits_for(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// `ceil(log2(x))` for x >= 1.
pub fn ceil_log2(x: u64) -> u32 {
    x.next_power_of_two().trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_lsb_first() {
        let mut w = BitWriter::new();
        w.write(1, 1);
        w.write(3, 0b101);
        w.write(12, 0xabc);
        assert_eq!(w.finish(), vec![0b1100_1011, 0xab]);
    }

    #[test]
    fn f16() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
    }
}
