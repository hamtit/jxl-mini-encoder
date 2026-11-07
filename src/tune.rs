//! Experimental multiplicative scales for hand-tuned constants, read once
//! from `MINI_TUNE` (comma-separated). Used by the offline parameter search.

use std::sync::OnceLock;

pub const N: usize = 13;

fn table() -> &'static [f32; N] {
    static T: OnceLock<[f32; N]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f32; N];
        if let Ok(s) = std::env::var("MINI_TUNE") {
            for (i, v) in s.split(',').enumerate().take(N) {
                if let Ok(x) = v.trim().parse() {
                    t[i] = x;
                }
            }
        }
        t
    })
}

#[inline]
pub fn k(i: usize) -> f32 {
    table()[i]
}
