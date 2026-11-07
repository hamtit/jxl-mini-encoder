//! Bench helper: raw RGBA frame dump (u32 w,h,n; then per frame u32 delay + RGBA)
//! encoded like imgconv-rs does. `gifbench FILE.raw QUALITY EFFORT THREADS [OUT.jxl]`
use jxl_mini_encoder::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let b = std::fs::read(&a[0]).unwrap();
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (w, h, n) = (u(0), u(4), u(8) as usize);
    let fsz = (w * h * 4) as usize;
    let mut o = 12;
    let mut data = Vec::new();
    let mut dl = Vec::new();
    for _ in 0..n {
        dl.push(u(o));
        o += 4;
        let rgb: Vec<u8> = b[o..o + fsz].as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        data.push(rgb);
        o += fsz;
    }
    let q: f32 = a[1].parse().unwrap();
    let frames: Vec<_> = data
        .iter()
        .zip(&dl)
        .map(|(d, &c)| AnimationFrame {
            image: ImageRef::rgb8(w, h, d).unwrap(),
            duration: if c <= 1 { 10 } else { c },
        })
        .collect();
    let mut enc = Encoder::new().quality(q).effort(a[2].parse().unwrap()).threads(a[3].parse().unwrap());
    let ll = std::env::var_os("FORCE_LL").is_some() || q >= 70.0 && animation_prefers_lossless(&frames);
    if ll {
        enc = enc.lossless();
    }
    let t = Instant::now();
    let out = enc.encode_animation(&frames, &AnimationOptions::default()).unwrap();
    let ms = t.elapsed().as_secs_f64();
    let mut hash = 0u64;
    for &x in &out {
        hash = hash.wrapping_mul(1099511628211).wrapping_add(x as u64 + 1);
    }
    println!("lossless={ll} {} B {ms:.2} s hash {hash:016x}", out.len());
    if let Some(p) = a.get(4) {
        std::fs::write(p, &out).unwrap();
    }
}
