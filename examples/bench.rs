use std::time::Instant;
fn synth(w: usize, h: usize, ch: usize) -> Vec<u8> {
    let mut s = 12345u32;
    let mut v = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / 40.0, y as f32 / 55.0);
            for c in 0..ch {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((s >> 24) as f32 - 128.0) * 0.08;
                let base = 128.0
                    + 70.0 * (fx + c as f32).sin() * (fy * 0.7).cos()
                    + 30.0 * ((x ^ y) as f32 / 90.0).sin();
                let a = if c == 3 {
                    200.0 + 40.0 * fx.sin()
                } else {
                    base + n
                };
                v.push(a.clamp(0.0, 255.0) as u8);
            }
        }
    }
    v
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let q: f32 = a[3].parse().unwrap();
    let e: u32 = a[4].parse().unwrap();
    let t: usize = a[5].parse().unwrap();
    let alpha = a.get(6).is_some();
    let ch = if alpha { 4 } else { 3 };
    let px = synth(w, h, ch);
    let img = if alpha {
        jxl_mini_encoder::ImageRef::rgba8(w as u32, h as u32, &px)
    } else {
        jxl_mini_encoder::ImageRef::rgb8(w as u32, h as u32, &px)
    }
    .unwrap();
    let t0 = Instant::now();
    let out = jxl_mini_encoder::Encoder::new()
        .quality(q)
        .effort(e)
        .threads(t)
        .encode(&img)
        .unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let mut hsh = 0xcbf29ce484222325u64;
    for &b in &out {
        hsh = (hsh ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    println!(
        "{}x{} q{} e{} t{} a{} -> {} B hash={:016x} {:.2}s",
        w,
        h,
        q,
        e,
        t,
        alpha as u8,
        out.len(),
        hsh,
        dt
    );
}
