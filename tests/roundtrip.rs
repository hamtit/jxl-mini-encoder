//! Encode synthetic images and decode them with jxl-oxide.

fn synth(w: u32, h: u32, ch: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut out = Vec::with_capacity((w * h) as usize * ch);
    for y in 0..h {
        for x in 0..w {
            let base = [
                (x * 255 / w.max(1)) as i32,
                (y * 255 / h.max(1)) as i32,
                ((x + y) * 3 % 256) as i32,
                200,
            ];
            for (c, &b) in base.iter().enumerate().take(ch) {
                let noise = (rnd() % 9) as i32 - 4;
                let v = if (x / 16 + y / 16) % 3 == 0 {
                    b + noise
                } else {
                    b
                };
                out.push(if c == 3 {
                    ((x ^ y) & 0xff) as u8
                } else {
                    v.clamp(0, 255) as u8
                });
            }
        }
    }
    out
}

fn decode(jxl: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(jxl))
        .expect("parse");
    let render = image.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    let data = fb
        .buf()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    (fb.width() as u32, fb.height() as u32, fb.channels(), data)
}

fn lossless(w: u32, h: u32, ch: usize, effort: u32) {
    let px = synth(w, h, ch, w * 31 + h);
    let img = if ch == 4 {
        jxl_mini_encoder::ImageRef::rgba8(w, h, &px)
    } else {
        jxl_mini_encoder::ImageRef::rgb8(w, h, &px)
    }
    .unwrap();
    let jxl = jxl_mini_encoder::Encoder::new()
        .lossless()
        .effort(effort)
        .threads(2)
        .encode(&img)
        .unwrap();
    let (dw, dh, dc, data) = decode(&jxl);
    assert_eq!((dw, dh, dc), (w, h, ch));
    assert!(data == px, "{w}x{h}x{ch} e{effort}: pixels differ");
}

#[test]
fn lossless_small() {
    for (w, h) in [(1, 1), (7, 3), (64, 64), (100, 37), (256, 256)] {
        for ch in [3, 4] {
            for e in [1, 3, 7] {
                lossless(w, h, ch, e);
            }
        }
    }
}

#[test]
fn lossless_groups() {
    for (w, h) in [(257, 20), (300, 600), (700, 513)] {
        for ch in [3, 4] {
            for e in [1, 3, 7] {
                lossless(w, h, ch, e);
            }
        }
    }
}

#[test]
fn constant_planes_and_binary_images() {
    for (w, h) in [(1u32, 1u32), (17, 9), (257, 259)] {
        for alpha in [0u8, 255] {
            let pixels: Vec<u8> = (0..w * h).flat_map(|_| [0, 127, 255, alpha]).collect();
            let image = jxl_mini_encoder::ImageRef::rgba8(w, h, &pixels).unwrap();
            for quality in [90.0, 100.0] {
                let out = jxl_mini_encoder::Encoder::new()
                    .quality(quality)
                    .encode(&image)
                    .unwrap();
                let (dw, dh, ch, decoded) = decode(&out);
                assert_eq!((dw, dh, ch), (w, h, 4));
                assert!(decoded.chunks_exact(4).all(|p| p[3] == alpha));
                if quality == 100.0 {
                    assert_eq!(decoded, pixels);
                }
            }
        }
    }
    let (w, h) = (257u32, 259u32);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = if (i % w + i / w) % 2 == 0 { 0 } else { 255 };
            [v, v, v]
        })
        .collect();
    let image = jxl_mini_encoder::ImageRef::rgb8(w, h, &pixels).unwrap();
    for effort in [1, 3, 7, 9] {
        let enc = jxl_mini_encoder::Encoder::new().lossless().effort(effort);
        let out = enc.clone().threads(1).encode(&image).unwrap();
        assert_eq!(out, enc.threads(4).encode(&image).unwrap());
        assert_eq!(decode(&out).3, pixels);
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn lossy(w: u32, h: u32, ch: usize, quality: f32, effort: u32) -> f64 {
    let px = synth(w, h, ch, w * 7 + h);
    let img = if ch == 4 {
        jxl_mini_encoder::ImageRef::rgba8(w, h, &px)
    } else {
        jxl_mini_encoder::ImageRef::rgb8(w, h, &px)
    }
    .unwrap();
    let jxl = jxl_mini_encoder::Encoder::new()
        .quality(quality)
        .effort(effort)
        .threads(3)
        .encode(&img)
        .unwrap();
    let (dw, dh, dc, data) = decode(&jxl);
    assert_eq!((dw, dh, dc), (w, h, ch));
    if ch == 4 {
        // Alpha is always lossless.
        let a: Vec<u8> = px.iter().skip(3).step_by(4).copied().collect();
        let b: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
        assert!(a == b, "alpha differs");
    }
    psnr(&px, &data)
}

#[test]
fn lossy_sizes() {
    for (w, h) in [(1, 1), (5, 3), (8, 8), (64, 48), (257, 300), (530, 270)] {
        for ch in [3, 4] {
            for e in [3, 7] {
                let p = lossy(w, h, ch, 90.0, e);
                assert!(p > 20.0, "{w}x{h}x{ch} e{e}: psnr {p}");
            }
        }
    }
}

#[test]
fn lossy_quality_range() {
    let mut prev = 0.0;
    for q in [0.0, 30.0, 60.0, 90.0, 99.0] {
        let p = lossy(200, 120, 3, q, 7);
        assert!(p >= prev - 0.5, "quality {q}: psnr {p} < {prev}");
        prev = p;
    }
    assert!(prev > 35.0);
}

#[test]
fn rejects_bad_input() {
    assert!(jxl_mini_encoder::ImageRef::rgb8(0, 4, &[]).is_err());
    assert!(jxl_mini_encoder::ImageRef::rgb8(2, 2, &[0; 11]).is_err());
}

#[test]
fn lossy_very_wide() {
    // Block coordinates beyond u16 range.
    let (w, h) = (524_296u32, 1u32);
    let px = vec![128u8; (w * h) as usize * 3];
    let img = jxl_mini_encoder::ImageRef::rgb8(w, h, &px).unwrap();
    let jxl = jxl_mini_encoder::Encoder::new()
        .quality(90.0)
        .effort(1)
        .threads(4)
        .encode(&img)
        .unwrap();
    let (dw, dh, _, data) = decode(&jxl);
    assert_eq!((dw, dh), (w, h));
    assert!(data.iter().all(|&v| v.abs_diff(128) <= 2));
}

/// Threading must not change the codestream.
#[test]
fn threads_do_not_change_output() {
    for (w, h, ch, quality, effort) in [
        (700u32, 530u32, 3usize, 100.0f32, 7u32),
        (700, 530, 4, 100.0, 4),
        (700, 530, 3, 85.0, 7),
        (700, 530, 4, 85.0, 5),
    ] {
        let px = synth(w, h, ch, 9);
        let img = if ch == 4 {
            jxl_mini_encoder::ImageRef::rgba8(w, h, &px)
        } else {
            jxl_mini_encoder::ImageRef::rgb8(w, h, &px)
        }
        .unwrap();
        let enc = |t| {
            jxl_mini_encoder::Encoder::new()
                .quality(quality)
                .effort(effort)
                .threads(t)
                .encode(&img)
                .unwrap()
        };
        assert_eq!(enc(1), enc(3), "{w}x{h} ch={ch} q={quality} e={effort}");
    }
}

/// Flat colour blocks with hard edges: libjxl does not use Gaborish at
/// distance <= 0.5, because its sharpening overshoots on such edges and
/// inflates the output several times over.
#[test]
fn hard_edges_stay_small_at_high_fidelity() {
    let (w, h) = (256u32, 256u32);
    let mut s = 12345u32;
    let levels: Vec<[u8; 3]> = (0..(w / 16) * (h / 16))
        .map(|_| {
            [0; 3].map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 24) % 6 * 50) as u8
            })
        })
        .collect();
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&levels[((y / 16) * (w / 16) + x / 16) as usize]);
        }
    }
    let img = jxl_mini_encoder::ImageRef::rgb8(w, h, &px).unwrap();
    let size = |d: f32| {
        jxl_mini_encoder::Encoder::new()
            .distance(d)
            .encode(&img)
            .unwrap()
            .len()
    };
    // About 1.4 kB with Gaborish off; 22 kB with it on.
    let high = size(0.3);
    assert!(high < 5000, "d0.3 produced {high} B");
}

/// A page of repeated glyph-like shapes on a flat background, with optional
/// alpha. Patches code each shape once; the result must still decode to the
/// original within lossy tolerance.
fn glyph_page(w: u32, h: u32, ch: usize) -> Vec<u8> {
    let mut s = 99u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 16
    };
    // Eight 9x11 bitmaps with about half the pixels set.
    let glyphs: Vec<Vec<bool>> = (0..8)
        .map(|_| (0..99).map(|_| rnd() % 2 == 0).collect())
        .collect();
    let mut px = vec![0u8; (w * h) as usize * ch];
    for p in px.chunks_mut(ch) {
        p[..3].copy_from_slice(&[245, 245, 240]);
        if ch == 4 {
            p[3] = 255;
        }
    }
    let mut y = 8;
    while y + 11 < h - 8 {
        let mut x = 8;
        while x + 9 < w - 8 {
            let g = &glyphs[(rnd() % 8) as usize];
            for gy in 0..11 {
                for gx in 0..9 {
                    if g[gy * 9 + gx] {
                        let i = ((y as usize + gy) * w as usize + x as usize + gx) * ch;
                        px[i..i + 3].copy_from_slice(&[20, 30, 90]);
                    }
                }
            }
            x += 12;
        }
        y += 16;
    }
    px
}

fn patches_roundtrip(w: u32, h: u32, ch: usize, distance: f32, max_bytes: usize) {
    let px = glyph_page(w, h, ch);
    let img = if ch == 4 {
        jxl_mini_encoder::ImageRef::rgba8(w, h, &px)
    } else {
        jxl_mini_encoder::ImageRef::rgb8(w, h, &px)
    }
    .unwrap();
    let jxl = jxl_mini_encoder::Encoder::new()
        .distance(distance)
        .effort(7)
        .encode(&img)
        .unwrap();
    let (dw, dh, dch, out) = decode(&jxl);
    assert_eq!((dw, dh, dch), (w, h, ch));
    let p = psnr(&px, &out);
    assert!(p > 28.0, "psnr {p}");
    assert!(
        jxl.len() <= max_bytes,
        "{} B for {w}x{h}x{ch} d{distance}",
        jxl.len()
    );
}

#[test]
fn patches_repeated_shapes() {
    // About 11.5 kB with patches, 28 kB without.
    patches_roundtrip(300, 200, 3, 1.5, 15_000);
    patches_roundtrip(300, 200, 4, 1.5, 16_000);
    // Odd size, high quality.
    patches_roundtrip(131, 77, 3, 0.8, 7_000);
}

#[test]
fn patches_reference_frame_in_several_groups() {
    // More distinct shapes than fit in one group of the reference frame.
    let (w, h) = (700u32, 520u32);
    let mut px = vec![245u8; (w * h) as usize * 3];
    let mut s = 7u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 16
    };
    let shapes: Vec<Vec<bool>> = (0..420)
        .map(|_| (0..100).map(|_| rnd() % 2 == 0).collect())
        .collect();
    for (k, g) in shapes.iter().enumerate() {
        for rep in 0..2 {
            let x = 6 + (k % 34) * 20 + rep * 7 % 3;
            let y = 6 + ((k / 34) * 2 + rep) * 20;
            if y + 12 >= h as usize || x + 12 >= w as usize {
                continue;
            }
            for gy in 0..10 {
                for gx in 0..10 {
                    if g[gy * 10 + gx] {
                        let i = ((y + gy) * w as usize + x + gx) * 3;
                        px[i..i + 3].copy_from_slice(&[10, 10, 10]);
                    }
                }
            }
        }
    }
    let img = jxl_mini_encoder::ImageRef::rgb8(w, h, &px).unwrap();
    let jxl = jxl_mini_encoder::Encoder::new()
        .distance(1.0)
        .effort(7)
        .encode(&img)
        .unwrap();
    let (dw, dh, _, out) = decode(&jxl);
    assert_eq!((dw, dh), (w, h));
    assert!(psnr(&px, &out) > 25.0);
    // 65 kB with patches, 135 kB without.
    assert!(jxl.len() < 90_000, "{} B", jxl.len());
}
