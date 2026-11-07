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
            let base = [(x * 255 / w.max(1)) as i32, (y * 255 / h.max(1)) as i32, ((x + y) * 3 % 256) as i32, 200];
            for (c, &b) in base.iter().enumerate().take(ch) {
                let noise = (rnd() % 9) as i32 - 4;
                let v = if (x / 16 + y / 16) % 3 == 0 { b + noise } else { b };
                out.push(if c == 3 { ((x ^ y) & 0xff) as u8 } else { v.clamp(0, 255) as u8 });
            }
        }
    }
    out
}

fn decode(jxl: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(jxl)).expect("parse");
    let render = image.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    let data = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
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
    let jxl = jxl_mini_encoder::Encoder::new().lossless().effort(effort).threads(2).encode(&img).unwrap();
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
                let out = jxl_mini_encoder::Encoder::new().quality(quality).encode(&image).unwrap();
                let (dw, dh, ch, decoded) = decode(&out);
                assert_eq!((dw, dh, ch), (w, h, 4));
                assert!(decoded.as_chunks::<4>().0.iter().all(|p| p[3] == alpha));
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
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
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
    let jxl = jxl_mini_encoder::Encoder::new().quality(quality).effort(effort).threads(3).encode(&img).unwrap();
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
    let jxl = jxl_mini_encoder::Encoder::new().quality(90.0).effort(1).threads(4).encode(&img).unwrap();
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
        (700, 530, 3, 40.0, 6),
        (700, 530, 4, 85.0, 5),
    ] {
        let px = synth(w, h, ch, 9);
        let img = if ch == 4 {
            jxl_mini_encoder::ImageRef::rgba8(w, h, &px)
        } else {
            jxl_mini_encoder::ImageRef::rgb8(w, h, &px)
        }
        .unwrap();
        let enc = |t| jxl_mini_encoder::Encoder::new().quality(quality).effort(effort).threads(t).encode(&img).unwrap();
        assert_eq!(enc(1), enc(3), "{w}x{h} ch={ch} q={quality} e={effort}");
    }
}

/// Flat colour blocks should avoid sharpening overshoot at both high and
/// moderate fidelity. The graphics detector must also be thread-independent.
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
    for distance in [0.3, 0.55, 1.0, 2.35] {
        let enc = jxl_mini_encoder::Encoder::new().distance(distance);
        let bytes = enc.clone().threads(1).encode(&img).unwrap();
        assert_eq!(bytes, enc.threads(4).encode(&img).unwrap());
        assert!(bytes.len() < 5000, "d{distance} produced {} B", bytes.len());
        let (dw, dh, ch, decoded) = decode(&bytes);
        assert_eq!((dw, dh, ch), (w, h, 3));
        assert!(psnr(&px, &decoded) > 35.0, "d{distance}: fidelity regression");
    }
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
    let glyphs: Vec<Vec<bool>> = (0..8).map(|_| (0..99).map(|_| rnd() % 2 == 0).collect()).collect();
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
    let jxl = jxl_mini_encoder::Encoder::new().distance(distance).effort(7).encode(&img).unwrap();
    let (dw, dh, dch, out) = decode(&jxl);
    assert_eq!((dw, dh, dch), (w, h, ch));
    let p = psnr(&px, &out);
    assert!(p > 28.0, "psnr {p}");
    assert!(jxl.len() <= max_bytes, "{} B for {w}x{h}x{ch} d{distance}", jxl.len());
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
    let shapes: Vec<Vec<bool>> = (0..420).map(|_| (0..100).map(|_| rnd() % 2 == 0).collect()).collect();
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
    let jxl = jxl_mini_encoder::Encoder::new().distance(1.0).effort(7).encode(&img).unwrap();
    let (dw, dh, _, out) = decode(&jxl);
    assert_eq!((dw, dh), (w, h));
    assert!(psnr(&px, &out) > 25.0);
    // 65 kB with patches, 135 kB without.
    assert!(jxl.len() < 90_000, "{} B", jxl.len());
}

/// RGB input may come back with an opaque alpha channel (blended frames of a
/// lossless animation need one); drop it after checking that it is opaque.
fn strip_opaque_alpha(data: Vec<u8>, dch: usize, ch: usize) -> Vec<u8> {
    if dch == ch {
        return data;
    }
    assert_eq!((dch, ch), (4, 3), "unexpected channel count");
    assert!(data.as_chunks::<4>().0.iter().all(|p| p[3] == 255), "alpha not opaque");
    data.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
}

fn animation(w: u32, h: u32, ch: usize, n: usize, distance: f32, effort: u32) {
    let px: Vec<Vec<u8>> = (0..n).map(|i| synth(w, h, ch, 7 + i as u32)).collect();
    let imgs: Vec<_> = px
        .iter()
        .map(|p| {
            if ch == 4 { jxl_mini_encoder::ImageRef::rgba8(w, h, p) } else { jxl_mini_encoder::ImageRef::rgb8(w, h, p) }
                .unwrap()
        })
        .collect();
    let durations = [10u32, 1, 300, 5, 7];
    let frames: Vec<_> = imgs
        .iter()
        .enumerate()
        .map(|(i, &image)| jxl_mini_encoder::AnimationFrame { image, duration: durations[i % durations.len()] })
        .collect();
    let opts = jxl_mini_encoder::EncoderOptions { distance, effort, threads: 2, noise: 0.0 };
    let anim = jxl_mini_encoder::AnimationOptions { num_loops: 3, ..Default::default() };
    let jxl = jxl_mini_encoder::encode_animation(&frames, &anim, &opts).unwrap();
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
    let header = image.image_header();
    let a = header.metadata.animation.as_ref().expect("animation");
    assert_eq!((a.tps_numerator, a.tps_denominator, a.num_loops), (100, 1, 3));
    for (i, p) in px.iter().enumerate() {
        let render = image.render_frame(i).expect("render");
        let fb = render.image_all_channels();
        let data: Vec<u8> = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        assert_eq!((fb.width() as u32, fb.height() as u32), (w, h));
        let data = strip_opaque_alpha(data, fb.channels(), ch);
        if distance == 0.0 {
            assert!(&data == p, "{w}x{h}x{ch} frame {i}: pixels differ");
        } else {
            let err: u64 = data.iter().zip(p).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
            assert!(err / (data.len() as u64) < 12, "frame {i}: mean error too large");
        }
    }
    assert!(image.render_frame(n).is_err(), "unexpected extra frame");
}

#[test]
fn animation_lossless() {
    for (w, h) in [(8, 8), (40, 30), (300, 270)] {
        for ch in [3, 4] {
            animation(w, h, ch, 1, 0.0, 3);
            animation(w, h, ch, 4, 0.0, 3);
        }
    }
}

#[test]
fn animation_lossy() {
    for (w, h) in [(64, 48), (300, 270)] {
        for ch in [3, 4] {
            for e in [3, 7] {
                animation(w, h, ch, 3, 1.0, e);
            }
        }
    }
}

#[test]
fn animation_errors() {
    let a = vec![0u8; 8 * 8 * 3];
    let b = vec![0u8; 8 * 16 * 3];
    let opts = jxl_mini_encoder::EncoderOptions::default();
    let anim = jxl_mini_encoder::AnimationOptions::default();
    assert_eq!(jxl_mini_encoder::encode_animation(&[], &anim, &opts), Err(jxl_mini_encoder::Error::NoFrames));
    let f = |image| jxl_mini_encoder::AnimationFrame { image, duration: 1 };
    let frames = [
        f(jxl_mini_encoder::ImageRef::rgb8(8, 8, &a).unwrap()),
        f(jxl_mini_encoder::ImageRef::rgb8(8, 16, &b).unwrap()),
    ];
    assert_eq!(jxl_mini_encoder::encode_animation(&frames, &anim, &opts), Err(jxl_mini_encoder::Error::FrameMismatch));
}

#[test]
fn animation_zero_duration_frame_is_not_a_keyframe() {
    // A zero-duration frame is a layer that is shown together with the next
    // frame, so decoders do not report it as a separate displayed frame.
    let px: Vec<Vec<u8>> = (0..3).map(|i| synth(32, 32, 3, i)).collect();
    let frames: Vec<_> = px
        .iter()
        .enumerate()
        .map(|(i, p)| jxl_mini_encoder::AnimationFrame {
            image: jxl_mini_encoder::ImageRef::rgb8(32, 32, p).unwrap(),
            duration: if i == 1 { 0 } else { 5 },
        })
        .collect();
    let opts = jxl_mini_encoder::EncoderOptions { distance: 0.0, effort: 3, threads: 1, noise: 0.0 };
    let jxl = jxl_mini_encoder::encode_animation(&frames, &Default::default(), &opts).unwrap();
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
    assert!(image.render_frame(1).is_ok());
    assert!(image.render_frame(2).is_err());
}

#[test]
fn animation_rejects_unrepresentable_tick_rate() {
    let a = vec![0u8; 3];
    let frames =
        [jxl_mini_encoder::AnimationFrame { image: jxl_mini_encoder::ImageRef::rgb8(1, 1, &a).unwrap(), duration: 1 }];
    let opts = jxl_mini_encoder::EncoderOptions::default();
    for (num, den) in [(0, 1), (1, 0), (100, 2000), (u32::MAX, 1), ((1 << 30) + 1, 1)] {
        let anim = jxl_mini_encoder::AnimationOptions { tps_numerator: num, tps_denominator: den, num_loops: 0 };
        assert_eq!(
            jxl_mini_encoder::encode_animation(&frames, &anim, &opts),
            Err(jxl_mini_encoder::Error::InvalidAnimation),
            "{num}/{den}"
        );
    }
    let max = jxl_mini_encoder::AnimationOptions { tps_numerator: 1 << 30, tps_denominator: 1024, num_loops: u32::MAX };
    assert!(jxl_mini_encoder::encode_animation(&frames, &max, &opts).is_ok());
}

/// Frames that share a static background and move a small sprite: only the
/// changed part is coded (inter-frame prediction), unchanged frames extend the
/// previous one, and every displayed frame still decodes to the right pixels.
fn interframe(w: u32, h: u32, ch: usize, distance: f32) {
    let bg = synth(w, h, ch, 99);
    let mut px: Vec<Vec<u8>> = Vec::new();
    // (sprite x, sprite y, duration); the repeated position gives a frame
    // without any change.
    let steps = [(3u32, 5u32), (3, 5), (20, 9), (21, 40), (21, 40), (60, 33)];
    for &(sx, sy) in &steps {
        let mut f = bg.clone();
        for y in sy..(sy + 12).min(h) {
            for x in sx..(sx + 12).min(w) {
                let o = (y as usize * w as usize + x as usize) * ch;
                f[o..o + 3].copy_from_slice(&[250, 20, 40]);
            }
        }
        px.push(f);
    }
    let frames: Vec<_> = px
        .iter()
        .enumerate()
        .map(|(i, p)| jxl_mini_encoder::AnimationFrame {
            image: if ch == 4 {
                jxl_mini_encoder::ImageRef::rgba8(w, h, p)
            } else {
                jxl_mini_encoder::ImageRef::rgb8(w, h, p)
            }
            .unwrap(),
            duration: 3 + i as u32,
        })
        .collect();
    let opts = jxl_mini_encoder::EncoderOptions { distance, effort: 5, threads: 2, noise: 0.0 };
    let anim = Default::default();
    let jxl = jxl_mini_encoder::encode_animation(&frames, &anim, &opts).unwrap();
    let whole = std::env::var_os("MINI_NO_INTERFRAME").is_some();
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
    // Identical neighbours (steps 0/1 and 3/4) are merged.
    let shown = if whole { px.len() } else { px.len() - 2 };
    assert_eq!(image.num_loaded_keyframes(), shown);
    let mut t = 0u32;
    let mut starts = Vec::new();
    for k in 0..shown {
        starts.push(t);
        t += image.render_frame(k).expect("render").duration();
    }
    assert_eq!(t, frames.iter().map(|f| f.duration).sum::<u32>());
    let mut at = 0u32;
    for (i, p) in px.iter().enumerate() {
        let k = starts.partition_point(|&s| s <= at) - 1;
        at += frames[i].duration;
        let fb = image.render_frame(k).expect("render").image_all_channels();
        let data: Vec<u8> = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        let data = strip_opaque_alpha(data, fb.channels(), ch);
        if distance == 0.0 {
            assert!(&data == p, "{w}x{h}x{ch} frame {i}: pixels differ");
        } else {
            let err: u64 = data.iter().zip(p).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
            assert!(err / (data.len() as u64) < 12, "frame {i}: error too large");
        }
    }
}

#[test]
fn animation_interframe() {
    for ch in [3, 4] {
        interframe(96, 72, ch, 0.0);
        interframe(300, 140, ch, 0.0);
        interframe(96, 72, ch, 1.0);
        interframe(300, 140, ch, 1.0);
    }
}

/// Unchanged frames are merged into the previous one only while the summed
/// display time fits the header; the total time must never shrink.
#[test]
fn animation_merge_keeps_total_duration() {
    for (w, h) in [(40u32, 24u32), (8, 8)] {
        let px = synth(w, h, 3, 5);
        let img = jxl_mini_encoder::ImageRef::rgb8(w, h, &px).unwrap();
        let frame = jxl_mini_encoder::AnimationFrame { image: img, duration: 3_000_000_000 };
        let anim = jxl_mini_encoder::AnimationOptions { tps_numerator: 1_000_000_000, ..Default::default() };
        let opts = jxl_mini_encoder::EncoderOptions { distance: 0.0, effort: 3, threads: 1, noise: 0.0 };
        let jxl = jxl_mini_encoder::encode_animation(&[frame; 3], &anim, &opts).unwrap();
        let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
        let total: u64 = (0..image.num_loaded_keyframes())
            .map(|k| u64::from(image.render_frame(k).expect("render").duration()))
            .sum();
        assert_eq!(total, 9_000_000_000, "{w}x{h}");
        let fb = image.render_frame(1).unwrap().image_all_channels();
        let data: Vec<u8> = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        assert!(data == px, "{w}x{h}: pixels differ");
    }
}

/// A small alpha-only change is still coded in lossy animations (alpha is
/// lossless), even though RGB changes of that size are reused.
#[test]
fn animation_lossy_alpha_change_is_kept() {
    let (w, h) = (40u32, 24u32);
    let a = synth(w, h, 4, 3);
    let mut b = a.clone();
    for p in b.chunks_mut(4) {
        p[3] = p[3].wrapping_add(1);
    }
    let frames: Vec<_> = [&a, &b]
        .iter()
        .map(|p| jxl_mini_encoder::AnimationFrame {
            image: jxl_mini_encoder::ImageRef::rgba8(w, h, p).unwrap(),
            duration: 5,
        })
        .collect();
    let opts = jxl_mini_encoder::EncoderOptions { distance: 1.0, effort: 3, threads: 1, noise: 0.0 };
    let jxl = jxl_mini_encoder::encode_animation(&frames, &Default::default(), &opts).unwrap();
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
    assert_eq!(image.num_loaded_keyframes(), 2);
    let alpha = |k: usize| -> Vec<u8> {
        let fb = image.render_frame(k).unwrap().image_all_channels();
        fb.buf().chunks(4).map(|p| (p[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect()
    };
    let want = |p: &Vec<u8>| p.chunks(4).map(|c| c[3]).collect::<Vec<u8>>();
    assert!(alpha(0) == want(&a) && alpha(1) == want(&b));
}

/// A palette image (few colours, flat areas) round-trips exactly, with and
/// without alpha, in one group and in several, including colours at the
/// extremes (white is the largest packed colour).
#[test]
fn lossless_palette() {
    for (w, h) in [(5, 3), (100, 80), (257, 40), (300, 300)] {
        for ch in [3, 4] {
            let colors: [[u8; 4]; 5] =
                [[0, 0, 0, 255], [255, 255, 255, 255], [10, 200, 30, 255], [255, 0, 0, 128], [7, 7, 7, 0]];
            let mut px = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let k = if (x / 7 + y / 5) % 4 == 0 { (x + y) % 5 } else { 1 };
                    px.extend_from_slice(&colors[k as usize][..ch]);
                }
            }
            let img = if ch == 4 {
                jxl_mini_encoder::ImageRef::rgba8(w as u32, h as u32, &px)
            } else {
                jxl_mini_encoder::ImageRef::rgb8(w as u32, h as u32, &px)
            }
            .unwrap();
            for effort in [1, 4, 7, 9] {
                let jxl = jxl_mini_encoder::Encoder::new().lossless().effort(effort).threads(2).encode(&img).unwrap();
                let (_, _, dc, data) = decode(&jxl);
                assert_eq!(dc, ch);
                assert!(data == px, "{w}x{h}x{ch} e{effort}: palette pixels differ");
            }
        }
    }
}

/// More than 256 colours falls back to the colour transform.
#[test]
fn lossless_many_colors_not_palette() {
    let (w, h) = (64u32, 64u32);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]);
        }
    }
    let img = jxl_mini_encoder::ImageRef::rgb8(w, h, &px).unwrap();
    let jxl = jxl_mini_encoder::Encoder::new().lossless().encode(&img).unwrap();
    assert!(decode(&jxl).3 == px);
}

/// Sparse changes of a palette animation are blended; scattered pixels all
/// over the frame (a large rectangle) must still decode exactly.
#[test]
fn animation_lossless_sparse_changes() {
    for ch in [3, 4] {
        let (w, h) = (300u32, 200u32);
        let palette = [[20u8, 30, 40, 255], [250, 250, 250, 255], [200, 10, 10, 255], [10, 200, 10, 255]];
        let mut frames_px: Vec<Vec<u8>> = Vec::new();
        for f in 0..6u32 {
            let mut px = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let sparkle = (x * 7 + y * 13 + f * 31) % 97 == 0;
                    let k = if sparkle { 1 + (f as usize + x as usize) % 3 } else { 0 };
                    px.extend_from_slice(&palette[k][..ch]);
                }
            }
            frames_px.push(px);
        }
        let frames: Vec<_> = frames_px
            .iter()
            .map(|p| jxl_mini_encoder::AnimationFrame {
                image: if ch == 4 {
                    jxl_mini_encoder::ImageRef::rgba8(w, h, p)
                } else {
                    jxl_mini_encoder::ImageRef::rgb8(w, h, p)
                }
                .unwrap(),
                duration: 5,
            })
            .collect();
        let opts = jxl_mini_encoder::EncoderOptions { distance: 0.0, effort: 5, threads: 3, noise: 0.0 };
        let jxl = jxl_mini_encoder::encode_animation(&frames, &Default::default(), &opts).unwrap();
        let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
        assert_eq!(image.num_loaded_keyframes(), frames_px.len());
        for (i, p) in frames_px.iter().enumerate() {
            let fb = image.render_frame(i).expect("render").image_all_channels();
            let data: Vec<u8> = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
            let data = strip_opaque_alpha(data, fb.channels(), ch);
            assert!(&data == p, "ch{ch} frame {i}: pixels differ");
        }
    }
}

#[test]
fn lossless_hint_for_palette_animation_with_sparse_changes() {
    let (w, h) = (64u32, 48u32);
    let make = |n: u32, noisy: bool| -> Vec<Vec<u8>> {
        (0..n)
            .map(|f| {
                let mut px = vec![30u8; (w * h * 3) as usize];
                for i in 0..(w * h) as usize {
                    // Few changing pixels, or a different texture in every frame.
                    let on = if noisy { (i as u32 * 7 + f).is_multiple_of(3) } else { i as u32 == f * 5 };
                    if on {
                        px[i * 3..i * 3 + 3].copy_from_slice(&[200, 40, 40]);
                    }
                }
                px
            })
            .collect()
    };
    for (noisy, expect) in [(false, true), (true, false)] {
        let px = make(6, noisy);
        let frames: Vec<_> = px
            .iter()
            .map(|p| jxl_mini_encoder::AnimationFrame {
                image: jxl_mini_encoder::ImageRef::rgb8(w, h, p).unwrap(),
                duration: 10,
            })
            .collect();
        assert_eq!(jxl_mini_encoder::animation_prefers_lossless(&frames), expect);
    }
    // Photos with many colours are not palette animations.
    let photo = synth(w, h, 3, 1);
    let frames = [photo.clone(), photo].map(|p| p);
    let frames: Vec<_> = frames
        .iter()
        .map(|p| jxl_mini_encoder::AnimationFrame {
            image: jxl_mini_encoder::ImageRef::rgb8(w, h, p).unwrap(),
            duration: 10,
        })
        .collect();
    assert!(!jxl_mini_encoder::animation_prefers_lossless(&frames));
    assert!(!jxl_mini_encoder::animation_prefers_lossless(&[]));
}

/// Blending must not lose the hidden colour of unchanged transparent pixels.
#[test]
fn animation_lossless_keeps_hidden_rgb_of_transparent_pixels() {
    let (w, h) = (16u32, 8u32);
    let mut f0 = Vec::new();
    for _ in 0..w * h {
        f0.extend_from_slice(&[10u8, 20, 30, 0]);
    }
    let mut f1 = f0.clone();
    f1[0..4].copy_from_slice(&[200, 100, 50, 255]);
    let frames: Vec<_> = [&f0, &f1]
        .iter()
        .map(|p| jxl_mini_encoder::AnimationFrame {
            image: jxl_mini_encoder::ImageRef::rgba8(w, h, p).unwrap(),
            duration: 5,
        })
        .collect();
    let opts = jxl_mini_encoder::EncoderOptions { distance: 0.0, effort: 5, threads: 2, noise: 0.0 };
    let jxl = jxl_mini_encoder::encode_animation(&frames, &Default::default(), &opts).unwrap();
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&jxl)).expect("parse");
    // The decoder may premultiply or drop hidden colours of fully transparent
    // pixels, so compare against what a replace-coded frame decodes to.
    let direct = |p: &Vec<u8>| {
        let img = jxl_mini_encoder::ImageRef::rgba8(w, h, p).unwrap();
        decode(&jxl_mini_encoder::Encoder::new().lossless().encode(&img).unwrap()).3
    };
    for (i, p) in [&f0, &f1].iter().enumerate() {
        let fb = image.render_frame(i).expect("render").image_all_channels();
        let data: Vec<u8> = fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        assert!(data == direct(p), "frame {i} differs from replace coding");
    }
}
