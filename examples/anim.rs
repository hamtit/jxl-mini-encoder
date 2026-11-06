//! Encode a sequence of PNG frames (e.g. a decoded GIF) as an animated JPEG XL,
//! decode it back with jxl-oxide and report size, time and SSIMULACRA2.
//!
//!   cargo run --release --example anim -- [-q QUALITY | -d DIST | -l] [-e EFFORT] [-t THREADS]
//!       [--delay CS] [-o OUT.jxl] [--no-decode] FRAME.png...
//!
//! `--delay` is the display time of every frame in 1/100 s (default 10, minimum 1).

use std::time::Instant;

fn load(path: &str) -> (u32, u32, bool, Vec<u8>) {
    let mut dec = png::Decoder::new(std::io::Cursor::new(std::fs::read(path).unwrap()));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().unwrap();
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    match info.color_type {
        png::ColorType::Rgb => (info.width, info.height, false, buf),
        png::ColorType::Rgba => (info.width, info.height, true, buf),
        c => panic!("unsupported {c:?}"),
    }
}

fn ssim2(w: u32, h: u32, a: &[u8], b: &[u8], ch: usize) -> f64 {
    use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic};
    let conv = |d: &[u8]| {
        let px: Vec<[f32; 3]> = d
            .chunks(ch)
            .map(|p| [p[0], p[1], p[2]].map(|v| f32::from(v) / 255.0))
            .collect();
        let (w, h) = (w as usize, h as usize);
        Rgb::new(
            px,
            w,
            h,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap()
    };
    ssimulacra2::compute_frame_ssimulacra2(conv(a), conv(b)).unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut enc = jxl_mini_encoder::Encoder::new();
    let (mut files, mut out, mut delay, mut decode) = (Vec::new(), None, 10u32, true);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-q" => enc = enc.quality(args.next().unwrap().parse().unwrap()),
            "-d" => enc = enc.distance(args.next().unwrap().parse().unwrap()),
            "-l" => enc = enc.lossless(),
            "-e" => enc = enc.effort(args.next().unwrap().parse().unwrap()),
            "-t" => enc = enc.threads(args.next().unwrap().parse().unwrap()),
            "--delay" => delay = args.next().unwrap().parse().unwrap(),
            "-o" => out = args.next(),
            "--no-decode" => decode = false,
            _ => files.push(a),
        }
    }
    // A zero-duration frame is a layer shown with the next frame, so decoders
    // report fewer displayed frames than inputs and the scores would not line up.
    assert!(delay >= 1, "--delay must be at least 1");
    let loaded: Vec<_> = files.iter().map(|f| load(f)).collect();
    let (w, h, alpha, _) = &loaded[0];
    let frames: Vec<_> = loaded
        .iter()
        .map(|(w, h, alpha, d)| jxl_mini_encoder::AnimationFrame {
            image: if *alpha {
                jxl_mini_encoder::ImageRef::rgba8(*w, *h, d)
            } else {
                jxl_mini_encoder::ImageRef::rgb8(*w, *h, d)
            }
            .unwrap(),
            duration: delay,
        })
        .collect();
    let t = Instant::now();
    let jxl = enc
        .encode_animation(&frames, &jxl_mini_encoder::AnimationOptions::default())
        .unwrap();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    if let Some(o) = out {
        std::fs::write(o, &jxl).unwrap();
    }
    print!(
        "{} frames {w}x{h}: {} B, {ms:.0} ms",
        frames.len(),
        jxl.len()
    );
    if !decode {
        println!();
        return;
    }
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&jxl))
        .expect("parse");
    let ch = if *alpha { 4 } else { 3 };
    let mut tot = 0.0;
    for (i, (_, _, _, src)) in loaded.iter().enumerate() {
        let fb = image.render_frame(i).expect("render").image_all_channels();
        let px: Vec<u8> = fb
            .buf()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect();
        tot += ssim2(*w, *h, src, &px, ch);
    }
    println!(", mean ssim2 {:.2}", tot / loaded.len() as f64);
}
