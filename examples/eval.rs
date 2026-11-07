//! Encode images, decode them back with jxl-oxide and report size and
//! fidelity.
//!
//!   cargo run --release --example eval -- [-q QUALITY | -l] [-e EFFORT] [-t THREADS] [--encode-only] FILE...
//!
//! `--encode-only` skips decoding and fidelity metrics, allowing external peak-RSS
//! measurements to isolate the encoder (plus input loading). It prints an output
//! hash to check bit-exactness between implementations.

use std::time::Instant;

fn load(path: &str) -> (u32, u32, bool, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap();
    if path.ends_with(".png") {
        let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut r = dec.read_info().unwrap();
        let mut buf = vec![0; r.output_buffer_size().unwrap()];
        let info = r.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        let (alpha, data) = match info.color_type {
            png::ColorType::Rgb => (false, buf),
            png::ColorType::Rgba => (true, buf),
            png::ColorType::Grayscale => (false, buf.iter().flat_map(|&g| [g, g, g]).collect()),
            png::ColorType::GrayscaleAlpha => (true, buf.chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect()),
            c => panic!("unsupported {c:?}"),
        };
        (info.width, info.height, alpha, data)
    } else {
        let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
        let data = dec.decode().unwrap();
        let info = dec.info().unwrap();
        let data = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => data,
            jpeg_decoder::PixelFormat::L8 => data.iter().flat_map(|&g| [g, g, g]).collect(),
            f => panic!("unsupported {f:?}"),
        };
        (u32::from(info.width), u32::from(info.height), false, data)
    }
}

fn decode(jxl: &[u8], channels: usize) -> Vec<u8> {
    let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(jxl)).expect("parse");
    let render = image.render_frame(0).expect("render");
    let fb = render.image_all_channels();
    assert_eq!(fb.channels(), channels, "channel count");
    fb.buf().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect()
}

fn ssim2(w: u32, h: u32, a: &[u8], b: &[u8], ch: usize) -> f64 {
    use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic};
    let conv = |d: &[u8]| {
        let px: Vec<[f32; 3]> = d.chunks(ch).map(|p| [p[0], p[1], p[2]].map(|v| f32::from(v) / 255.0)).collect();
        Rgb::new(px, w as usize, h as usize, TransferCharacteristic::SRGB, ColorPrimaries::BT709).unwrap()
    };
    ssimulacra2::compute_frame_ssimulacra2(conv(a), conv(b)).unwrap()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut enc = jxl_mini_encoder::Encoder::new();
    let mut lossless = false;
    let mut encode_only = false;
    let mut files = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--encode-only" => encode_only = true,
            "-q" => enc = enc.quality(args.next().unwrap().parse().unwrap()),
            "-d" => enc = enc.distance(args.next().unwrap().parse().unwrap()),
            "-l" => {
                enc = enc.lossless();
                lossless = true;
            }
            "-e" => enc = enc.effort(args.next().unwrap().parse().unwrap()),
            "-t" => enc = enc.threads(args.next().unwrap().parse().unwrap()),
            _ => files.push(a),
        }
    }
    let (mut tot_bytes, mut tot_px, mut tot_ms, mut tot_s) = (0usize, 0usize, 0.0f64, 0.0f64);
    for f in &files {
        let (w, h, alpha, data) = load(f);
        let (w, h, data) = match std::env::var("CROP") {
            Ok(c) => {
                let (cw, ch) = c.split_once('x').unwrap();
                let (cw, ch): (u32, u32) = (cw.parse().unwrap(), ch.parse().unwrap());
                let n = if alpha { 4 } else { 3 };
                let mut d = Vec::new();
                for y in 0..ch.min(h) {
                    let row = (y * w) as usize * n;
                    d.extend_from_slice(&data[row..row + cw.min(w) as usize * n]);
                }
                (cw.min(w), ch.min(h), d)
            }
            Err(_) => (w, h, data),
        };
        let img = if alpha {
            jxl_mini_encoder::ImageRef::rgba8(w, h, &data)
        } else {
            jxl_mini_encoder::ImageRef::rgb8(w, h, &data)
        }
        .unwrap();
        let t = Instant::now();
        let jxl = match std::env::var("REF_DIR") {
            // Evaluate a file produced by another encoder instead.
            Ok(dir) => std::fs::read(format!("{dir}/ref_{}.jxl", f.rsplit('/').next().unwrap())).unwrap(),
            Err(_) => enc.encode(&img).unwrap(),
        };
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let ch = if alpha { 4 } else { 3 };
        if let Ok(dir) = std::env::var("DUMP") {
            let name = f.rsplit('/').next().unwrap();
            std::fs::write(format!("{dir}/{name}.jxl"), &jxl).unwrap();
        }
        if encode_only {
            let hash = jxl.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
            println!("{f}: {} B {ms:.3} ms hash={hash:016x}", jxl.len());
            continue;
        }
        let out = decode(&jxl, ch);
        if let Ok(dir) = std::env::var("DUMP") {
            let name = f.rsplit('/').next().unwrap();
            let file = std::fs::File::create(format!("{dir}/{name}.out.png")).unwrap();
            let mut e = png::Encoder::new(std::io::BufWriter::new(file), w, h);
            e.set_color(if alpha { png::ColorType::Rgba } else { png::ColorType::Rgb });
            e.set_depth(png::BitDepth::Eight);
            e.write_header().unwrap().write_image_data(&out).unwrap();
            std::fs::write(format!("{dir}/{name}.jxl"), &jxl).unwrap();
        }
        let bpp = jxl.len() as f64 * 8.0 / f64::from(w * h);
        let name = f.rsplit('/').next().unwrap();
        if lossless {
            let ok = out == data;
            println!(
                "{name:24} {w}x{h} {:>9} B {bpp:6.3} bpp {ms:8.1} ms {}",
                jxl.len(),
                if ok { "exact" } else { "MISMATCH" }
            );
            assert!(ok, "lossless mismatch");
        } else {
            let s = ssim2(w, h, &data, &out, ch);
            tot_s += s;
            println!(
                "{name:24} {w}x{h} {:>9} B {bpp:6.3} bpp {ms:8.1} ms  psnr {:6.2}  ssim2 {s:6.2}",
                jxl.len(),
                psnr(&data, &out)
            );
        }
        tot_bytes += jxl.len();
        tot_px += (w * h) as usize;
        tot_ms += ms;
    }
    if encode_only {
        return;
    }
    println!(
        "total {tot_bytes} B, {:.3} bpp, {tot_ms:.0} ms, mean ssim2 {:.2}",
        tot_bytes as f64 * 8.0 / tot_px as f64,
        tot_s / files.len() as f64
    );
}
