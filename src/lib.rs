//! A small JPEG XL encoder in pure Rust.
//!
//! Based on libjxl's encoding algorithms, heuristics and constants, with
//! independent changes to supported features and parameter handling.
//! It supports 8-bit sRGB RGB or RGBA input, lossy coding (VarDCT in XYB)
//! and lossless coding (Modular), with an effort setting and optional threads.
//! The output is a bare JPEG XL codestream. Behavior is not guaranteed to
//! match libjxl.
//!
//! ```no_run
//! let pixels = vec![0u8; 64 * 64 * 3];
//! let image = jxl_mini_encoder::ImageRef::rgb8(64, 64, &pixels).unwrap();
//! let jxl = jxl_mini_encoder::Encoder::new().quality(75.0).effort(7).encode(&image).unwrap();
//! assert_eq!(&jxl[..2], &[0xff, 0x0a]);
//! ```

// Tuning constants are copied from libjxl with their original precision.
#![allow(clippy::excessive_precision)]
// Index loops mirror the specification and libjxl.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod bit_writer;
mod entropy;
mod error;
mod headers;
mod lossless;
mod modular;
mod parallel;
mod patches;
mod vardct;

pub use error::Error;

use bit_writer::BitWriter;
use headers::{AnimationInfo, FramePlacement, ImageInfo, write_image_header};

/// Borrowed 8-bit sRGB pixels, interleaved, without row padding.
#[derive(Clone, Copy, Debug)]
pub struct ImageRef<'a> {
    width: u32,
    height: u32,
    alpha: bool,
    data: &'a [u8],
}

impl<'a> ImageRef<'a> {
    /// RGB pixels (3 bytes per pixel).
    pub fn rgb8(width: u32, height: u32, data: &'a [u8]) -> Result<Self, Error> {
        Self::new(width, height, false, data)
    }

    /// RGBA pixels (4 bytes per pixel, straight alpha).
    pub fn rgba8(width: u32, height: u32, data: &'a [u8]) -> Result<Self, Error> {
        Self::new(width, height, true, data)
    }

    fn new(width: u32, height: u32, alpha: bool, data: &'a [u8]) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(if alpha { 4 } else { 3 }))
            .ok_or(Error::InvalidDimensions { width, height })?;
        if data.len() != expected {
            return Err(Error::BufferSize {
                expected,
                actual: data.len(),
            });
        }
        Ok(ImageRef {
            width,
            height,
            alpha,
            data,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn has_alpha(&self) -> bool {
        self.alpha
    }

    fn channels(&self) -> usize {
        if self.alpha { 4 } else { 3 }
    }
}

/// Encoder settings for [`encode`].
///
/// Effort and thread count are normalized when encoding. Distance is not
/// clamped; supply a finite value in 0..=25.
#[derive(Clone, Debug)]
pub struct EncoderOptions {
    /// Butteraugli distance; 0 means lossless.
    pub distance: f32,
    /// 1 (fastest) to 10 (slowest).
    pub effort: u32,
    /// Worker threads (1 = encode on the calling thread).
    pub threads: usize,
    /// Strength of decoder-side noise synthesis for lossy coding: 0 turns it
    /// off, 1 restores the noise that coding is estimated to remove. It makes
    /// noisy sources (video, dithered GIFs) look less smooth, but the added
    /// noise is not the source's noise, so SSIMULACRA2 drops.
    pub noise: f32,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            distance: 1.0,
            effort: 7,
            threads: 1,
            noise: 0.0,
        }
    }
}

/// Builder-style front end.
#[derive(Clone, Debug, Default)]
pub struct Encoder {
    opts: EncoderOptions,
}

impl Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Quality converted to an encoding distance.
    ///
    /// Supply a finite value in 0..=100. Values >= 100 select lossless coding;
    /// negative values are not clamped to zero.
    pub fn quality(mut self, quality: f32) -> Self {
        self.opts.distance = distance_from_quality(quality);
        self
    }

    /// Butteraugli distance (0 = lossless, 1 = visually lossless, up to 25).
    ///
    /// Supply a finite value. Values outside 0..=25 are clamped, unlike
    /// libjxl's distance-setting API, which rejects out-of-range values.
    pub fn distance(mut self, distance: f32) -> Self {
        self.opts.distance = distance.clamp(0.0, 25.0);
        self
    }

    pub fn lossless(self) -> Self {
        self.distance(0.0)
    }

    /// Encoding effort, clamped to 1..=10 rather than rejected as in libjxl.
    pub fn effort(mut self, effort: u32) -> Self {
        self.opts.effort = effort.clamp(1, 10);
        self
    }

    /// Worker thread count; zero is normalized to one.
    pub fn threads(mut self, threads: usize) -> Self {
        self.opts.threads = threads.max(1);
        self
    }

    /// Noise synthesis strength (see [`EncoderOptions::noise`]); 0 = off.
    pub fn noise(mut self, strength: f32) -> Self {
        self.opts.noise = strength.clamp(0.0, 4.0);
        self
    }

    pub fn options(&self) -> &EncoderOptions {
        &self.opts
    }

    /// Encode an animation to a bare JPEG XL codestream.
    pub fn encode_animation(
        &self,
        frames: &[AnimationFrame],
        anim: &AnimationOptions,
    ) -> Result<Vec<u8>, Error> {
        encode_animation(frames, anim, &self.opts)
    }

    /// Encode to a bare JPEG XL codestream.
    pub fn encode(&self, image: &ImageRef) -> Result<Vec<u8>, Error> {
        encode(image, &self.opts)
    }
}

/// Convert quality to an encoding distance.
///
/// Supply a finite value in 0..=100. Values >= 100 map to zero (lossless);
/// negative values are not clamped to zero.
pub fn distance_from_quality(quality: f32) -> f32 {
    if quality >= 100.0 {
        0.0
    } else if quality >= 30.0 {
        0.1 + (100.0 - quality) * 0.09
    } else {
        53.0 / 3000.0 * quality * quality - 23.0 / 20.0 * quality + 25.0
    }
}

/// Encode to a bare JPEG XL codestream.
pub fn encode(image: &ImageRef, opts: &EncoderOptions) -> Result<Vec<u8>, Error> {
    let lossless = opts.distance <= 0.0;
    let mut opts = opts.clone();
    opts.effort = opts.effort.clamp(1, 10);
    opts.threads = opts.threads.max(1);
    let mut w = BitWriter::with_capacity(image.data.len() / 8);
    let info = ImageInfo {
        width: image.width,
        height: image.height,
        has_alpha: image.alpha,
        xyb: !lossless,
        animation: None,
    };
    write_image_header(&mut w, &info);
    encode_frame(image, &opts, FramePlacement::STILL, None, &mut w);
    Ok(w.finish())
}

fn encode_frame(
    image: &ImageRef,
    opts: &EncoderOptions,
    placement: FramePlacement,
    noise: Option<u32>,
    w: &mut BitWriter,
) {
    if opts.distance <= 0.0 {
        lossless::encode(image, opts, placement, w);
    } else {
        vardct::encode(image, opts, placement, noise, w);
    }
}

/// One frame of an animation: full-canvas pixels and a display time.
#[derive(Clone, Copy, Debug)]
pub struct AnimationFrame<'a> {
    /// Fully composited canvas for this frame. Every frame must have the same
    /// size and channel layout.
    pub image: ImageRef<'a>,
    /// Display time in ticks (see [`AnimationOptions`]).
    pub duration: u32,
}

/// Timing and looping of an animation.
#[derive(Clone, Copy, Debug)]
pub struct AnimationOptions {
    /// Ticks per second is `tps_numerator / tps_denominator`. The numerator
    /// must be in 1..=2^30 and the denominator in 1..=1024.
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    /// Number of loops; 0 repeats forever.
    pub num_loops: u32,
}

impl Default for AnimationOptions {
    /// 100 ticks per second, so a GIF delay (in centiseconds) is a duration
    /// as is. Loops forever.
    fn default() -> Self {
        AnimationOptions {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
        }
    }
}

/// Encode full-canvas frames as an animated bare JPEG XL codestream.
///
/// Each frame is coded independently (replace blending, no cropping or
/// inter-frame prediction), so frames that were decoded from a GIF must
/// already be composited onto the canvas. Lossy and lossless coding follow
/// `opts.distance` for all frames.
pub fn encode_animation(
    frames: &[AnimationFrame],
    anim: &AnimationOptions,
    opts: &EncoderOptions,
) -> Result<Vec<u8>, Error> {
    let first = frames.first().ok_or(Error::NoFrames)?;
    // The header can store a numerator up to 2^30 and a denominator up to 1024.
    if !(1..=1 << 30).contains(&anim.tps_numerator) || !(1..=1024).contains(&anim.tps_denominator) {
        return Err(Error::InvalidAnimation);
    }
    for f in frames {
        let (a, b) = (&f.image, &first.image);
        if (a.width, a.height, a.alpha) != (b.width, b.height, b.alpha) {
            return Err(Error::FrameMismatch);
        }
    }
    let lossless = opts.distance <= 0.0;
    let mut opts = opts.clone();
    opts.effort = opts.effort.clamp(1, 10);
    opts.threads = opts.threads.max(1);
    let plan = plan_frames(frames, if lossless { 0 } else { INTERFRAME_TOLERANCE });
    // Sparse lossless frames are blended: unchanged pixels are transparent,
    // which needs an alpha channel even if the input has none.
    let blending = plan.iter().any(|p| p.blend);
    let mut w = BitWriter::with_capacity(first.image.data.len() * frames.len() / 8);
    let info = ImageInfo {
        width: first.image.width,
        height: first.image.height,
        has_alpha: first.image.alpha || blending,
        xyb: !lossless,
        animation: Some(AnimationInfo {
            tps_numerator: anim.tps_numerator,
            tps_denominator: anim.tps_denominator,
            num_loops: anim.num_loops,
        }),
    };
    write_image_header(&mut w, &info);
    // Decide the noise-dependent tools once for the whole animation so that
    // frames near the threshold do not flip between settings (visible flicker).
    let noise = (!lossless).then(|| {
        let mut levels: Vec<u32> = frames
            .iter()
            .map(|f| vardct::noise_level(&f.image))
            .collect();
        levels.sort_unstable();
        levels[levels.len() / 2]
    });
    let last = plan.len() - 1;
    // Frames are independent once planned (each starts and ends on a byte
    // boundary), so they are coded in parallel into separate buffers. Threads
    // not needed across frames are left to the frame-internal parallelism.
    let outer = opts.threads.min(plan.len());
    let mut inner_opts = opts.clone();
    inner_opts.threads = (opts.threads / outer).max(1);
    let parts = parallel::par_map(plan.len(), outer, |k| {
        let step = &plan[k];
        let src = &frames[step.index].image;
        let placement = FramePlacement {
            duration: Some(step.duration),
            is_last: k == last,
            save: k != last,
            crop: step.rect.map(|(x, y, _, _)| (x, y)),
            blend: step.blend,
        };
        let mut fw = BitWriter::new();
        let sch = src.channels();
        // In a blended animation every frame carries alpha.
        let och = if blending { 4 } else { sch };
        let (x, y, rw, rh) = step.rect.unwrap_or((0, 0, src.width, src.height));
        let whole = step.rect.is_none() && och == sch;
        let mut buf = Vec::new();
        if !whole {
            buf.reserve(rw as usize * rh as usize * och);
            // The canvas before this frame is the previous coded frame.
            let prev = (step.blend && k > 0).then(|| frames[plan[k - 1].index].image.data);
            for row in y..y + rh {
                let o = (row as usize * src.width as usize + x as usize) * sch;
                let line = &src.data[o..o + rw as usize * sch];
                for (i, px) in line.chunks_exact(sch).enumerate() {
                    if och == 4 && sch == 3 {
                        buf.extend_from_slice(px);
                        buf.push(255);
                    } else {
                        buf.extend_from_slice(px);
                    }
                    if let Some(prev) = prev {
                        if prev[o + i * sch..o + (i + 1) * sch] == *px {
                            let n = buf.len();
                            buf[n - och..].fill(0); // transparent: keep the canvas
                        }
                    }
                }
            }
        }
        let part = if whole {
            *src
        } else {
            ImageRef {
                width: rw,
                height: rh,
                alpha: och == 4,
                data: &buf,
            }
        };
        encode_frame(&part, &inner_opts, placement, noise, &mut fw);
        fw.finish()
    });
    for part in parts {
        w.append_bytes(&part);
    }
    Ok(w.finish())
}

/// Version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Largest per-channel difference (8 bit) to the pixel last coded at the same
/// place that lossy inter-frame prediction still reuses unchanged. Lossless
/// coding reuses only identical pixels.
const INTERFRAME_TOLERANCE: u8 = 2;

/// One coded frame of an animation: which input it comes from, how long it is
/// shown, and the canvas rectangle (x, y, w, h) it covers (`None`: all of it).
struct PlannedFrame {
    index: usize,
    duration: u32,
    rect: Option<(u32, u32, u32, u32)>,
    /// Blend the cropped frame (lossless only): pixels equal to the canvas
    /// are made transparent.
    blend: bool,
}

/// Inter-frame prediction: a frame that differs from the canvas only in a
/// part is coded as that part alone (blocks of 8x8), replacing it on top of
/// the previous canvas. Pixels are compared with what was last coded at the
/// same place, so small changes cannot add up. A frame without any change
/// only extends the display time of the previous one.
fn plan_frames(frames: &[AnimationFrame], tolerance: u8) -> Vec<PlannedFrame> {
    let full = |index: usize| PlannedFrame {
        index,
        duration: frames[index].duration,
        rect: None,
        blend: false,
    };
    // Zero-duration frames are layers of the next frame; keep them as given.
    if frames.iter().any(|f| f.duration == 0) || std::env::var_os("MINI_NO_INTERFRAME").is_some() {
        return (0..frames.len()).map(full).collect();
    }
    let first = &frames[0].image;
    let (w, h, ch) = (
        first.width as usize,
        first.height as usize,
        first.channels(),
    );
    let mut canvas = first.data.to_vec();
    let mut plan = vec![full(0)];
    for (index, f) in frames.iter().enumerate().skip(1) {
        let data = f.image.data;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        // Changed pixels, and whether every one of them is opaque (a blended
        // frame cannot make a pixel transparent).
        let (mut count, mut opaque) = (0usize, true);
        for y in 0..h {
            let row = y * w * ch;
            let (a, b) = (&canvas[row..row + w * ch], &data[row..row + w * ch]);
            if a == b {
                continue;
            }
            // Alpha is coded losslessly, so it is compared exactly.
            let changed = |x: usize| {
                (0..ch).any(|c| {
                    let tol = if ch == 4 && c == 3 { 0 } else { tolerance };
                    a[x * ch + c].abs_diff(b[x * ch + c]) > tol
                })
            };
            if tolerance == 0 {
                for x in 0..w {
                    if changed(x) {
                        count += 1;
                        opaque &= ch == 3 || b[x * ch + 3] == 255;
                    }
                }
            }
            if let Some(l) = (0..w).find(|&x| changed(x)) {
                let r = (0..w).rfind(|&x| changed(x)).unwrap_or(l);
                x0 = x0.min(l);
                x1 = x1.max(r + 1);
                y0 = y0.min(y);
                y1 = y1.max(y + 1);
            }
        }
        if x1 == 0 {
            let prev = plan.last_mut().unwrap();
            if let Some(sum) = prev.duration.checked_add(f.duration) {
                prev.duration = sum;
                continue;
            }
            // The merged time would not fit the header: show this frame on
            // its own, coded as a single block (or whole, if that is all of it).
            let (bw, bh) = (w.min(8), h.min(8));
            let rect = (bw < w || bh < h).then_some((0, 0, bw as u32, bh as u32));
            for y in 0..bh {
                canvas[y * w * ch..(y * w + bw) * ch]
                    .copy_from_slice(&data[y * w * ch..(y * w + bw) * ch]);
            }
            plan.push(PlannedFrame {
                index,
                duration: f.duration,
                rect,
                blend: false,
            });
            continue;
        }
        let (x0, y0) = (x0 & !7, y0 & !7);
        let (x1, y1) = (x1.next_multiple_of(8).min(w), y1.next_multiple_of(8).min(h));
        let rect = (x1 - x0) * (y1 - y0) * 10 < w * h * 9;
        if rect {
            for y in y0..y1 {
                let o = (y * w + x0) * ch;
                canvas[o..o + (x1 - x0) * ch].copy_from_slice(&data[o..o + (x1 - x0) * ch]);
            }
        } else {
            canvas.copy_from_slice(data);
        }
        // Lossless frames in which most of the rectangle is unchanged are
        // blended, leaving the unchanged pixels transparent.
        let blend = rect
            && tolerance == 0
            && opaque
            && count * 10 < (x1 - x0) * (y1 - y0) * 8
            && std::env::var_os("MINI_NO_BLEND").is_none();
        plan.push(PlannedFrame {
            index,
            duration: f.duration,
            rect: rect.then_some((x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)),
            blend,
        });
    }
    plan
}
