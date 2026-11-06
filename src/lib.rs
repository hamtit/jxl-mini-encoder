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
}

impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            distance: 1.0,
            effort: 7,
            threads: 1,
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
    encode_frame(image, &opts, FramePlacement::STILL, &mut w);
    Ok(w.finish())
}

fn encode_frame(
    image: &ImageRef,
    opts: &EncoderOptions,
    placement: FramePlacement,
    w: &mut BitWriter,
) {
    if opts.distance <= 0.0 {
        lossless::encode(image, opts, placement, w);
    } else {
        vardct::encode(image, opts, placement, w);
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
    let mut w = BitWriter::with_capacity(first.image.data.len() * frames.len() / 8);
    let info = ImageInfo {
        width: first.image.width,
        height: first.image.height,
        has_alpha: first.image.alpha,
        xyb: !lossless,
        animation: Some(AnimationInfo {
            tps_numerator: anim.tps_numerator,
            tps_denominator: anim.tps_denominator,
            num_loops: anim.num_loops,
        }),
    };
    write_image_header(&mut w, &info);
    for (i, f) in frames.iter().enumerate() {
        let placement = FramePlacement {
            duration: Some(f.duration),
            is_last: i + 1 == frames.len(),
        };
        encode_frame(&f.image, &opts, placement, &mut w);
    }
    Ok(w.finish())
}

/// Version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
