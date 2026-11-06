//! Codestream headers: signature, size, image metadata and frame header.

use crate::bit_writer::BitWriter;
use crate::bit_writer::U32Dist::{Bits, Val};

/// What the image header needs to know about the image.
#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    /// Lossy (XYB) or lossless (original RGB).
    pub xyb: bool,
    /// Animation header, if the image is animated.
    pub animation: Option<AnimationInfo>,
}

/// Animation parameters stored in the image metadata.
#[derive(Clone, Copy, Debug)]
pub struct AnimationInfo {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    /// 0 loops forever.
    pub num_loops: u32,
}

/// Position of a frame in the codestream, for animations.
#[derive(Clone, Copy, Debug)]
pub struct FramePlacement {
    /// Display time in ticks; `None` for a still image.
    pub duration: Option<u32>,
    pub is_last: bool,
    /// Save the blended canvas in slot 1 for the next frame to build on.
    pub save: bool,
    /// The frame covers only this canvas rectangle (x0, y0) and replaces it
    /// on top of the canvas saved in slot 1. Its size is the image's size.
    pub crop: Option<(u32, u32)>,
    /// Alpha-blend a cropped frame onto the canvas instead of replacing it
    /// (needs an alpha channel; transparent pixels leave the canvas as is).
    pub blend: bool,
}

impl FramePlacement {
    pub const STILL: FramePlacement = FramePlacement {
        duration: None,
        is_last: true,
        save: false,
        crop: None,
        blend: false,
    };
}

fn write_size(w: &mut BitWriter, width: u32, height: u32) {
    let small = |v: u32| v.is_multiple_of(8) && (8..=256).contains(&v);
    let dims = [Bits(9, 1), Bits(13, 1), Bits(18, 1), Bits(30, 1)];
    let ratio = ratio_of(width, height);
    if small(width) && small(height) {
        w.write_bool(true);
        w.write(5, u64::from(height / 8 - 1));
        w.write(3, u64::from(ratio));
        if ratio == 0 {
            w.write(5, u64::from(width / 8 - 1));
        }
    } else {
        w.write_bool(false);
        w.write_u32(height, dims);
        w.write(3, u64::from(ratio));
        if ratio == 0 {
            w.write_u32(width, dims);
        }
    }
}

/// The aspect-ratio code that yields `width` from `height`, or 0.
fn ratio_of(width: u32, height: u32) -> u32 {
    let h = u64::from(height);
    let candidates = [
        h,
        h * 12 / 10,
        h * 4 / 3,
        h * 3 / 2,
        h * 16 / 9,
        h * 5 / 4,
        h * 2,
    ];
    candidates
        .iter()
        .position(|&c| c == u64::from(width))
        .map_or(0, |i| i as u32 + 1)
}

/// Signature, SizeHeader and ImageMetadata, padded to a byte boundary.
pub fn write_image_header(w: &mut BitWriter, info: &ImageInfo) {
    w.write(8, 0xff);
    w.write(8, 0x0a);
    write_size(w, info.width, info.height);
    if info.xyb && !info.has_alpha && info.animation.is_none() {
        w.write_bool(true); // all_default: 8-bit sRGB, XYB
    } else {
        w.write_bool(false); // all_default
        w.write_bool(info.animation.is_some()); // extra_fields
        if let Some(a) = info.animation {
            w.write(3, 0); // orientation - 1
            w.write_bool(false); // have_intrinsic_size
            w.write_bool(false); // have_preview
            w.write_bool(true); // have_animation
            w.write_u32(
                a.tps_numerator,
                [Val(100), Val(1000), Bits(10, 1), Bits(30, 1)],
            );
            w.write_u32(
                a.tps_denominator,
                [Val(1), Val(1001), Bits(8, 1), Bits(10, 1)],
            );
            w.write_u32(a.num_loops, [Val(0), Bits(3, 0), Bits(16, 0), Bits(32, 0)]);
            w.write_bool(false); // have_timecodes
        }
        w.write_bool(false); // float_sample
        w.write_u32(8, [Val(8), Val(10), Val(12), Bits(6, 1)]);
        w.write_bool(true); // modular_16bit_buffers
        let num_extra = u32::from(info.has_alpha);
        w.write_u32(num_extra, [Val(0), Val(1), Bits(4, 2), Bits(12, 1)]);
        if info.has_alpha {
            w.write_bool(true); // default alpha channel: 8-bit, straight
        }
        w.write_bool(info.xyb);
        w.write_bool(true); // colour_encoding all_default: sRGB
        if info.animation.is_some() {
            w.write_bool(true); // tone_mapping all_default (read when extra_fields)
        }
        w.write_u64(0); // extensions
    }
    w.write_bool(true); // default_m (default opsin matrix and upsampling weights)
    w.zero_pad_to_byte();
}

#[derive(Clone, Copy, Debug)]
pub enum Encoding {
    VarDct { x_qm_scale: u32, b_qm_scale: u32 },
    Modular,
}

/// The parts of a frame header this encoder varies.
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    pub encoding: Encoding,
    pub num_extra: u32,
    pub xyb: bool,
    pub gaborish: bool,
    pub epf_iters: u32,
    /// Do not smooth the LF image in the decoder.
    pub skip_lf_smoothing: bool,
    /// The frame carries a patch dictionary.
    pub patches: bool,
    /// Noise synthesis LUT (10 bit entries); the parameters follow in LfGlobal.
    pub noise: Option<[u16; 8]>,
    /// A patch reference frame of this size, saved in slot 3 and not shown.
    pub reference: Option<(u32, u32)>,
    pub placement: FramePlacement,
}

pub const GROUP_DIM: usize = 256;
/// Group size shift for modular frames (128 << 1 = 256).
const GROUP_SIZE_SHIFT: u32 = 1;

pub fn write_frame_header(w: &mut BitWriter, f: &FrameInfo, size: (u32, u32)) {
    w.write_bool(false); // all_default
    let is_ref = f.reference.is_some();
    w.write(2, if is_ref { 2 } else { 0 }); // reference only / regular frame
    let modular = matches!(f.encoding, Encoding::Modular);
    w.write(1, u64::from(modular));
    // flags: kPatches = 2, kSkipAdaptiveDCSmoothing = 128
    w.write_u64(
        u64::from(f.noise.is_some())
            + u64::from(f.patches) * 2
            + if f.skip_lf_smoothing { 128 } else { 0 },
    );
    if !f.xyb {
        w.write_bool(false); // do_ycbcr
    }
    let upsampling = [Val(1), Val(2), Val(4), Val(8)];
    w.write_u32(1, upsampling);
    for _ in 0..f.num_extra {
        w.write_u32(1, upsampling);
    }
    match f.encoding {
        Encoding::Modular => w.write(2, u64::from(GROUP_SIZE_SHIFT)),
        Encoding::VarDct {
            x_qm_scale,
            b_qm_scale,
        } => {
            if f.xyb {
                w.write(3, u64::from(x_qm_scale));
                w.write(3, u64::from(b_qm_scale));
            }
        }
    }
    if !is_ref {
        w.write_u32(1, [Val(1), Val(2), Val(3), Bits(3, 4)]); // num_passes
    }
    match f.reference {
        Some((rw, rh)) => {
            w.write_bool(true); // custom size
            let size = [Bits(8, 0), Bits(11, 256), Bits(14, 2304), Bits(30, 18688)];
            w.write_u32(rw, size);
            w.write_u32(rh, size);
            w.write_u32(3, [Val(0), Val(1), Val(2), Val(3)]); // save_as_reference
            w.write_bool(true); // save_before_color_transform
        }
        None => {
            let place = f.placement;
            let coord = [Bits(8, 0), Bits(11, 256), Bits(14, 2304), Bits(30, 18688)];
            w.write_bool(place.crop.is_some()); // have_crop
            if let Some((x0, y0)) = place.crop {
                // x0 / y0 are signed (zig-zag packed); width / height are not.
                w.write_u32(x0 * 2, coord);
                w.write_u32(y0 * 2, coord);
                w.write_u32(size.0, coord);
                w.write_u32(size.1, coord);
            }
            let blend_mode = [Val(0), Val(1), Val(2), Bits(2, 3)];
            // A cropped frame is drawn onto slot 1, which then also needs its
            // source written. Blending uses the first extra channel as alpha.
            let source = if place.crop.is_some() { 1 } else { 0 };
            let blend = place.blend && place.crop.is_some() && f.num_extra > 0;
            let mode = if blend { 2 } else { 0 };
            for _ in 0..=f.num_extra {
                w.write_u32(mode, blend_mode);
                if blend {
                    w.write_u32(0, [Val(0), Val(1), Val(2), Bits(3, 3)]); // alpha channel
                    w.write_bool(false); // clamp
                }
                if place.crop.is_some() {
                    w.write(2, source);
                }
            }
            if let Some(d) = place.duration {
                w.write_u32(d, [Val(0), Val(1), Bits(8, 0), Bits(32, 0)]);
            }
            w.write_bool(place.is_last);
            if !place.is_last {
                w.write_u32(u32::from(place.save), [Val(0), Val(1), Val(2), Val(3)]); // save_as_reference
                // Only a frame that replaces the whole canvas carries this flag.
                if place.crop.is_none() && (place.duration == Some(0) || place.save) {
                    w.write_bool(false); // save_before_color_transform
                }
            }
        }
    }
    w.write_u32(0, [Val(0), Bits(4, 0), Bits(5, 16), Bits(10, 48)]); // name length
    // Restoration filter.
    w.write_bool(false); // all_default
    w.write_bool(f.gaborish);
    if f.gaborish {
        w.write_bool(false); // default weights
    }
    w.write(2, u64::from(f.epf_iters));
    if f.epf_iters > 0 {
        if !modular {
            w.write_bool(false); // default sharpness LUT
        }
        w.write_bool(false); // default channel scale
        w.write_bool(false); // default sigma
        if modular {
            w.write_f16(1.0); // sigma for modular
        }
    }
    w.write_u64(0); // restoration filter extensions
    w.write_u64(0); // frame header extensions
}

/// Write the table of contents for the given section sizes (in bytes).
pub fn write_toc(w: &mut BitWriter, sizes: &[usize]) {
    w.write_bool(false); // no permutation
    w.zero_pad_to_byte();
    for &s in sizes {
        let s = u32::try_from(s).expect("section too large");
        w.write_u32(
            s,
            [
                Bits(10, 0),
                Bits(14, 1024),
                Bits(22, 17408),
                Bits(30, 4_211_712),
            ],
        );
    }
    w.zero_pad_to_byte();
}

/// Frame layout derived from the image size.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub groups_x: usize,
    pub groups_y: usize,
    pub lf_groups_x: usize,
    pub lf_groups_y: usize,
}

impl Layout {
    pub fn new(width: usize, height: usize) -> Self {
        let lf = GROUP_DIM * 8;
        Layout {
            width,
            height,
            groups_x: width.div_ceil(GROUP_DIM),
            groups_y: height.div_ceil(GROUP_DIM),
            lf_groups_x: width.div_ceil(lf),
            lf_groups_y: height.div_ceil(lf),
        }
    }

    pub fn num_groups(&self) -> usize {
        self.groups_x * self.groups_y
    }

    pub fn num_lf_groups(&self) -> usize {
        self.lf_groups_x * self.lf_groups_y
    }

    /// Pixel rectangle (x, y, w, h) of a group.
    pub fn group_rect(&self, g: usize) -> (usize, usize, usize, usize) {
        let (gx, gy) = (g % self.groups_x, g / self.groups_x);
        let (x, y) = (gx * GROUP_DIM, gy * GROUP_DIM);
        (
            x,
            y,
            GROUP_DIM.min(self.width - x),
            GROUP_DIM.min(self.height - y),
        )
    }

    /// Pixel rectangle of an LF group.
    pub fn lf_group_rect(&self, g: usize) -> (usize, usize, usize, usize) {
        let d = GROUP_DIM * 8;
        let (gx, gy) = (g % self.lf_groups_x, g / self.lf_groups_x);
        let (x, y) = (gx * d, gy * d);
        (x, y, d.min(self.width - x), d.min(self.height - y))
    }
}

/// Concatenate frame sections behind a TOC. A frame with a single group and
/// pass has a single section holding everything.
pub fn write_sections(w: &mut BitWriter, sections: Vec<BitWriter>) {
    let bytes: Vec<Vec<u8>> = sections.into_iter().map(BitWriter::finish).collect();
    let sizes: Vec<usize> = bytes.iter().map(Vec::len).collect();
    write_toc(w, &sizes);
    for b in &bytes {
        w.append_bytes(b);
    }
}
