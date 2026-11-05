//! Patches: small repeated shapes (glyphs, icons) are coded once in a
//! reference frame and added back at every place they occur, following
//! libjxl's `FindTextLikePatches`. The image is coded without them.

use crate::bit_writer::{BitWriter, round_to_f16};
use crate::entropy::{CodeOptions, Token, pack_signed, write_stream};
use crate::headers::{Encoding, FrameInfo, GROUP_DIM, Layout, write_frame_header, write_sections};
use crate::modular::{Channel, GlobalTree, ModularOptions, write_header, write_standalone};
use crate::parallel::par_map;
use crate::vardct::xyb::Xyb;

/// Largest side of a patch plus one (libjxl's kMaxPatchSize).
const MAX_PATCH_SIZE: usize = 32;
const SIDE: usize = 4;
/// Reference frame slot used for the patch image.
const REF_SLOT: u32 = 3;

// Quantisation steps used to decide that two patches are the same, and the
// weights of the (X, Y, B) channels in the similarity measure.
const DEQUANT: [f32; 3] = [0.01615, 0.08875, 0.1922];
const WEIGHTS: [f32; 3] = [30.0, 3.0, 1.0];

/// The reference frame's modular integers are multiples of these steps, which
/// are coarser at lower quality, as in libjxl (X, Y, B).
fn dc_quant(distance: f32) -> [f32; 3] {
    let inv = [
        65536.0 / (1.0 + 23.0 * distance),
        4096.0 / (1.0 + 14.0 * distance),
        4096.0 / (1.0 + 14.0 * distance),
    ];
    // The header holds 128 / inv as a half float.
    inv.map(|i| round_to_f16(128.0 / i) / 128.0)
}

// Contexts of the dictionary stream.
const NUM_REF_PATCH: u32 = 0;
const REFERENCE_FRAME: u32 = 1;
const PATCH_SIZE: u32 = 2;
const PATCH_REF_POSITION: u32 = 3;
const PATCH_POSITION: u32 = 4;
const PATCH_BLEND_MODE: u32 = 5;
const PATCH_OFFSET: u32 = 6;
const PATCH_COUNT: u32 = 7;
const NUM_CONTEXTS: usize = 10;

const BLEND_NONE: u32 = 0;
const BLEND_ADD: u32 = 2;

struct Patch {
    w: usize,
    h: usize,
    /// Position in the reference frame.
    rx: usize,
    ry: usize,
    /// Pixel differences to the background, per channel (X, Y, B).
    pixels: [Vec<f32>; 3],
    positions: Vec<(usize, usize)>,
}

/// The patches of an image and the reference frame that holds their pixels.
pub struct Patches {
    patches: Vec<Patch>,
    ref_w: usize,
    ref_h: usize,
    /// Reference frame samples as the decoder reconstructs them (X, Y, B).
    decoded: [Vec<f32>; 3],
    /// Modular channels of the reference frame: Y, X, B - Y.
    channels: [Channel; 3],
    /// The decoder's DC dequantisation steps (X, Y, B), as coded.
    dc_quant: [f32; 3],
}

fn similar(a: [f32; 3], b: [f32; 3], threshold: f32) -> bool {
    let d: f32 = (0..3).map(|c| (a[c] - b[c]).abs() * WEIGHTS[c]).sum();
    d <= threshold
}

/// Look for patches in the XYB image (the first `w` x `h` pixels).
pub fn find(xyb: &Xyb, w: usize, h: usize, distance: f32) -> Option<Patches> {
    let stride = xyb.w;
    let px = |x: usize, y: usize| -> [f32; 3] {
        let i = y * stride + x;
        [xyb.planes[0][i], xyb.planes[1][i], xyb.planes[2][i]]
    };
    let (pw, ph) = (w / SIDE, h / SIDE);
    if pw < 3 || ph < 3 {
        return None;
    }
    let same_color = |x: usize, y: usize, c: [f32; 3]| {
        let p = px(x, y);
        (0..3).all(|i| (c[i] - p[i]).abs() <= 1e-4)
    };

    // Flat 4x4 squares with at least eight flat neighbours of the same
    // colour mark "screenshot like" areas.
    let mut seeds = vec![false; pw * ph];
    let mut num_seeds = 0usize;
    for py in 1..ph - 2 {
        for qx in 1..=pw - 2 {
            let (ox, oy) = (qx * SIDE, py * SIDE);
            let base = px(ox, oy);
            let flat = (0..SIDE).all(|iy| (0..SIDE).all(|ix| same_color(ox + ix, oy + iy, base)));
            if !flat {
                continue;
            }
            let mut same = 0;
            for y in (py - 1..=py + 1).map(|v| v * SIDE) {
                for x in (qx - 1..=qx + 1).map(|v| v * SIDE) {
                    same += usize::from(same_color(x, y, base));
                }
            }
            if same >= 8 {
                seeds[py * pw + qx] = true;
                num_seeds += 1;
            }
        }
    }
    if num_seeds == 0 {
        return None;
    }

    // Grow the background from the seeds over similar pixels. Pixel indices
    // are u32 and the colour of a background pixel is looked up through the
    // pixel it grew from, which keeps the memory use small on large images.
    if w * h > u32::MAX as usize {
        return None;
    }
    const DISTANCE_LIMIT: usize = 50;
    let mut is_bg = vec![false; w * h];
    let mut bg_src = vec![0u32; w * h];
    // Seed pixels are background by themselves; only those next to a pixel
    // that is not (yet) background can grow it, so only those are queued.
    // The queue is FIFO and consumed entries are dropped, so it holds only
    // the frontier of the flood fill.
    // Entries carry their coordinates and the source colour so that the hot
    // loop needs neither divisions nor a lookup of the source pixel.
    struct Item {
        x: u32,
        y: u32,
        sx: u32,
        sy: u32,
        color: [f32; 3],
    }
    let mut queue: std::collections::VecDeque<Item> = std::collections::VecDeque::new();
    for py in 1..ph - 1 {
        for qx in 1..pw - 1 {
            if !seeds[py * pw + qx] {
                continue;
            }
            for y in py * SIDE..(py + 1) * SIDE {
                for x in qx * SIDE..(qx + 1) * SIDE {
                    is_bg[y * w + x] = true;
                    bg_src[y * w + x] = (y * w + x) as u32;
                }
            }
        }
    }
    for py in 1..ph - 1 {
        for qx in 1..pw - 1 {
            if !seeds[py * pw + qx] {
                continue;
            }
            // Pixels of a block whose eight neighbouring blocks are seeds too
            // are never next to a non-background pixel.
            if (py - 1..=py + 1).all(|v| (qx - 1..=qx + 1).all(|u| seeds[v * pw + u])) {
                continue;
            }
            for y in py * SIDE..(py + 1) * SIDE {
                for x in qx * SIDE..(qx + 1) * SIDE {
                    let open = (y.saturating_sub(1)..(y + 2).min(h)).any(|ny| {
                        (x.saturating_sub(1)..(x + 2).min(w)).any(|nx| !is_bg[ny * w + nx])
                    });
                    if open {
                        queue.push_back(Item {
                            x: x as u32,
                            y: y as u32,
                            sx: x as u32,
                            sy: y as u32,
                            color: px(x, y),
                        });
                    }
                }
            }
        }
    }
    while let Some(Item {
        x: cx,
        y: cy,
        sx,
        sy,
        color,
    }) = queue.pop_front()
    {
        let (cx, cy) = (cx as usize, cy as usize);
        bg_src[cy * w + cx] = sy * w as u32 + sx;
        let (x0, x1) = (cx.saturating_sub(1), (cx + 2).min(w));
        for ny in cy.saturating_sub(1)..(cy + 2).min(h) {
            let row = ny * w;
            for nx in x0..x1 {
                if is_bg[row + nx] {
                    continue;
                }
                if nx.abs_diff(sx as usize) + ny.abs_diff(sy as usize) > DISTANCE_LIMIT {
                    continue;
                }
                if similar(color, px(nx, ny), 0.8) {
                    queue.push_back(Item {
                        x: nx as u32,
                        y: ny as u32,
                        sx,
                        sy,
                        color,
                    });
                    is_bg[row + nx] = true;
                }
            }
        }
    }
    drop(queue);
    let background = |x: usize, y: usize| -> [f32; 3] {
        let s = bg_src[y * w + x] as usize;
        px(s % w, s / w)
    };

    // Connected components outside the background are candidate patches.
    const MIN_PEAK: i32 = 2;
    const SIMILAR_RADIUS: usize = 2;
    // Background pixels count as visited, so the fills below test one array.
    // It has a one-pixel border (also "visited") so neighbours need no bounds
    // checks: `vi` maps a pixel to its index.
    let vw = w + 2;
    let vi = |x: usize, y: usize| (y + 1) * vw + x + 1;
    let mut visited = vec![true; vw * (h + 2)];
    for y in 0..h {
        visited[vi(0, y)..vi(0, y) + w].copy_from_slice(&is_bg[y * w..(y + 1) * w]);
    }
    type Key = (usize, usize, [Vec<i8>; 3]);
    type Found = (Key, [Vec<f32>; 3], Vec<(usize, usize)>);
    let mut found: Vec<Found> = Vec::new();
    let mut index: std::collections::HashMap<Key, usize> = std::collections::HashMap::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut fill: Vec<u32> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if visited[vi(x, y)] {
                continue;
            }
            stack.clear();
            stack.push((x, y));
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (x, x, y, y);
            let mut reference: Option<(usize, usize)> = None;
            let mut all_similar = true;
            let mut oversize = None;
            while let Some(cur) = stack.pop() {
                if visited[vi(cur.0, cur.1)] {
                    continue;
                }
                visited[vi(cur.0, cur.1)] = true;
                min_x = min_x.min(cur.0);
                max_x = max_x.max(cur.0);
                min_y = min_y.min(cur.1);
                max_y = max_y.max(cur.1);
                if max_x - min_x >= MAX_PATCH_SIZE || max_y - min_y >= MAX_PATCH_SIZE {
                    // `cur`'s neighbours are not pushed yet; the fill below
                    // starts from `cur` as well so none are missed.
                    oversize = Some(cur);
                    break;
                }
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (cur.0 as i32 + dx, cur.1 as i32 + dy);
                        if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                            continue;
                        }
                        let next = (nx as usize, ny as usize);
                        if !is_bg[next.1 * w + next.0] {
                            stack.push(next);
                        } else if let Some(r) = reference {
                            if !similar(background(next.0, next.1), background(r.0, r.1), 0.03) {
                                all_similar = false;
                            }
                        } else {
                            reference = Some(next);
                        }
                    }
                }
            }
            if let Some(cutoff) = oversize {
                // Too big to be a patch whatever else is true. Mark the rest
                // of the component as visited, marking on push so that the
                // stack never holds more than one entry per pixel.
                fill.clear();
                fill.push(vi(cutoff.0, cutoff.1) as u32);
                for &(sx, sy) in &stack {
                    let i = vi(sx, sy);
                    if !visited[i] {
                        visited[i] = true;
                        fill.push(i as u32);
                    }
                }
                let around = [
                    -(vw as isize) - 1,
                    -(vw as isize),
                    -(vw as isize) + 1,
                    -1,
                    1,
                    vw as isize - 1,
                    vw as isize,
                    vw as isize + 1,
                ];
                while let Some(cur) = fill.pop() {
                    for off in around {
                        let i = (cur as isize + off) as usize;
                        if !visited[i] {
                            visited[i] = true;
                            fill.push(i as u32);
                        }
                    }
                }
                continue;
            }
            let Some(reference) = reference else { continue };
            if !all_similar || max_x - min_x >= MAX_PATCH_SIZE || max_y - min_y >= MAX_PATCH_SIZE {
                continue;
            }
            let rc = background(reference.0, reference.1);
            let has_similar = (min_y.saturating_sub(SIMILAR_RADIUS)
                ..(max_y + SIMILAR_RADIUS + 1).min(h))
                .any(|iy| {
                    (min_x.saturating_sub(SIMILAR_RADIUS)..(max_x + SIMILAR_RADIUS + 1).min(w))
                        .any(|ix| similar(rc, px(ix, iy), 0.03))
                });
            if !has_similar {
                continue;
            }
            let (pxs, pys) = (max_x - min_x + 1, max_y - min_y + 1);
            let mut fpix: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; pxs * pys]);
            let mut qpix: [Vec<i8>; 3] = std::array::from_fn(|_| vec![0; pxs * pys]);
            let (mut too_big, mut too_small) = (false, true);
            for c in [1usize, 0, 2] {
                for iy in min_y..=max_y {
                    for ix in min_x..=max_x {
                        let o = (iy - min_y) * pxs + ix - min_x;
                        let f = px(ix, iy)[c] - rc[c];
                        fpix[c][o] = f;
                        let val = (f / DEQUANT[c]).clamp(-32768.0, 32767.0).trunc() as i32;
                        let q = val as i8;
                        qpix[c][o] = q;
                        too_big |= val != i32::from(q);
                        too_small &= val < MIN_PEAK && val > -MIN_PEAK;
                    }
                }
            }
            if too_small || too_big {
                continue;
            }
            let key = (pxs, pys, qpix);
            match index.get(&key) {
                Some(&i) => found[i].2.push((min_x, min_y)),
                None => {
                    index.insert(key.clone(), found.len());
                    found.push((key, fpix, vec![(min_x, min_y)]));
                }
            }
        }
    }

    // Keep shapes that occur at least twice; skip if all are tiny.
    const MIN_OCCURRENCES: usize = 2;
    const MIN_MAX_PATCH_PIXELS: usize = 20;
    let mut patches: Vec<Patch> = found
        .into_iter()
        .filter(|(_, _, pos)| pos.len() >= MIN_OCCURRENCES)
        .map(|((pxs, pys, _), pixels, positions)| Patch {
            w: pxs,
            h: pys,
            rx: 0,
            ry: 0,
            pixels,
            positions,
        })
        .collect();
    if patches.iter().map(|p| p.w * p.h).max().unwrap_or(0) < MIN_MAX_PATCH_PIXELS {
        return None;
    }
    patches.sort_by_key(|p| std::cmp::Reverse(p.w * p.h));

    // Pack the patches into one reference image.
    let max_x = patches.iter().map(|p| p.w).max()?;
    let max_y = patches.iter().map(|p| p.h).max()?;
    let total: usize = patches.iter().map(|p| p.w * p.h).sum();
    let root = (total as f32).sqrt();
    let (mut rw, mut rh) = (
        (max_x as f32).max(root) as usize,
        (max_y as f32).max(root) as usize,
    );
    let used_h = loop {
        rw = (rw as f32 * 1.05 + 1.0) as usize;
        rh = (rh as f32 * 1.05 + 1.0) as usize;
        let mut occupied = vec![false; rw * rh];
        let mut max_row = 0;
        let mut ok = true;
        for p in &mut patches {
            let mut spot = None;
            let mut y0 = 0;
            'search: while y0 + p.h <= rh {
                let mut x0 = 0;
                while x0 + p.w <= rw {
                    let mut blocked = None;
                    'check: for y in y0..y0 + p.h {
                        for x in x0..x0 + p.w {
                            if occupied[y * rw + x] {
                                blocked = Some(x);
                                break 'check;
                            }
                        }
                    }
                    match blocked {
                        None => {
                            spot = Some((x0, y0));
                            break 'search;
                        }
                        Some(x) => x0 = x + 1,
                    }
                }
                y0 += 1;
            }
            let Some((x0, y0)) = spot else {
                ok = false;
                break;
            };
            p.rx = x0;
            p.ry = y0;
            for y in y0..y0 + p.h {
                occupied[y * rw + x0..y * rw + x0 + p.w].fill(true);
            }
            max_row = max_row.max(y0 + p.h);
        }
        if ok {
            break max_row;
        }
    };
    let rh = used_h;

    // Quantise the reference image the way the decoder will read it.
    let mut channels = [
        Channel::new(rw, rh),
        Channel::new(rw, rh),
        Channel::new(rw, rh),
    ];
    let mut decoded: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; rw * rh]);
    let dq = dc_quant(distance);
    let inv: [f32; 3] = dq.map(|q| 1.0 / q);
    for p in &patches {
        for y in 0..p.h {
            for x in 0..p.w {
                let (o, i) = ((p.ry + y) * rw + p.rx + x, y * p.w + x);
                let qx = (p.pixels[0][i] * inv[0] + 0.5).floor() as i32;
                let qy = (p.pixels[1][i] * inv[1] + 0.5).floor() as i32;
                let qb = (p.pixels[2][i] * inv[2] + 0.5).floor() as i32;
                channels[0].data[o] = qy;
                channels[1].data[o] = qx;
                channels[2].data[o] = qb - qy;
                decoded[0][o] = qx as f32 * dq[0];
                decoded[1][o] = qy as f32 * dq[1];
                decoded[2][o] = qb as f32 * dq[2];
            }
        }
    }
    Some(Patches {
        patches,
        ref_w: rw,
        ref_h: rh,
        decoded,
        channels,
        dc_quant: dq,
    })
}

impl Patches {
    /// Remove the patches from the image (they are added back when decoding).
    pub fn subtract(&self, xyb: &mut Xyb) {
        let stride = xyb.w;
        for p in &self.patches {
            for &(px, py) in &p.positions {
                for y in 0..p.h {
                    for x in 0..p.w {
                        let r = (p.ry + y) * self.ref_w + p.rx + x;
                        let i = (py + y) * stride + px + x;
                        for c in 0..3 {
                            xyb.planes[c][i] -= self.decoded[c][r];
                        }
                    }
                }
            }
        }
    }

    /// The patch dictionary stream (at the start of the global section).
    pub fn write_dictionary(&self, num_extra: u32, w: &mut BitWriter) {
        let mut t: Vec<Token> = Vec::new();
        let mut add = |ctx: u32, v: u32| t.push(Token::new(ctx, v));
        add(NUM_REF_PATCH, self.patches.len() as u32);
        for p in &self.patches {
            add(REFERENCE_FRAME, REF_SLOT);
            add(PATCH_REF_POSITION, p.rx as u32);
            add(PATCH_REF_POSITION, p.ry as u32);
            add(PATCH_SIZE, p.w as u32 - 1);
            add(PATCH_SIZE, p.h as u32 - 1);
            add(PATCH_COUNT, p.positions.len() as u32 - 1);
            for (j, &(x, y)) in p.positions.iter().enumerate() {
                if j == 0 {
                    add(PATCH_POSITION, x as u32);
                    add(PATCH_POSITION, y as u32);
                } else {
                    let (lx, ly) = p.positions[j - 1];
                    add(PATCH_OFFSET, pack_signed(x as i32 - lx as i32));
                    add(PATCH_OFFSET, pack_signed(y as i32 - ly as i32));
                }
                // Colour is added; extra channels are left alone.
                add(PATCH_BLEND_MODE, BLEND_ADD);
                for _ in 0..num_extra {
                    add(PATCH_BLEND_MODE, BLEND_NONE);
                }
            }
        }
        write_stream(&t, NUM_CONTEXTS, CodeOptions::default(), w);
    }

    /// Write the reference frame (an XYB Modular frame, not shown).
    pub fn write_reference_frame(
        &self,
        num_extra: u32,
        effort: u32,
        threads: usize,
        out: &mut BitWriter,
    ) {
        let (w, h) = (self.ref_w, self.ref_h);
        let mut channels: Vec<Channel> = self.channels.to_vec();
        // The decoder expects every extra channel in every frame.
        for _ in 0..num_extra {
            channels.push(Channel::new(w, h));
        }
        let mopts = ModularOptions::for_effort(effort.min(7), w * h * channels.len())
            .nearest_channel_only()
            .with_threads(threads);
        write_frame_header(
            out,
            &FrameInfo {
                encoding: Encoding::Modular,
                num_extra,
                xyb: true,
                gaborish: false,
                epf_iters: 0,
                skip_lf_smoothing: false,
                patches: false,
                reference: Some((w as u32, h as u32)),
            },
        );
        let write_dc_quant = |s: &mut BitWriter| {
            s.write_bool(false); // custom DC dequantisation
            for q in self.dc_quant {
                s.write_f16(q * 128.0);
            }
        };
        if w <= GROUP_DIM && h <= GROUP_DIM {
            let refs: Vec<&Channel> = channels.iter().collect();
            let mut s = BitWriter::new();
            write_dc_quant(&mut s);
            s.write_bool(false); // no global tree
            write_standalone(&mut s, &refs, 0, &[], &mopts);
            write_sections(out, vec![s]);
            return;
        }
        let layout = Layout::new(w, h);
        let (ng, nlf) = (layout.num_groups(), layout.num_lf_groups());
        let streams: Vec<(u32, Vec<Channel>)> = (0..ng)
            .map(|g| {
                let (x, y, gw, gh) = layout.group_rect(g);
                (
                    (1 + 3 * nlf + 17 + g) as u32,
                    channels.iter().map(|c| c.crop(x, y, gw, gh)).collect(),
                )
            })
            .collect();
        let global = GlobalTree::build_owned(streams, &mopts);
        let mut lf_global = BitWriter::new();
        write_dc_quant(&mut lf_global);
        lf_global.write_bool(true); // global tree
        global.write(&mut lf_global);
        write_header(&mut lf_global, true, &[]);
        global.write_empty_stream(&mut lf_global);
        let mut sections = vec![lf_global];
        sections.extend((0..nlf).map(|_| BitWriter::new()));
        sections.push(BitWriter::new()); // HF global
        sections.extend(par_map(ng, threads, |g| {
            let mut s = BitWriter::new();
            write_header(&mut s, true, &[]);
            global.write_stream(g, &mut s);
            s
        }));
        write_sections(out, sections);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A component larger than a patch is discarded as a whole, including
    /// the part only reachable through the pixel where the limit was hit.
    #[test]
    fn oversized_component_is_discarded_entirely() {
        let (w, h) = (160usize, 120usize);
        let mut planes = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
        for x in [40, 100] {
            for y in 30..90 {
                for p in &mut planes {
                    p[y * w + x] = 0.5;
                }
            }
        }
        let xyb = Xyb { w, h, planes };
        assert!(find(&xyb, w, h, 1.0).is_none());
    }
}
