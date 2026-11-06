use super::*;

#[test]
fn epf_strength_keeps_high_fidelity_and_unsharpened_images_unchanged() {
    for distance in [0.1, 0.5, 1.0, 1.499] {
        assert_eq!(epf_sharpness(distance, true), 4);
    }
    for distance in [1.5, 2.35, 3.7, 5.5, 25.0] {
        assert_eq!(epf_sharpness(distance, true), 3);
    }
    for distance in [0.1, 1.0, 2.35, 5.5, 25.0] {
        assert_eq!(epf_sharpness(distance, false), 4);
    }
}

#[test]
fn flat_graphics_requires_eighty_five_percent_constant_tiles() {
    let mut pixels = vec![127; 80 * 80 * 3];
    for tile in 0..16 {
        let x = tile % 10 * 8 + 1;
        let y = tile / 10 * 8;
        pixels[(y * 80 + x) * 3] = 128;
        let image = ImageRef::rgb8(80, 80, &pixels).unwrap();
        assert_eq!(flat_graphics(&image), tile < 15);
    }
}

#[test]
fn flat_graphics_detection_ignores_alpha_but_not_rgb_gradients() {
    for (w, h) in [(1, 1), (7, 17), (16, 16), (129, 65), (1024, 1024)] {
        let pixels: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let v = if (x / 32 + y / 32) % 2 == 0 { 0 } else { 255 };
                    [v, v, v, (x ^ y) as u8]
                })
            })
            .collect();
        let image = ImageRef::rgba8(w, h, &pixels).unwrap();
        assert_eq!(flat_graphics(&image), w / 8 * (h / 8) >= 4);
        let gradient: Vec<u8> = (0..w * h)
            .flat_map(|i| [i as u8, (i / w) as u8, 0])
            .collect();
        let image = ImageRef::rgb8(w, h, &gradient).unwrap();
        assert!(!flat_graphics(&image));
    }
}

fn pattern_image(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> u8) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let v = f(i % w, i / w);
            [v, v, v]
        })
        .collect()
}

#[test]
fn noise_level_separates_smooth_images_from_noisy_ones() {
    let (w, h) = (64, 48);
    let smooth = pattern_image(w, h, |x, y| (x * 2 + y) as u8);
    let flat = pattern_image(w, h, |_, _| 90);
    // Ordered dithering and pseudo-random noise of a few levels.
    let dither = pattern_image(w, h, |x, y| 100 + ((x ^ y) & 1) as u8 * 24);
    let mut s = 1u32;
    let noisy = pattern_image(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (100 + (s >> 28)) as u8
    });
    let level = |d: &[u8]| noise_level(&ImageRef::rgb8(w as u32, h as u32, d).unwrap());
    assert_eq!(level(&flat), 0);
    assert!(level(&smooth) < NOISY_LEVEL);
    assert!(level(&dither) >= NOISY_LEVEL);
    assert!(level(&noisy) >= NOISY_LEVEL);
}

#[test]
fn noise_level_handles_tiny_images() {
    let px = [0u8; 12];
    assert_eq!(noise_level(&ImageRef::rgb8(2, 2, &px).unwrap()), 0);
    assert_eq!(noise_level(&ImageRef::rgb8(1, 4, &px).unwrap()), 0);
}
