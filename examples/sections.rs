//! Print the section sizes of JPEG XL files (LF global / LF groups / HF global / pass groups).
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let image = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&bytes)).unwrap();
        let frame = image.frame(0).unwrap();
        let toc = frame.toc();
        let (mut lfg, mut lfgr, mut hfg, mut pg, mut all) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for g in toc.iter_bitstream_order() {
            let s = u64::from(g.size);
            let k = format!("{:?}", g.kind);
            if k == "All" {
                all += s
            } else if k == "LfGlobal" {
                lfg += s
            } else if k.starts_with("LfGroup") {
                lfgr += s
            } else if k == "HfGlobal" {
                hfg += s
            } else {
                pg += s
            }
        }
        let h = frame.header();
        println!(
            "{}: total {} | all {all} lfglobal {lfg} lfgroup {lfgr} hfglobal {hfg} passgroups {pg} | flags {:?} {:?}",
            path.rsplit('/').next().unwrap(),
            bytes.len(),
            h.flags,
            h.restoration_filter
        );
    }
}
