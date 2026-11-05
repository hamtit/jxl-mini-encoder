#!/usr/bin/env python3
"""Compare this encoder with libjxl's cjxl at the same quality and effort.

Both encoders' output is decoded and scored with SSIMULACRA2 by the `eval`
example (libjxl's files are passed through `REF_DIR`), so the numbers come from
the same decoder and metric. No Python dependencies; needs a prebuilt `eval`
(`cargo build --release --example eval`) and a `cjxl` binary.

  tools/compare_libjxl.py --cjxl /path/to/cjxl -q 50 -q 75 -q 90 -e 7 images/*.png

Prints, per quality and effort, total bytes and mean SSIMULACRA2 of each
encoder and the differences. A negative score difference with fewer bytes is
not necessarily a regression: compare on the rate-distortion curve, or use
several qualities.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOTAL = re.compile(r"total (\d+) B, [\d.]+ bpp, \d+ ms, mean ssim2 (-?[\d.]+)")


def evaluate(eval_bin, images, quality, effort, ref_dir=None):
    env = None
    if ref_dir is not None:
        env = {"REF_DIR": str(ref_dir)}
    cmd = [str(eval_bin), "-q", str(quality), "-e", str(effort), "-t", "1", *map(str, images)]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode:
        sys.exit(f"eval failed: {proc.stderr}")
    match = TOTAL.search(proc.stdout)
    if match is None:
        sys.exit(f"unrecognized eval output: {proc.stdout}")
    return int(match[1]), float(match[2])


def encode_reference(cjxl, images, quality, effort, ref_dir):
    for image in images:
        out = ref_dir / f"ref_{image.name}.jxl"
        cmd = [str(cjxl), str(image), str(out), "-q", str(quality), "-e", str(effort), "--num_threads=1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            sys.exit(f"cjxl failed on {image}: {proc.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cjxl", required=True, type=Path)
    parser.add_argument("--eval", type=Path, default=Path("target/release/examples/eval"))
    parser.add_argument("-q", "--quality", type=float, action="append", help="repeatable; default: 50,75,90")
    parser.add_argument("-e", "--effort", type=int, action="append", help="repeatable; default: 7")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    qualities = args.quality or [50, 75, 90]
    efforts = args.effort or [7]
    print(f"{'effort':>6} {'quality':>7} | {'cjxl B':>9} {'ssim2':>6} | {'mini B':>9} {'ssim2':>6} | {'bytes':>7} {'ssim2':>6}")
    with tempfile.TemporaryDirectory() as tmp:
        for effort in efforts:
            for quality in qualities:
                ref = Path(tmp) / f"e{effort}_q{quality}"
                ref.mkdir()
                encode_reference(args.cjxl, args.images, quality, effort, ref)
                cb, cs = evaluate(args.eval, args.images, quality, effort, ref)
                mb, ms = evaluate(args.eval, args.images, quality, effort)
                print(
                    f"{effort:>6} {quality:>7g} | {cb:>9} {cs:>6.2f} | {mb:>9} {ms:>6.2f} | "
                    f"{(mb / cb - 1) * 100:>+6.1f}% {ms - cs:>+6.2f}"
                )


if __name__ == "__main__":
    main()
