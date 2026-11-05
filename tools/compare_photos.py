#!/usr/bin/env python3
"""Compare two prebuilt `eval` examples; no Python dependencies required.

Encoder time is measured inside eval, while GNU time measures peak process RSS.
Fidelity is measured in separate processes so decoding/SSIMULACRA2 allocations
never contaminate the encoder-only memory measurement. Inputs are not downloaded.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ENCODE = re.compile(r": (\d+) B ([\d.]+) ms hash=([0-9a-f]+)")
FIDELITY = re.compile(r"\bpsnr\s+([\d.]+)\s+ssim2\s+(-?[\d.]+)")
RSS = re.compile(r"^PHOTO_RSS_KIB (\d+)$", re.MULTILINE)


def invoke(binary, image, quality, effort, threads, encode_only):
    command = [str(binary), "-q", str(quality), "-e", str(effort), "-t", str(threads)]
    if encode_only:
        command += ["--encode-only"]
    command += [str(image)]
    if encode_only:
        command = ["/usr/bin/time", "-f", "PHOTO_RSS_KIB %M", *command]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"command failed: {command!r}\n{proc.stdout}\n{proc.stderr}")
    if encode_only:
        match = ENCODE.search(proc.stdout)
        rss = RSS.search(proc.stderr)
        if match is None or rss is None:
            raise RuntimeError(f"unrecognized encoder output: {proc.stdout}\n{proc.stderr}")
        size, ms, digest = match.groups()
        return dict(bytes=int(size), ms=float(ms), hash=digest, rss_kib=int(rss[1]))
    match = FIDELITY.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"unrecognized fidelity output: {proc.stdout}")
    return dict(psnr=float(match[1]), ssim2=float(match[2]))


def match_size(binary, image, quality, effort, threads, target, tolerance, steps):
    """Bracket all lossy qualities, retaining the closest measured byte count.

    Transform/EPF transitions can make size nonmonotonic. Report the actual
    residual size error instead of claiming a guaranteed exact match.
    """
    low, high = 0.0, 99.999
    best = None
    q = quality
    for _ in range(steps + 1):
        measured = invoke(binary, image, q, effort, threads, True)
        error = abs(measured["bytes"] / target - 1)
        if best is None or error < best[0]:
            best = (error, q, measured)
        if error <= tolerance:
            break
        if measured["bytes"] < target:
            low = q
        else:
            high = q
        q = (low + high) / 2
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--quality", type=float, action="append", help="repeatable; default: 75,90,95")
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--threads", type=int, action="append", help="repeatable; default: 1")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--match-size", action="store_true", help="also evaluate near-equal-size fidelity")
    parser.add_argument("--tolerance", type=float, default=0.002, help="relative size error target")
    parser.add_argument("--steps", type=int, default=10, help="maximum size-matching bisection steps")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    qualities = args.quality or [75.0, 90.0, 95.0]
    threads = args.threads or [1]
    if args.runs < 1 or args.steps < 0 or not 1 <= args.effort <= 10:
        parser.error("require runs >= 1, steps >= 0 and effort in 1..10")
    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance < 1:
        parser.error("tolerance must be finite and in 0..1")
    if any(not math.isfinite(q) or not 0 <= q < 100 for q in qualities):
        parser.error("quality must be finite and in 0..100 (lossy comparisons only)")
    if any(t < 1 for t in threads):
        parser.error("threads must be >= 1")
    binaries = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    images = [p.resolve(strict=True) for p in args.images]
    report = dict(binaries={k: str(v) for k, v in binaries.items()}, effort=args.effort, cases=[])
    for quality in qualities:
        for thread_count in threads:
            for image in images:
                case = dict(image=str(image), quality=quality, threads=thread_count, runs={k: [] for k in binaries})
                for run in range(args.runs):
                    # Alternate execution order to reduce thermal/order bias.
                    order = list(binaries) if run % 2 == 0 else list(reversed(binaries))
                    for name in order:
                        case["runs"][name].append(invoke(binaries[name], image, quality, args.effort, thread_count, True))
                case["median"] = {}
                for name, values in case["runs"].items():
                    if len({(v["bytes"], v["hash"]) for v in values}) != 1:
                        raise RuntimeError(f"nondeterministic output: {name}, {image}")
                    case["median"][name] = dict(
                        bytes=values[0]["bytes"], hash=values[0]["hash"],
                        ms=statistics.median(v["ms"] for v in values),
                        rss_kib=statistics.median(v["rss_kib"] for v in values),
                    )
                base, new = (case["median"][k] for k in binaries)
                case["same_output"] = base["hash"] == new["hash"] and base["bytes"] == new["bytes"]
                case["time_change_pct"] = (new["ms"] / base["ms"] - 1) * 100
                case["rss_change_pct"] = (new["rss_kib"] / base["rss_kib"] - 1) * 100
                if args.match_size:
                    error, q, measured = match_size(binaries["candidate"], image, quality, args.effort, thread_count, base["bytes"], args.tolerance, args.steps)
                    reference = invoke(binaries["baseline"], image, quality, args.effort, thread_count, False)
                    candidate = invoke(binaries["candidate"], image, q, args.effort, thread_count, False)
                    case["matched"] = dict(quality=q, bytes=measured["bytes"], size_error_pct=(measured["bytes"] / base["bytes"] - 1) * 100,
                                           within_tolerance=error <= args.tolerance, baseline=reference, candidate=candidate,
                                           ssim2_delta=candidate["ssim2"] - reference["ssim2"])
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{image.name} q{quality:g} t{thread_count}: time {case['time_change_pct']:+.1f}%, RSS {case['rss_change_pct']:+.1f}%, same output={case['same_output']}", file=sys.stderr)


if __name__ == "__main__":
    main()
