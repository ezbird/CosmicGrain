#!/usr/bin/env python3
"""
make_music2_zoom_config_v2.py

Create the Halo 3886 MUSIC2 pilot zoom configuration from the exact parent
configuration while preserving the parent large-scale phases.

Changes relative to parent:
  - levelmax = 10
  - ref_center / ref_extent = traced Halo 3886 Lagrangian region
  - baryons = yes
  - add seed[10] for new level-10 small-scale modes
  - specialize output filename
  - remove parent-only / non-current setup keys such as region and ref_offset

Everything else is inherited verbatim.
"""

import argparse
import re
from pathlib import Path

DEFAULT_CENTER = (0.7702, 0.8228, 0.6546)
DEFAULT_EXTENT = (0.0376, 0.0665, 0.1173)
DEFAULT_SEED10 = 38861010


def get_section(text, section):
    pat = re.compile(rf'(?ms)^\[{re.escape(section)}\]\s*\n(?P<body>.*?)(?=^\[|\Z)')
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Missing [{section}] section")
    return m


def set_key(text, section, key, value):
    m = get_section(text, section)
    body = m.group("body")
    pat = re.compile(rf'(?m)^(\s*{re.escape(key)}\s*=).*$')
    if pat.search(body):
        body2 = pat.sub(rf'\1 {value}', body, count=1)
    else:
        if body and not body.endswith("\n"):
            body += "\n"
        body2 = body + f"{key} = {value}\n"
    return text[:m.start("body")] + body2 + text[m.end("body"):]


def remove_key(text, section, key):
    m = get_section(text, section)
    body = m.group("body")
    pat = re.compile(rf'(?m)^\s*{re.escape(key)}\s*=.*\n?')
    body2 = pat.sub("", body)
    return text[:m.start("body")] + body2 + text[m.end("body"):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_config")
    ap.add_argument("output_config")
    ap.add_argument("--center", nargs=3, type=float, default=DEFAULT_CENTER)
    ap.add_argument("--extent", nargs=3, type=float, default=DEFAULT_EXTENT)
    ap.add_argument("--seed10", type=int, default=DEFAULT_SEED10)
    ap.add_argument("--filename", default="IC_halo3886_level10")
    args = ap.parse_args()

    src = Path(args.parent_config)
    dst = Path(args.output_config)
    text = src.read_text()

    # Remove settings inherited from the uniform-parent config that would either
    # be misleading or conflict with the current MUSIC2 zoom-style setup.
    for key in ("region", "ref_offset"):
        text = remove_key(text, "setup", key)

    # These are not needed for this one-level nested pilot. Keep padding and
    # align_top because they are standard MUSIC2 zoom parameters.
    for key in ("overlap", "force_equal_extent"):
        text = remove_key(text, "setup", key)

    text = set_key(text, "setup", "levelmax", "10")
    text = set_key(
        text, "setup", "ref_center",
        ", ".join(f"{x:.8f}" for x in args.center)
    )
    text = set_key(
        text, "setup", "ref_extent",
        ", ".join(f"{x:.8f}" for x in args.extent)
    )
    text = set_key(text, "setup", "baryons", "yes")

    # Preserve seed[9] from the parent exactly. Add a deterministic independent
    # seed only for the newly introduced level-10 modes.
    text = set_key(text, "random", "seed[10]", str(args.seed10))

    text = set_key(text, "output", "filename", args.filename)

    header = """# ================================================================
# CosmicGrain Halo 3886 pilot MUSIC2 zoom
#
# Parent level-9 phases are inherited unchanged from the supplied parent
# configuration. seed[10] controls only the newly added level-10 modes.
#
# Traced z~99 region:
#   center = (38.51, 41.14, 32.73) Mpc/h
#   raw extent = (1.63, 2.89, 5.10) Mpc/h
#   padded ref_extent = (0.0376, 0.0665, 0.1173) box fractions
# ================================================================

"""

    dst.write_text(header + text)

    print(f"Wrote: {dst}")
    print("")
    print("Expected critical settings:")
    print("  levelmin    = inherited from parent")
    print("  levelmin_TF = inherited from parent")
    print("  levelmax    = 10")
    print("  ref_center  = " + ", ".join(f"{x:.8f}" for x in args.center))
    print("  ref_extent  = " + ", ".join(f"{x:.8f}" for x in args.extent))
    print("  baryons     = yes")
    print(f"  seed[10]    = {args.seed10}")
    print(f"  filename    = {args.filename}")
    print("")
    print("seed[9] is deliberately NOT changed.")


if __name__ == "__main__":
    main()
