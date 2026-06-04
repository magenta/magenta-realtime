#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare two .mlxfn files for byte-level parity.

MLX .mlxfn files contain a small amount of non-deterministic metadata
(timestamps, trace info) in the header/trailer, so byte-identical comparison
is too strict. This script reports the number of differing bytes and
classifies the result as PASS (metadata-only diffs, <0.01%) or FAIL.

Usage:
    python scripts/compare_mlxfn.py path/to/a.mlxfn path/to/b.mlxfn
    python scripts/compare_mlxfn.py path/to/dir_a/ path/to/dir_b/

When given directories, compares all matching .mlxfn and .safetensors files.
"""

import argparse
import sys
from pathlib import Path

# Byte diff threshold for metadata-only diffs (0.01% of file size).
# Empirically, MLX export metadata accounts for ~100K bytes in a ~2GB file.
METADATA_THRESHOLD_FRAC = 0.0001


def compare_files(path_a: Path, path_b: Path) -> bool:
    """Compare two files byte-by-byte. Returns True if parity holds."""
    if not path_a.exists():
        print(f"  ✗ {path_a.name}: not found at {path_a}")
        return False
    if not path_b.exists():
        print(f"  ✗ {path_b.name}: not found at {path_b}")
        return False

    data_a = path_a.read_bytes()
    data_b = path_b.read_bytes()

    if len(data_a) != len(data_b):
        print(f"  ✗ {path_a.name}: size mismatch ({len(data_a)} vs {len(data_b)})")
        return False

    if data_a == data_b:
        print(f"  ✓ {path_a.name}: byte-identical ({len(data_a):,} bytes)")
        return True

    # Count diffs
    diffs = sum(1 for a, b in zip(data_a, data_b) if a != b)
    frac = diffs / len(data_a)
    positions = [i for i, (a, b) in enumerate(zip(data_a, data_b)) if a != b]

    if frac < METADATA_THRESHOLD_FRAC:
        print(f"  ✓ {path_a.name}: {diffs:,}/{len(data_a):,} byte diffs "
              f"({100*frac:.4f}%) — metadata only "
              f"[byte {positions[0]}..{positions[-1]}]")
        return True
    else:
        print(f"  ✗ {path_a.name}: {diffs:,}/{len(data_a):,} byte diffs "
              f"({100*frac:.4f}%) — WEIGHT DATA DIFFERS "
              f"[byte {positions[0]}..{positions[-1]}]")
        return False


def compare_dirs(dir_a: Path, dir_b: Path) -> bool:
    """Compare matching .mlxfn and .safetensors files in two directories.

    Files are matched by extension type:
      - .mlxfn files compared pairwise (typically one per dir)
      - _state.safetensors files compared pairwise
      - _hessians.safetensors files ignored (different by design)
    """
    def _get_files(d):
        mlxfn = sorted(d.glob("*.mlxfn"))
        state = sorted(f for f in d.glob("*_state.safetensors"))
        return mlxfn, state

    mlxfn_a, state_a = _get_files(dir_a)
    mlxfn_b, state_b = _get_files(dir_b)

    pairs = []
    if mlxfn_a and mlxfn_b:
        pairs.append((mlxfn_a[0], mlxfn_b[0]))
    if state_a and state_b:
        pairs.append((state_a[0], state_b[0]))

    if not pairs:
        print(f"No matching files found between {dir_a.name}/ and {dir_b.name}/")
        return False

    all_pass = True
    for fa, fb in pairs:
        all_pass &= compare_files(fa, fb)
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Compare .mlxfn files for byte-level parity.")
    parser.add_argument("path_a", type=Path, help="First .mlxfn file or directory")
    parser.add_argument("path_b", type=Path, help="Second .mlxfn file or directory")
    args = parser.parse_args()

    print(f"Comparing:\n  A: {args.path_a}\n  B: {args.path_b}\n")

    if args.path_a.is_dir() and args.path_b.is_dir():
        passed = compare_dirs(args.path_a, args.path_b)
    elif args.path_a.is_file() and args.path_b.is_file():
        passed = compare_files(args.path_a, args.path_b)
    else:
        print("Error: both paths must be files or both must be directories.")
        sys.exit(1)

    print(f"\n{'PASS ✅' if passed else 'FAIL ❌'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
