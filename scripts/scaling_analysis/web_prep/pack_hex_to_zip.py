#!/usr/bin/env python3
import argparse
import os
import shutil
import zipfile


def main():
    ap = argparse.ArgumentParser(description="Zip per-city hex JSON files and optionally remove the folder")
    ap.add_argument("--hex_dir", default="../../web/public/webdata/hex", help="Path to hex folder")
    ap.add_argument("--zip_path", default="../../web/public/webdata/hex.zip", help="Output zip path")
    ap.add_argument("--remove", action="store_true", help="Remove the hex_dir after zipping")
    args = ap.parse_args()

    hex_dir = os.path.abspath(args.hex_dir)
    zip_path = os.path.abspath(args.zip_path)

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(hex_dir):
            for fn in files:
                if not fn.endswith('.json'):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, hex_dir)
                zf.write(full, arcname=rel)
    print(f"Wrote {zip_path}")

    if args.remove:
        shutil.rmtree(hex_dir)
        print(f"Removed {hex_dir}")

if __name__ == "__main__":
    raise SystemExit(main())

