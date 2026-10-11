"""Command line for the SDK: new | validate | pack | install | export | list.

    python -m swarm.sdk new my_tactic --type tactic             # -> brains/my_tactic/
    python -m swarm.sdk validate brains/my_tactic               # exit 1 if any check fails
    python -m swarm.sdk pack brains/my_tactic -o my_tactic.zip  # for the studio's ARMORY upload
    python -m swarm.sdk install my_tactic.zip                   # same as uploading it in the studio
    python -m swarm.sdk export brains/my_brain --out deploy/my_brain
    python -m swarm.sdk list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import export_for_device
from .install import LIBRARY, install_folder, install_upload, pack
from .packages import LIBRARY_DIRS, PackageError, discover_packages
from .templates import KINDS, new_package
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m swarm.sdk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="create a package skeleton")
    n.add_argument("name")
    n.add_argument("--type", choices=KINDS, required=True)
    n.add_argument("--dir", type=Path, default=LIBRARY, help="parent folder (default: brains/)")
    n.add_argument("--title", default="")
    n.add_argument("--author", default="")
    v = sub.add_parser("validate", help="run the conformance checks")
    v.add_argument("path", type=Path)
    v.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    p = sub.add_parser("pack", help="zip a package for upload")
    p.add_argument("path", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    i = sub.add_parser("install", help="install a .zip / .swm / .py / folder into brains/")
    i.add_argument("source", type=Path)
    i.add_argument("--replace", action="store_true")
    e = sub.add_parser("export", help="device artefacts for a perception brain")
    e.add_argument("path", type=Path)
    e.add_argument("--out", type=Path, required=True, help="new (or empty) folder")
    e.add_argument("--target", default="linux-aarch64")
    sub.add_parser("list", help="packages in " + ", ".join(str(d) for d in LIBRARY_DIRS))
    args = ap.parse_args(argv)

    try:
        if args.cmd == "new":
            root = new_package(args.dir, args.name, args.type, args.title, args.author)
            print(f"created {root}")
            if args.type == "perception-swm":
                print(f"next: python {root / 'train.py'}   (compiles model.swm), then validate")
            else:
                print(f"next: edit it, then python -m swarm.sdk validate {root}")
            return 0
        if args.cmd == "validate":
            rep = validate(args.path)
            print(rep.render_text())
            if args.json:
                args.json.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
            return 0 if rep.ok else 1
        if args.cmd == "pack":
            print(f"wrote {pack(args.path, args.out)}")
            return 0
        if args.cmd == "install":
            src = args.source
            pkg = install_folder(src, replace=args.replace) if src.is_dir() else \
                install_upload(src.read_bytes(), src.name, replace=args.replace)
            print(f"installed {pkg.label()} ({pkg.type}) -> {pkg.root}")
            print(f"next: python -m swarm.sdk validate {pkg.root}")
            return 0
        if args.cmd == "export":
            res = export_for_device(args.path, args.out, args.target)
            print(res["instructions"])
            print(f"artefacts in {res['out']} (see DEPLOY.txt)")
            return 0
        if args.cmd == "list":
            for folder, pkg, err in discover_packages():
                if pkg is None:
                    print(f"BROKEN      {folder}: {err}")
                    continue
                what = pkg.type + (f"/{pkg.format}" if pkg.type == "perception" else f" -> strategy {pkg.strategy}")
                print(f"{pkg.label():<28} {what:<34} {folder}")
            return 0
    except (PackageError, OSError) as exc:
        ap.exit(1, f"swarm.sdk {args.cmd}: {exc}\n")
    return 0
