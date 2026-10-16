import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_LOCATIONS = {
    "conduction": ["gamma_180w/example_180w.py", "example_180w.py"],
    "cfd": ["simple_multipath/multi_track_180w.py", "multi_track_180w.py"],
    "informed": ["gamma_180w/example_cfd_informed_180w.py",
                 "example_cfd_informed_180w.py"],
    "compare": ["compare_180w.py"],
    "view": ["view_180w.py"],
}


def find_script(root, explicit, key):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for relative in DEFAULT_LOCATIONS[key]:
        path = root / relative
        if path.is_file():
            return path.resolve()
    matches = list(root.rglob(DEFAULT_LOCATIONS[key][-1]))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {DEFAULT_LOCATIONS[key][-1]} below {root}")
    raise RuntimeError(
        f"Multiple {DEFAULT_LOCATIONS[key][-1]} files found; specify its path explicitly")


def run_stage(number, total, name, script, extra_args, log):
    command = [sys.executable, "-u", str(script), *extra_args]
    header = (f"\n{'=' * 72}\n[{number}/{total}] {name}\n"
              f"Script: {script}\nStart: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"Command: {' '.join(command)}\n{'=' * 72}\n")
    print(header, end="", flush=True)
    log.write(header); log.flush()
    start = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=str(script.parent), env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line); log.flush()
    return_code = process.wait()
    elapsed = time.perf_counter() - start
    footer = f"[{name}] return code={return_code}, wall time={elapsed:.3f} s\n"
    print(footer, flush=True); log.write(footer); log.flush()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the three 180 W simulations, timing comparison, and visualization in order.")
    parser.add_argument(
        "--root", default="/root/jax-am/applications/cfd",
        help="Directory containing the five scripts or their case subdirectories")
    parser.add_argument("--conduction", help="Path to example_180w.py")
    parser.add_argument("--cfd", help="Path to multi_track_180w.py")
    parser.add_argument("--informed", help="Path to example_cfd_informed_180w.py")
    parser.add_argument("--compare", help="Path to compare_180w.py")
    parser.add_argument("--view", help="Path to view_180w.py")
    parser.add_argument(
        "--log", default="/root/autodl-tmp/cfd/run_all_180w.log",
        help="Master console log")
    parser.add_argument("--fps", type=int, default=4, help="Visualization GIF frame rate")
    parser.add_argument("--no-gif", action="store_true", help="Generate static figures only")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    scripts = {
        key: find_script(root, getattr(args, key), key)
        for key in DEFAULT_LOCATIONS
    }
    log_path = Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    view_args = ["--fps", str(args.fps)]
    if args.no_gif:
        view_args.append("--no-gif")

    stages = [
        ("Heat-conduction simulation", scripts["conduction"], []),
        ("Full-CFD simulation and velocity output", scripts["cfd"], []),
        ("CFD-informed convection simulation", scripts["informed"], []),
        ("Three-model computation-time comparison", scripts["compare"], []),
        ("Three-model visualization", scripts["view"], view_args),
    ]

    overall_start = time.perf_counter()
    timings = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Python: {sys.executable}\nRoot: {root}\n")
        for key, path in scripts.items():
            log.write(f"{key}: {path}\n")
        for index, (name, script, extra) in enumerate(stages, 1):
            timings.append((name, run_stage(index, len(stages), name,
                                            script, extra, log)))
        total = time.perf_counter() - overall_start
        summary = ["\nAll stages completed successfully.",
                   *(f"{name}: {elapsed:.3f} s" for name, elapsed in timings),
                   f"Total workflow wall time: {total:.3f} s",
                   f"Master log: {log_path}"]
        text = "\n".join(summary) + "\n"
        print(text, end=""); log.write(text)


if __name__ == "__main__":
    main()
