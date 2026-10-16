import json, hashlib
import glob
import os
import re
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import meshio
import numpy as np
from scipy.interpolate import interp1d

from jax_am.cfd.gamma_convection import update_T_from_velocity_frames
from jax_am.common import box_mesh

jax.config.update("jax_enable_x64", True)

DOMAIN = (0.75e-3, 0.75e-3, 0.10e-3)
GRID = (150, 150, 20)
DT = 0.5e-6
VELOCITY_INTERVAL = 100
WRITE_INTERVAL = 100
CFD_DIR = "/root/autodl-tmp/cfd/simple_180w/vtk/cfd/sols"
DATA_DIR = "/root/autodl-tmp/cfd/gamma_cfd_informed_180w"


def file_number(path):
    match = re.search(r"u_?(\d+)\.vtu$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def load_toolpath(case_dir):
    candidates = [os.path.join(case_dir, folder, "toolpath.txt")
                  for folder in ("Input", "input")]
    for filename in candidates:
        if os.path.isfile(filename):
            path = np.loadtxt(filename, ndmin=2)
            if path.shape[1] != 4:
                raise ValueError("toolpath.txt must contain: time, x, y, power")
            return filename, path
    raise FileNotFoundError(f"Cannot find toolpath in {candidates}")


def path_values(path, times):
    position = interp1d(path[:, 0], path[:, 1:3].T,
                        kind="linear")(times)
    index = np.searchsorted(path[:, 0], times, side="right") - 1
    power = path[np.clip(index, 0, len(path) - 1), 3]
    return position, power


def load_velocity_frames(folder, total_steps):
    """Direct reshape is used because VTU ordering is assumed correct."""
    files = sorted(glob.glob(os.path.join(folder, "u*.vtu")),
                   key=file_number)
    expected = total_steps // VELOCITY_INTERVAL + 1
    if len(files) != expected:
        raise ValueError(
            f"Expected {expected} CFD files, found {len(files)} in {folder}")

    frames = []
    for i, filename in enumerate(files):
        mesh = meshio.read(filename)
        if "vel" not in mesh.cell_data:
            raise KeyError(f"Field 'vel' is missing from {filename}")
        value = np.asarray(mesh.cell_data["vel"][0], dtype=np.float32)
        if value.shape != (np.prod(GRID), 3):
            raise ValueError(f"Unexpected velocity shape {value.shape}")
        frames.append(value.reshape(*GRID, 3))
        print(f"\rLoading CFD velocity: {i + 1}/{len(files)}",
              end="", flush=True)
    print()

    host = np.stack(frames, axis=0)
    if not np.all(np.isfinite(host)):
        raise ValueError("CFD velocity contains NaN or Inf")

    dx, dy, dz = (length / count
                  for length, count in zip(DOMAIN, GRID))
    speed = np.linalg.norm(host, axis=-1)
    cfl = DT * (np.abs(host[..., 0]) / dx
                + np.abs(host[..., 1]) / dy
                + np.abs(host[..., 2]) / dz)
    print(f"Velocity shape={host.shape}")
    print(f"Maximum speed={float(speed.max()):.6f} m/s; "
          f"advection CFL={float(cfl.max()):.6f}")
    if cfl.max() > 1.0:
        raise ValueError(f"Advection CFL={float(cfl.max()):.6f} exceeds 1")

    device = jax.device_put(host.astype(np.float64))
    jax.block_until_ready(device)
    return device


def clean_output(folder):
    os.makedirs(folder, exist_ok=True)
    for filename in glob.glob(os.path.join(folder, "*.vtu")):
        os.remove(filename)


def write_solution(mesh, T, folder, frame):
    values = np.asarray(jax.device_get(T[1:-1, 1:-1, 1:-1]),
                        dtype=np.float32).reshape(-1, 1)
    mesh.cell_data["T"] = [values]
    mesh.write(os.path.join(folder, f"u{frame:03d}.vtu"), binary=True)


MODEL = "CFD-informed Gamma"

def simulation():
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; detected {jax.devices()}")
    toolpath_file, path = load_toolpath(os.path.dirname(os.path.abspath(__file__)))
    steps = int(round(path[-1, 0]/DT))
    if steps < 2 or steps % WRITE_INTERVAL or not np.isclose(steps*DT, path[-1, 0]):
        raise ValueError("Duration must equal an integer multiple of 100 steps")
    nx, ny, nz = GRID
    dx, dy, dz = (length/n for length, n in zip(DOMAIN, GRID))
    axes = [(np.arange(n)+0.5)*d for n, d in zip(GRID, (dx, dy, dz))]
    x, y, _ = jnp.meshgrid(*axes, indexing="ij")
    args = dict(eta=.43, r=5e-5, rho=8440., h=10., eps=.4, SB=5.67e-8,
                T0=298., Ts=1563., Tl=1623., L=290e3/60.)
    T = jnp.full((nx+2, ny+2, nz+2), args["T0"], dtype=jnp.float64)
    times = (np.arange(steps)+.5)*DT
    positions, powers = path_values(path, times)
    positions, powers = jax.device_put((positions, powers))
    jax.block_until_ready((T, x, y, positions, powers))
    read_time = 0.
    start = time.perf_counter()
    frames = load_velocity_frames(CFD_DIR, steps)
    read_time = time.perf_counter()-start

    @jax.jit
    def advance(T, i, frames, positions, powers):
        return update_T_from_velocity_frames(
            T, frames, i, positions[0, i], positions[1, i], x, y, powers[i],
            dx, dy, dz, DT, args, velocity_interval=VELOCITY_INTERVAL)

    inputs = (frames, positions, powers)
    start = time.perf_counter()
    trial = advance(T, np.int32(0), *inputs)
    trial = advance(trial, np.int32(1), *inputs)
    jax.block_until_ready(trial)
    warmup = time.perf_counter()-start  # Includes compilation AND two executed steps.
    del trial                         # Formal run starts from the original T.
    vtk_dir = os.path.join(DATA_DIR, "vtk", "cfd", "sols")
    clean_output(vtk_dir)
    mesh = box_mesh(*GRID, *DOMAIN)
    start = time.perf_counter()
    write_solution(mesh, T, vtk_dir, 0)
    io_time = time.perf_counter()-start
    compute = 0.
    print(f"{MODEL}: {GRID}, {steps} steps; path={toolpath_file}", flush=True)
    for begin in range(0, steps, WRITE_INTERVAL):
        end = min(begin+WRITE_INTERVAL, steps)
        start = time.perf_counter()
        for i in range(begin, end):
            T = advance(T, np.int32(i), *inputs)
        jax.block_until_ready(T)
        compute += time.perf_counter()-start
        peak = float(jax.device_get(T.max()))
        if not np.isfinite(peak):
            raise FloatingPointError(f"Nonfinite temperature at step {end}")
        start = time.perf_counter()
        write_solution(mesh, T, vtk_dir, end//WRITE_INTERVAL)
        io_time += time.perf_counter()-start
        print(f"Step {end}/{steps}, Tmax={peak:.3f} K", flush=True)

    result = dict(model=MODEL, domain=DOMAIN, grid=GRID, dt=DT, steps=steps,
                  dtype=str(T.dtype), device=str(jax.devices()[0]),
                  path_sha256=hashlib.sha256(path.astype(np.float64).tobytes()).hexdigest(),
                  warmup_s=warmup, compute_s=compute, velocity_read_transfer_s=read_time,
                  compute_plus_velocity_s=compute+read_time, output_s=io_time,
                  final_Tmax_K=float(jax.device_get(T.max())))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "timing.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    simulation()
