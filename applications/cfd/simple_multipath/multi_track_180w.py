import json, hashlib
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import numpy as np

from jax_am.cfd.cfd_am import AM_3d, mesh3d
from jax_am.common import box_mesh

jax.config.update("jax_enable_x64", True)

DOMAIN = (0.75e-3, 0.75e-3, 0.10e-3)
GRID = (150, 150, 20)                 # same 5 um cells as Gamma
DT = 0.5e-6
DATA_DIR = "/root/autodl-tmp/cfd/simple_180w"
WRITE_INTERVAL = 100
CHECK_INTERVAL = 100


class MultiTrackAM3D(AM_3d):
    def __init__(self, args, path):
        self.path = jnp.asarray(path)
        args["t_OFF"] = float(path[-1, 0])
        super().__init__(args)

    def toolpath(self, t):
        xl = jnp.interp(t, self.path[:, 0], self.path[:, 1])
        yl = jnp.interp(t, self.path[:, 0], self.path[:, 2])
        index = jnp.searchsorted(self.path[:, 0], t, side="right")-1
        index = jnp.clip(index, 0, self.path.shape[0]-1)
        power = self.path[index, 3]
        return xl, yl, self.args["X0"][2], power


def load_toolpath(case_dir):
    candidates = [
        os.path.join(case_dir, "..", "gamma_180w", "Input", "toolpath.txt"),
        os.path.join(case_dir, "..", "gamma_180w", "input", "toolpath.txt"),
        os.path.join(case_dir, "toolpath.txt"),
    ]
    for filename in candidates:
        if os.path.isfile(filename):
            path = np.loadtxt(filename, ndmin=2)
            if path.shape[1] != 4:
                raise ValueError("toolpath.txt must contain: time, x, y, power")
            return filename, path
    raise FileNotFoundError(f"Cannot find toolpath in {candidates}")


def build_solver(path):
    nx, ny, nz = GRID
    lx, ly, lz = DOMAIN
    mesh = mesh3d(list(DOMAIN), list(GRID))
    local_grid = GRID
    local_domain = DOMAIN
    mesh_local = mesh3d(list(local_domain), list(local_grid))
    args = {
        "rho": 8440.0, "visco": 7e-3, "Marangoni": -2e-5,
        "dt": DT, "speed": 0.8, "X0": [0.1e-3, 0.15e-3, 0.1e-3],
        "rb": 5e-5, "eta": 0.43, "P": 180.0, "T_ref": 298.0,
        "Ts": 1563.0, "Tl": 1623.0, "latent_heat": 290e3,
        "heat_source": 0, "h": 10.0, "stefan_boltzmann": 5.67e-8,
        "emissivity": 0.4, "write_sol_interval": WRITE_INTERVAL,
        "check_sol_interval": CHECK_INTERVAL, "mesh": mesh,
        "mesh_local": mesh_local,
        "meshio_mesh": box_mesh(nx, ny, nz, lx, ly, lz),
        "data_dir": DATA_DIR,
    }
    args["cp"] = lambda T: 0.2441*jnp.clip(T, 300.0, args["Tl"])+338.39
    args["k"] = lambda T: 0.0163*jnp.clip(T, 300.0, args["Tl"])+4.5847
    return MultiTrackAM3D(args, path), args


MODEL = "Full CFD"

def simulation():
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; detected {jax.devices()}")
    toolpath_file, path = load_toolpath(os.path.dirname(os.path.abspath(__file__)))
    steps = int(round(path[-1, 0]/DT))
    if steps < 2 or steps % WRITE_INTERVAL or not np.isclose(steps*DT, path[-1, 0]):
        raise ValueError("Duration must equal an integer multiple of 100 steps")
    solver, args = build_solver(path)
    fields = ("T", "vel", "grad_p0", "conv", "conv_T", "solidID", "t")
    initial = {key: getattr(solver, key) for key in fields}

    def sync():
        jax.block_until_ready(tuple(getattr(solver, key) for key in fields))

    sync()
    start = time.perf_counter()
    for _ in range(2):
        solver.time_integration()
    sync()
    warmup = time.perf_counter()-start
    for key, value in initial.items():
        setattr(solver, key, value)
    del initial
    sync()
    start = time.perf_counter()
    solver.write_sols(0)
    io_time = time.perf_counter()-start
    compute = read_time = 0.
    print(f"{MODEL}: {GRID}, {steps} steps; path={toolpath_file}", flush=True)
    for begin in range(0, steps, WRITE_INTERVAL):
        end = min(begin+WRITE_INTERVAL, steps)
        start = time.perf_counter()
        for _ in range(begin, end):
            solver.time_integration()
        sync()
        compute += time.perf_counter()-start
        peak = float(jax.device_get(solver.T.max()))
        if not np.isfinite(peak):
            raise FloatingPointError(f"Nonfinite temperature at step {end}")
        start = time.perf_counter()
        solver.write_sols(end)
        io_time += time.perf_counter()-start
        print(f"Step {end}/{steps}, Tmax={peak:.3f} K", flush=True)
    T = solver.T

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
