import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import glob
import logging
import platform
import re
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import meshio
import numpy as np
from jax.scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator, interp1d

from jax_am.cfd.gamma import update_T as update_T_conduction
from jax_am.cfd.gamma_convection import update_T as update_T_convection
from jax_am.common import box_mesh


# -----------------------------------------------------------------------------
# Paths and experiment controls
# -----------------------------------------------------------------------------
GAMMA_INPUT = "/root/jax-am/applications/cfd/gamma/input"
CFD_SOLS = (
    "/root/autodl-tmp/jax_am_data/"
    "simple_gamma_match_data/vtk/cfd/sols"
)
RESULT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    "/root/autodl-tmp/jax_am_gamma/full_comparison"
)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60000"))
LOCAL_INTERVAL = 50
VTU_INTERVAL = 500

# CFD run used to generate the velocity snapshots
CFD_DT = 1e-6
CFD_WRITE_INTERVAL = 20
CFD_SPEED = 0.8
CFD_X0 = np.array([0.2e-3, 0.1e-3, 0.1e-3])
CFD_SHAPE = (464, 93, 46)
CFD_DOMAIN = (1e-3, 2e-4, 1e-4)


def numbered_key(path):
    match = re.search(r"u(\d+)\.vtu$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def latest_cfd_file():
    files = glob.glob(os.path.join(CFD_SOLS, "u*.vtu"))
    if not files:
        raise FileNotFoundError(f"No CFD VTU files found in {CFD_SOLS}")
    return max(files, key=numbered_key)


def setup_run_directories():
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULT_ROOT, run_id)
    paths = {
        "run": run_dir,
        "cond_vtk": os.path.join(run_dir, "conduction", "vtk"),
        "cond_numpy": os.path.join(run_dir, "conduction", "numpy"),
        "conv_vtk": os.path.join(run_dir, "convection", "vtk"),
        "conv_numpy": os.path.join(run_dir, "convection", "numpy"),
        "comparison": os.path.join(run_dir, "comparison"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def setup_logger(run_dir):
    logger = logging.getLogger("gamma_compare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        os.path.join(run_dir, "experiment_log.txt"), encoding="utf-8"
    )
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@jax.jit
def interp_data(T, rot, xl, yl, dx, dy):
    surface_T = (T[1:-1, 1:-1, -1] + T[1:-1, 1:-1, -2])/2
    gridx_ = np.arange(-1e-3, 0.28e-3, 1e-5)
    gridy_ = np.arange(-0.32e-3, 0.32e-3, 1e-5)
    gridx, gridy = np.meshgrid(gridx_, gridy_, indexing="ij")
    X_rot = (gridx*rot[0] - gridy*rot[1] + xl - dx/2)/dx
    Y_rot = (gridx*rot[1] + gridy*rot[0] + yl - dy/2)/dy
    melt_2d = map_coordinates(surface_T, (X_rot, Y_rot), 1)
    X_3d = jnp.repeat(X_rot[:, :, None], 30, axis=2)
    Y_3d = jnp.repeat(Y_rot[:, :, None], 30, axis=2)
    Z_3d = np.arange(30)[None, None, :].repeat(64, axis=1).repeat(128, axis=0)
    melt_3d = map_coordinates(
        T[1:-1, 1:-1, -31:-1], (X_3d, Y_3d, Z_3d), 1
    )
    return melt_2d, melt_3d


def load_velocity_template(path, spacing, logger):
    nx, ny, nz = CFD_SHAPE
    lx, ly, lz = CFD_DOMAIN
    frame = numbered_key(path)
    frame_time = frame*CFD_WRITE_INTERVAL*CFD_DT
    laser = CFD_X0 + np.array([CFD_SPEED*frame_time, 0.0, 0.0])

    mesh = meshio.read(path)
    velocity = np.asarray(
        mesh.cell_data["vel"][0], dtype=np.float32
    ).reshape(nx, ny, nz, 3)
    xc = (np.arange(nx) + 0.5)*lx/nx
    yc = (np.arange(ny) + 0.5)*ly/ny
    zc = (np.arange(nz) + 0.5)*lz/nz
    xq = np.arange(spacing/2, lx, spacing)
    yq = np.arange(spacing/2, ly, spacing)
    zq = np.arange(spacing/2, lz, spacing)
    X, Y, Z = np.meshgrid(xq, yq, zq, indexing="ij")
    query = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    sampled = np.empty((len(query), 3), dtype=np.float32)
    for component in range(3):
        interpolator = RegularGridInterpolator(
            (xc, yc, zc), velocity[..., component],
            method="linear", bounds_error=False, fill_value=0.0
        )
        sampled[:, component] = interpolator(query)

    local_coords = query - laser
    original_speed = np.linalg.norm(velocity, axis=3)
    sampled_speed = np.linalg.norm(sampled, axis=1)
    logger.info("CFD velocity file: %s", path)
    logger.info("CFD frame/time: u%03d / %.6f s", frame, frame_time)
    logger.info("CFD laser centre: %s m", laser.tolist())
    logger.info("Original CFD max speed: %.8f m/s", original_speed.max())
    logger.info("Resampled template: %d points", len(sampled))
    logger.info("Resampled max speed: %.8f m/s", sampled_speed.max())
    return local_coords.astype(np.float32), sampled


def melt_metrics(local_delta_T, T0, Ts, spacing):
    temperature = np.asarray(local_delta_T) + T0
    molten = np.argwhere(temperature >= Ts)
    result = {
        "Tmax_K": float(np.nanmax(temperature)),
        "molten_cells": int(len(molten)),
        "L_mm": 0.0,
        "W_mm": 0.0,
        "D_mm": 0.0,
    }
    if len(molten):
        result["L_mm"] = float((np.ptp(molten[:, 0]) + 1)*spacing*1e3)
        result["W_mm"] = float((np.ptp(molten[:, 1]) + 1)*spacing*1e3)
        if temperature.ndim == 3:
            result["D_mm"] = float((np.ptp(molten[:, 2]) + 1)*spacing*1e3)
    return result


def write_vtu(mesh, temperature, path):
    mesh.cell_data["T"] = [np.asarray(temperature, dtype=np.float32)]
    mesh.write(path, binary=True)


def simulation():
    paths = setup_run_directories()
    logger = setup_logger(paths["run"])
    start_time = time.time()

    nx, ny, nz = 500, 500, 100
    dx = dy = dz = 10e-6
    dt = 2e-6
    gamma_args = {
        "eta": 0.43, "r": 5e-5, "rho": 8440, "h": 10,
        "eps": 0.4, "SB": 5.67e-8, "T0": 298.,
        "Ts": 1563., "Tl": 1623.
    }
    gamma_args["L"] = 290e3/(gamma_args["Tl"]-gamma_args["Ts"])

    logger.info("===== Gamma conduction/convection comparison =====")
    logger.info("Host/platform: %s", platform.platform())
    logger.info("JAX version/devices: %s / %s", jax.__version__, jax.devices())
    logger.info("Run directory: %s", paths["run"])
    logger.info("Grid: %d x %d x %d", nx, ny, nz)
    logger.info("Spacing: dx=dy=dz=%.8g m", dx)
    logger.info("Time step: %.8g s", dt)
    logger.info("Requested steps: %d", MAX_STEPS)
    logger.info("Local/VTU save intervals: %d / %d steps", LOCAL_INTERVAL, VTU_INTERVAL)
    logger.info("Gamma parameters: %s", gamma_args)

    x_ = np.linspace(dx/2, dx/2 + dx*(nx-1), nx)
    y_ = np.linspace(dy/2, dy/2 + dy*(ny-1), ny)
    z_ = np.linspace(dz/2, dz/2 + dz*(nz-1), nz)
    x, y, z = jnp.meshgrid(x_, y_, z_, indexing="ij")
    initial = jnp.ones((nx+2, ny+2, nz+2), dtype=float)*gamma_args["T0"]
    T_cond = initial
    T_conv = initial

    toolpath = np.loadtxt(os.path.join(GAMMA_INPUT, "toolpath.txt"))
    t_all = np.arange(0, toolpath[-1, 0], dt) + dt/2
    num_steps = min(MAX_STEPS, len(t_all))
    t = t_all[:num_steps]
    direction = np.copy(toolpath[:, 0:3])
    direction[:, 1:3] = 0.0
    direction[1:, 1:3] = toolpath[1:, 1:3] - toolpath[:-1, 1:3]
    norm = np.linalg.norm(direction[1:, 1:3], axis=1)
    direction[1:, 1:3] /= norm[:, None]
    pos_interp = interp1d(toolpath[:, 0], toolpath[:, 1:3].T, kind="linear")
    dir_interp = interp1d(direction[:, 0], direction[:, 1:3].T, kind="next")
    power = np.load(os.path.join(GAMMA_INPUT, "power_0.npy"))
    P_interp = interp1d(power[:, 0], power[:, 1], kind="linear")
    pos, P, D = pos_interp(t), P_interp(t), dir_interp(t)
    weighted_power = np.trapezoid(power[:, 1], power[:, 0])/(power[-1, 0]-power[0, 0])
    logger.info("Toolpath duration/full steps: %.6f s / %d", toolpath[-1, 0], len(t_all))
    logger.info("Actual steps/end time: %d / %.6f s", num_steps, t[-1])
    logger.info("Power min/max/time-weighted mean: %.6f / %.6f / %.6f W",
                P.min(), P.max(), weighted_power)

    velocity_file = latest_cfd_file()
    coords_np, velocity_np = load_velocity_template(velocity_file, dx, logger)
    cfl = dt*np.max(np.sum(np.abs(velocity_np), axis=1))/dx
    logger.info("Advection CFL: %.8f", cfl)
    if cfl > 1.0:
        raise ValueError(f"Advection CFL={cfl:.4f} exceeds 1")
    velocity_coords = jnp.asarray(coords_np)
    velocity_template = jnp.asarray(velocity_np)

    mesh = box_mesh(nx, ny, nz, nx*dx, ny*dy, nz*dz)
    metric_path = os.path.join(paths["comparison"], "metrics.csv")
    metric_file = open(metric_path, "w", newline="", encoding="utf-8")
    fields = ["step", "time_s", "model", "Tmax_K", "molten_cells", "L_mm", "W_mm", "D_mm"]
    metric_writer = csv.DictWriter(metric_file, fieldnames=fields)
    metric_writer.writeheader()
    local_num = vtu_num = 0
    last_cond_metrics = last_conv_metrics = None

    try:
        for i in range(num_steps):
            T_cond = update_T_conduction(
                T_cond, pos[0, i], pos[1, i], x, y, P[i],
                dx, dy, dz, dt, gamma_args
            )
            T_conv = update_T_convection(
                T_conv, pos[0, i], pos[1, i], D[:, i], x, y, P[i],
                dx, dy, dz, dt, gamma_args,
                velocity_coords, velocity_template
            )

            save_local = (i-24) % LOCAL_INTERVAL == 0
            save_vtu = (i-24) % VTU_INTERVAL == 0 or i == num_steps-1

            if save_local:
                cond_2d, _ = interp_data(
                    T_cond-gamma_args["T0"], D[:, i], pos[0, i], pos[1, i], dx, dy
                )
                conv_2d, _ = interp_data(
                    T_conv-gamma_args["T0"], D[:, i], pos[0, i], pos[1, i], dx, dy
                )
                np.save(os.path.join(paths["cond_numpy"], f"melt_2d_step_{local_num:05d}.npy"),
                        np.asarray(jax.device_get(cond_2d), dtype=np.float32))
                np.save(os.path.join(paths["conv_numpy"], f"melt_2d_step_{local_num:05d}.npy"),
                        np.asarray(jax.device_get(conv_2d), dtype=np.float32))
                local_num += 1

            if save_vtu:
                cond_2d, cond_3d = interp_data(
                    T_cond-gamma_args["T0"], D[:, i], pos[0, i], pos[1, i], dx, dy
                )
                conv_2d, conv_3d = interp_data(
                    T_conv-gamma_args["T0"], D[:, i], pos[0, i], pos[1, i], dx, dy
                )
                cond_3d = np.asarray(jax.device_get(cond_3d), dtype=np.float32)
                conv_3d = np.asarray(jax.device_get(conv_3d), dtype=np.float32)
                last_cond_metrics = melt_metrics(cond_3d, gamma_args["T0"], gamma_args["Ts"], dx)
                last_conv_metrics = melt_metrics(conv_3d, gamma_args["T0"], gamma_args["Ts"], dx)
                for model, values in (("conduction", last_cond_metrics),
                                      ("convection", last_conv_metrics)):
                    metric_writer.writerow({"step": i, "time_s": float(t[i]),
                                            "model": model, **values})
                metric_file.flush()

                cond_full = np.asarray(jax.device_get(T_cond[1:-1, 1:-1, 1:-1]), dtype=np.float32)
                conv_full = np.asarray(jax.device_get(T_conv[1:-1, 1:-1, 1:-1]), dtype=np.float32)
                write_vtu(mesh, cond_full, os.path.join(paths["cond_vtk"], f"u_{vtu_num:05d}.vtu"))
                write_vtu(mesh, conv_full, os.path.join(paths["conv_vtk"], f"u_{vtu_num:05d}.vtu"))
                logger.info(
                    "Saved frame %d at step %d, t=%.6f s | "
                    "cond Tmax/L/W/D=%.2f/%.4f/%.4f/%.4f | "
                    "conv Tmax/L/W/D=%.2f/%.4f/%.4f/%.4f",
                    vtu_num, i, t[i],
                    last_cond_metrics["Tmax_K"], last_cond_metrics["L_mm"],
                    last_cond_metrics["W_mm"], last_cond_metrics["D_mm"],
                    last_conv_metrics["Tmax_K"], last_conv_metrics["L_mm"],
                    last_conv_metrics["W_mm"], last_conv_metrics["D_mm"]
                )
                vtu_num += 1

            if (i+1) % 100 == 0 or i == num_steps-1:
                logger.info("Progress: %d/%d (%.2f%%)", i+1, num_steps, 100*(i+1)/num_steps)
    finally:
        metric_file.close()

    elapsed = time.time() - start_time
    logger.info("===== Experiment completed =====")
    logger.info("Elapsed time: %.2f s (%.2f h)", elapsed, elapsed/3600)
    logger.info("Saved local/VTU frames: %d / %d", local_num, vtu_num)
    logger.info("Final conduction metrics: %s", last_cond_metrics)
    logger.info("Final convection metrics: %s", last_conv_metrics)
    if last_cond_metrics and last_conv_metrics:
        logger.info(
            "Final convection minus conduction: dTmax=%.4f K, dL=%.4f mm, "
            "dW=%.4f mm, dD=%.4f mm",
            last_conv_metrics["Tmax_K"]-last_cond_metrics["Tmax_K"],
            last_conv_metrics["L_mm"]-last_cond_metrics["L_mm"],
            last_conv_metrics["W_mm"]-last_cond_metrics["W_mm"],
            last_conv_metrics["D_mm"]-last_cond_metrics["D_mm"]
        )
    logger.info("Metrics CSV: %s", metric_path)
    logger.info("Run directory: %s", paths["run"])


if __name__ == "__main__":
    simulation()
