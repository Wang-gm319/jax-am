import os
import glob
import numpy as np
import meshio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

T0, Ts, Tl = 298.0, 1563.0, 1623.0
nx, ny, nz = 500, 500, 100
dx = dy = dz = 10e-6

output_dir = "/root/autodl-tmp/jax_am_gamma/output_b"
numpy_dir = os.path.join(output_dir, "numpy")
vtk_dir = os.path.join(output_dir, "vtk")
figure_dir = os.path.join(output_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

x = (np.arange(nx) + 0.5) * dx * 1e3
y = (np.arange(ny) + 0.5) * dy * 1e3
z = (np.arange(nz) + 0.5) * dz * 1e3


def read_vtu_temperature(filename):
    mesh = meshio.read(filename)
    if "T" not in mesh.cell_data: raise KeyError(f"{filename} do not have T")
    temperature = np.asarray(mesh.cell_data["T"][0]).squeeze()
    if temperature.size != nx * ny * nz: raise ValueError(f"{filename} have {temperature.size} T data, expected {nx * ny * nz}")
    return temperature.reshape(nx, ny, nz)


def remove_contours(ax):
    for collection in list(ax.collections):
        collection.remove()


def draw_contours(ax, x_axis, y_axis, temperature):
    if np.nanmax(temperature) >= Ts:
        ax.contour(x_axis, y_axis, temperature.T, levels=[Ts], colors="cyan", linewidths=1)
    if np.nanmax(temperature) >= Tl:
        ax.contour(x_axis, y_axis, temperature.T, levels=[Tl], colors="lime", linewidths=1)


# ============================================================
# 1. T GIF near laser
# ============================================================
local_files = sorted(glob.glob(os.path.join(numpy_dir, "melt_2d_step_*.npy")))

if local_files:
    stride = max(1, len(local_files) // 300)
    selected_files = local_files[::stride]
    local_x = np.arange(-1e-3, 0.28e-3, 1e-5) * 1e3
    local_y = np.arange(-0.32e-3, 0.32e-3, 1e-5) * 1e3
    local_extent = [local_x.min(), local_x.max(), local_y.min(), local_y.max()]
    local_max = max(float(np.nanmax(np.load(file) + T0)) for file in selected_files)

    fig, ax = plt.subplots(figsize=(9, 5))
    first = np.load(selected_files[0]) + T0

    image = ax.imshow(first.T, origin="lower", extent=local_extent, cmap="inferno", vmin=T0, vmax=local_max, aspect="auto")
    fig.colorbar(image, ax=ax, label="Temperature (K)")
    ax.set_xlabel("x relative to laser (mm)")
    ax.set_ylabel("y relative to laser (mm)")

    writer = PillowWriter(fps=12)
    path = os.path.join(figure_dir, "local_temperature.gif")

    with writer.saving(fig, path, dpi=110):
        for index, file in enumerate(selected_files):
            temperature = np.load(file) + T0
            image.set_data(temperature.T)
            remove_contours(ax)
            draw_contours(ax, local_x, local_y, temperature)
            ax.set_title(f"Local temperature: {index + 1}/{len(selected_files)}\n Tmax = {temperature.max():.1f} K")
            writer.grab_frame()

    plt.close(fig)
    print(f"Generated {path}")


# ============================================================
# 2.Extract T from VTU files
# ============================================================
vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "u_*.vtu")))

if not vtk_files:
    raise FileNotFoundError(f"VTU is not found in {vtk_dir}")

print(f"Find {len(vtk_files)} VTU files")
print("Loading ...")

surfaces = []
maximum_temperatures = []
final_temperature = None

for index, file in enumerate(vtk_files):
    print(f"Loading {index + 1}/{len(vtk_files)}: {os.path.basename(file)}")
    temperature = read_vtu_temperature(file)
    surface = (temperature[:, :, -1] + temperature[:, :, -2]) / 2
    surfaces.append(surface.astype(np.float32))
    maximum_temperatures.append(float(np.nanmax(temperature)))
    if index == len(vtk_files) - 1:
        final_temperature = temperature

global_max = max(maximum_temperatures)
plate_extent = [x.min(), x.max(), y.min(), y.max()]


# ============================================================
# 3. Surface T GIF
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(surfaces[0].T, origin="lower", extent=plate_extent,cmap="inferno", vmin=T0, vmax=global_max, aspect="equal")
fig.colorbar(image, ax=ax, label="Temperature (K)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

writer = PillowWriter(fps=8)
path = os.path.join(figure_dir, "full_plate_temperature.gif")

with writer.saving(fig, path, dpi=110):
    for index, surface in enumerate(surfaces):
        image.set_data(surface.T)
        remove_contours(ax)
        draw_contours(ax, x, y, surface)
        ax.set_title(
            f"Whole-plate surface temperature\n"
            f"VTU {index + 1}/{len(surfaces)}, "
            f"Tmax = {maximum_temperatures[index]:.1f} K"
        )
        writer.grab_frame()

plt.close(fig)
print(f"Generated：{path}")


# ============================================================
# 4. Top view of the final plate temperature
# ============================================================
final_surface = surfaces[-1]

fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(final_surface.T, origin="lower", extent=plate_extent, cmap="inferno", vmin=T0, vmax=global_max, aspect="equal")
fig.colorbar(image, ax=ax, label="Temperature (K)")
draw_contours(ax, x, y, final_surface)

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(
    f"Final whole-plate surface temperature\n"
    f"Tmax = {maximum_temperatures[-1]:.1f} K"
)
fig.tight_layout()

path = os.path.join(figure_dir, "final_top_view.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"Generated：{path}")


# ============================================================
# 5. Longitudinal view
# ============================================================
_, peak_y, _ = np.unravel_index(np.nanargmax(final_temperature),final_temperature.shape)
section = final_temperature[:, peak_y, :]

fig, ax = plt.subplots(figsize=(10, 4))
image = ax.imshow(section.T, origin="lower", extent=[x.min(), x.max(), z.min(), z.max()], cmap="inferno", vmin=T0, vmax=global_max, aspect="auto")
fig.colorbar(image, ax=ax, label="Temperature (K)")
draw_contours(ax, x, z, section)

ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title(f"Final longitudinal section at y = {y[peak_y]:.3f} mm")
fig.tight_layout()

path = os.path.join(figure_dir, "final_longitudinal_view.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"Generated：{path}")


# ============================================================
# 6. Maximum temperature history
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(maximum_temperatures, color="red", marker="o", markersize=2)
ax.axhline(Ts, color="cyan", linestyle="--", label=f"Solidus {Ts:.0f} K")
ax.axhline(Tl, color="green", linestyle="--", label=f"Liquidus {Tl:.0f} K")
ax.set_xlabel("Saved VTU frame")
ax.set_ylabel("Maximum temperature (K)")
ax.set_title("Maximum temperature history")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

path = os.path.join(figure_dir, "maximum_temperature_history.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"Generated：{path}")


# ============================================================
# 7. Final melt pool scale
# ============================================================
melt_indices = np.argwhere(final_temperature >= Ts)

if len(melt_indices):
    length = (np.ptp(melt_indices[:, 0]) + 1) * dx * 1e3
    width = (np.ptp(melt_indices[:, 1]) + 1) * dy * 1e3
    depth = (np.ptp(melt_indices[:, 2]) + 1) * dz * 1e3

    print(f"Final melt pool length：{length:.4f} mm")
    print(f"Final melt pool width：{width:.4f} mm")
    print(f"Final melt pool depth：{depth:.4f} mm")
else:
    print("Final T field have not reached T_s~T_l region")

print(f"Visualization result saved：{figure_dir}")