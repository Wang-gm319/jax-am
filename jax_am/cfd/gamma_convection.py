from functools import partial

import jax
import jax.numpy as jnp

from jax_am.cfd.gamma import update_mat, BC_thermal


@jax.jit
def conduction_divergence(T, k, dx, dy, dz):
    Tc = T[1:-1, 1:-1, 1:-1]
    kc = k[1:-1, 1:-1, 1:-1]

    def contribution(Tn, kn, spacing):
        k_face = 2.0 * kc * kn / (kc + kn)
        return k_face * (Tn - Tc) / spacing**2

    kxm = k[:-2, 1:-1, 1:-1].at[0, :, :].set(kc[0, :, :])
    kxp = k[2:, 1:-1, 1:-1].at[-1, :, :].set(kc[-1, :, :])
    kym = k[1:-1, :-2, 1:-1].at[:, 0, :].set(kc[:, 0, :])
    kyp = k[1:-1, 2:, 1:-1].at[:, -1, :].set(kc[:, -1, :])
    kzm = k[1:-1, 1:-1, :-2].at[:, :, 0].set(kc[:, :, 0])
    kzp = k[1:-1, 1:-1, 2:].at[:, :, -1].set(kc[:, :, -1])

    return (
        contribution(T[:-2, 1:-1, 1:-1], kxm, dx)
        + contribution(T[2:, 1:-1, 1:-1], kxp, dx)
        + contribution(T[1:-1, :-2, 1:-1], kym, dy)
        + contribution(T[1:-1, 2:, 1:-1], kyp, dy)
        + contribution(T[1:-1, 1:-1, :-2], kzm, dz)
        + contribution(T[1:-1, 1:-1, 2:], kzp, dz)
    )


@jax.jit
def update_T_conduction(T, xl, yl, x, y, P, dx, dy, dz, dt, gamma_args):
    _, k, cp = update_mat(T, gamma_args)
    T = BC_thermal(T, k, xl, yl, x, y, P, dx, dy, dz, gamma_args)
    conduction = conduction_divergence(T, k, dx, dy, dz)
    capacity = gamma_args["rho"] * cp[1:-1, 1:-1, 1:-1]

    return T.at[1:-1, 1:-1, 1:-1].add(dt * conduction / capacity)


# Same calling signature as the original Gamma update_T.
update_T = update_T_conduction


@jax.jit
def update_T_full_velocity(T, velocity, xl, yl, x, y, P,
                           dx, dy, dz, dt, gamma_args):
    _, k, cp = update_mat(T, gamma_args)
    T = BC_thermal(T, k, xl, yl, x, y, P, dx, dy, dz, gamma_args)

    Tc = T[1:-1, 1:-1, 1:-1]
    u = velocity[..., 0]
    v = velocity[..., 1]
    w = velocity[..., 2]

    gx = jnp.where(u >= 0.0, Tc - T[:-2, 1:-1, 1:-1], T[2:, 1:-1, 1:-1] - Tc) / dx
    gy = jnp.where(v >= 0.0, Tc - T[1:-1, :-2, 1:-1], T[1:-1, 2:, 1:-1] - Tc) / dy
    gz = jnp.where(w >= 0.0, Tc - T[1:-1, 1:-1, :-2], T[1:-1, 1:-1, 2:] - Tc) / dz

    conduction = conduction_divergence(T, k, dx, dy, dz)
    capacity = gamma_args["rho"] * cp[1:-1, 1:-1, 1:-1]
    advection = u * gx + v * gy + w * gz

    return T.at[1:-1, 1:-1, 1:-1].add(dt * (conduction / capacity - advection))


@partial(jax.jit, static_argnames=("velocity_interval",))
def update_T_from_velocity_frames(T, frames, step, xl, yl, x, y, P, dx, dy, dz, dt, gamma_args, velocity_interval=100):
    position = (jnp.asarray(step, dtype=T.dtype) + 0.5) / velocity_interval
    lo = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, frames.shape[0] - 1)
    hi = jnp.minimum(lo + 1, frames.shape[0] - 1)
    weight = jnp.clip(position - lo, 0.0, 1.0)
    velocity = ((1.0 - weight) * frames[lo]+ weight * frames[hi])

    return update_T_full_velocity(T, velocity, xl, yl, x, y, P, dx, dy, dz, dt, gamma_args)


@partial(jax.jit, static_argnames=("velocity_interval",))
def advance_T_full_velocity(T, frames, steps, xls, yls, powers, x, y, dx, dy, dz, dt, gamma_args, velocity_interval=100):
    def body(state, inputs):
        step, xl, yl, power = inputs
        state = update_T_from_velocity_frames(
            state, frames, step, xl, yl, x, y, power,
            dx, dy, dz, dt, gamma_args,
            velocity_interval=velocity_interval)
        return state, None

    T, _ = jax.lax.scan(body, T, (steps, xls, yls, powers))
    return T
