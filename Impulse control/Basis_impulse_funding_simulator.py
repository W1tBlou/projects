"""
Impulse-control funding simulator with Monte Carlo Bellman/value iteration.

Main features
-------------
1. Precompute an impulse-control value table V(D, alpha_old) on D in [100, 10000].
2. Save/read the table as CSV for later simulations.
3. Rebalance using the intervention operator:

       max_alpha_new { V(D_pre_rebalance, alpha_new) - K(Q0,H0,Q1,H1,p1) }.

   Transaction costs are therefore subtracted in the intervention operator.
4. Recalculate P_L and P_U after every rebalance from relative boundaries.
5. Generate synthetic GBM prices and simulate portfolio paths with rebalances.
6. Visualize one portfolio trajectory with rebalance markers and alpha*(D).

Time unit: years.

Author: generated for impulse optimal-control funding strategy.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import erfc
from scipy.stats import norm

Array = NDArray[np.float64]
RebalanceReason = Literal["lower_yield", "upper_liq_prob"]


# =============================================================================
# Parameters
# =============================================================================


@dataclass(frozen=True)
class ModelParams:
    """Model, boundary, cost, and simulation parameters."""

    # GBM parameters. Time is measured in years.
    mu: float = -0.3
    sigma: float = 0.2

    # Funding and discount parameters.
    kappa: float = 0.015
    iota: float = 0.00375
    kappa_tilda: float = 0.545
    rho: float = 0.065
    rx: float = 0.0725

    # Boundary construction parameters.
    gamma: float = 0.05
    theta_F: float = 0.0100
    epsilon: float = 0.06
    h: float = 1.0

    # Initial market/account state.
    p0: float = 2300.0
    D0: float = 1000.0

    # Rebalancing costs.
    c_gas: float = 10.0
    c_spot: float = 5.0      # basis points
    c_fut: float = 5.0       # basis points

    # Numerical discretization.
    dt: float = 1.0 / 365.0
    simulation_horizon: float = 5.0

    # Position convention.
    # If True, after a rebalance the new positions are sized using D_after_cost.
    # The alpha choice itself is still made with K subtracted in the intervention
    # operator. Set False for a stricter convention-2 state reset.
    size_positions_after_cost: bool = True

    @property
    def zeta(self) -> float:
        denom = self.kappa + self.rho - self.rx
        if abs(denom) < 1e-14:
            raise ZeroDivisionError("kappa + rho - rx is too close to zero; zeta is undefined.")
        return (self.kappa - self.iota) / denom

    @property
    def nu(self) -> float:
        return self.mu / (self.sigma**2) - 0.5

    @property
    def funding_coeff(self) -> float:
        """Instantaneous funding coefficient iota + kappa * (zeta - 1)."""
        return self.iota + self.kappa * (self.zeta - 1.0)


def default_params() -> ModelParams:
    """Return the parameter set requested in the prompt."""
    return ModelParams()


# =============================================================================
# P_U and P_L formulas from the attached PuPl.txt
# =============================================================================


def z_upper_from_relative_upper(r_u: float, alpha0: float, params: ModelParams) -> float:
    """z(P_U) with r_u=P_U/p0: z = 1 / ((1-alpha0) r_u (1+theta_F))."""
    return 1.0 / ((1.0 - alpha0) * r_u * (1.0 + params.theta_F))


def prob_hit_upper_barrier(z: float, params: ModelParams, z0: float = 1.0) -> float:
    """
    Probability P(sup_{0<=s<=h} B_s >= z), where
    B_s = exp(sigma^2 * nu * s + sigma W_s).
    """
    if z <= 0.0 or z0 <= 0.0:
        return np.nan
    sigma = params.sigma
    t = params.h
    nu = params.nu
    a = np.log(z / z0) / (sigma * np.sqrt(2.0 * t))
    b = nu * sigma * np.sqrt(t) / np.sqrt(2.0)
    return float(0.5 * erfc(a - b) + 0.5 * (z / z0) ** (2.0 * nu) * erfc(a + b))


def solve_relative_upper_bound(alpha0: float, params: ModelParams) -> float:
    """Solve r_U=P_U/p0 from probability of futures liquidation = epsilon."""

    def f(r: float) -> float:
        z = z_upper_from_relative_upper(r, alpha0, params)
        return prob_hit_upper_barrier(z, params) - params.epsilon

    # Robust scan. In most calibrations r_U>1, but for some alpha and vol/drift
    # the crossing may be closer to 1 or even not bracketed by a small interval.
    grid = np.geomspace(0.05, 20.0, 800)
    vals = np.array([f(r) for r in grid], dtype=float)
    roots: List[float] = []
    for i in range(len(grid) - 1):
        if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(f, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-12)))

    if roots:
        # Prefer upper rebalancing boundary above the current price if available.
        above = [r for r in roots if r > 1.0]
        return float(min(above, key=lambda x: abs(x - 1.0)) if above else min(roots, key=lambda x: abs(x - 1.0)))

    # Calibration fallback: choose the relative price with probability closest to epsilon.
    idx = int(np.nanargmin(np.abs(vals)))
    return float(grid[idx])


def expected_B_no_liq(x: float, z: float, params: ModelParams) -> float:
    """E[B_t * 1_{no liquidation}] from the attached PuPl.txt formula."""
    if x <= 0.0 or z <= 0.0:
        return np.nan
    sigma = params.sigma
    t = params.h
    nu = params.nu
    denom = sigma * np.sqrt(t)
    term1 = norm.cdf((np.log(z / x) - (nu + 1.0) * sigma**2 * t) / denom)
    term2 = (z / x) ** (2.0 * nu + 2.0) * norm.cdf(
        (-np.log(z / x) - (nu + 1.0) * sigma**2 * t) / denom
    )
    return float(x * np.exp(params.mu * t) * (term1 - term2))


def expected_yield_proxy_at_lower(r_l: float, alpha0: float, r_u: float, params: ModelParams) -> float:
    """
    Expected annual-yield proxy used to define P_L.

    The attached derivation shows that D cancels and the boundary depends only
    on relative price movement. The formula below keeps the sign convention
    positive for the carry target gamma.
    """
    alpha_l = 1.0 - (1.0 - alpha0) * r_l
    z = z_upper_from_relative_upper(r_u, alpha0, params)
    e_b_no_liq = expected_B_no_liq(x=1.0, z=z, params=params)
    return float((1.0 - alpha_l) * abs(params.kappa_tilda * (params.zeta - 1.0)) * e_b_no_liq)


def solve_relative_lower_bound(alpha0: float, r_u: float, params: ModelParams) -> float:
    """Solve r_L=P_L/p0 from expected annual yield = gamma."""

    def f(r: float) -> float:
        return expected_yield_proxy_at_lower(r, alpha0, r_u, params) - params.gamma

    grid = np.linspace(0.05, 0.999999, 800)
    vals = np.array([f(r) for r in grid], dtype=float)
    for i in range(len(grid) - 1):
        if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        if vals[i] == 0.0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0.0:
            return float(brentq(f, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-12))

    # Fallback: closest achievable target under current calibration.
    idx = int(np.nanargmin(np.abs(vals)))
    return float(grid[idx])


def relative_bounds_for_alpha(alpha0: float, params: ModelParams) -> Tuple[float, float]:
    """Return relative bounds (r_L, r_U) for a given alpha."""
    r_u = solve_relative_upper_bound(alpha0, params)
    r_l = solve_relative_lower_bound(alpha0, r_u, params)
    return r_l, r_u


def absolute_bounds_after_rebalance(
    p_rebalance: float,
    alpha: float,
    params: ModelParams,
    bounds_cache: Optional[Dict[float, Tuple[float, float]]] = None,
) -> Tuple[float, float, float, float]:
    """Return P_L, P_U, r_L, r_U after a rebalance at p_rebalance."""
    if bounds_cache is not None and alpha in bounds_cache:
        r_l, r_u = bounds_cache[alpha]
    else:
        r_l, r_u = relative_bounds_for_alpha(alpha, params)
    return p_rebalance * r_l, p_rebalance * r_u, r_l, r_u


def precompute_relative_bounds_parallel(
    alpha_grid: Array,
    params: ModelParams,
    n_jobs: int = -1,
    backend: Literal["loky", "threading"] = "threading",
) -> Dict[float, Tuple[float, float]]:
    """Parallel precompute of relative bounds for every alpha in alpha_grid."""

    def one(a: float) -> Tuple[float, float, float]:
        r_l, r_u = relative_bounds_for_alpha(float(a), params)
        return float(a), float(r_l), float(r_u)

    rows = Parallel(n_jobs=n_jobs, backend=backend)(delayed(one)(a) for a in alpha_grid)
    return {a: (r_l, r_u) for a, r_l, r_u in rows}


# =============================================================================
# Positions, portfolio accounting, and costs
# =============================================================================


@dataclass
class PositionState:
    """State immediately after a rebalance."""

    t_rebalance: float
    p_rebalance: float
    D_rebalance: float
    alpha: float
    Q: float
    H: float
    P_L: float
    P_U: float
    r_L: float
    r_U: float


def positions_from_alpha(D: float, p: float, alpha: float, params: ModelParams) -> Tuple[float, float]:
    """Q=(1-alpha)D/p, H=Q/zeta."""
    Q = (1.0 - alpha) * D / p
    H = Q / params.zeta
    return float(Q), float(H)


def futures_margin_value(
    alpha0: float,
    D0: float,
    H: float,
    p_t: float,
    p0: float,
    integral_price: float,
    params: ModelParams,
    I_no_liq: float = 1.0,
) -> float:
    """
    Futures margin account value requested in the prompt:

        alpha0*D + H*zeta*(p_t-p_0)
        - H*kappa_tilda*(zeta-1)*int_0^t P_r dr,

    multiplied by I_no_liq. In this simulator I_no_liq defaults to 1, because
    P_U is used as a rebalance trigger for liquidation probability rather than
    as an actual exchange liquidation event.
    """
    val = (
        alpha0 * D0
        + H * params.zeta * (p_t - p0)
        - H * params.kappa_tilda * (params.zeta - 1.0) * integral_price
    )
    return float(I_no_liq * val)


def portfolio_value_from_state(
    pos: PositionState,
    p_t: float,
    integral_price_since_rebalance: float,
    params: ModelParams,
    I_no_liq: float = 1.0,
) -> float:
    """Spot value plus futures margin account value."""
    spot_value = pos.Q * p_t
    fut_value = futures_margin_value(
        alpha0=pos.alpha,
        D0=pos.D_rebalance,
        H=pos.H,
        p_t=p_t,
        p0=pos.p_rebalance,
        integral_price=integral_price_since_rebalance,
        params=params,
        I_no_liq=I_no_liq,
    )
    return float(spot_value + fut_value)


def transaction_cost(
    Q0: float,
    H0: float,
    Q1: float,
    H1: float,
    p1: float,
    params: ModelParams,
) -> float:
    """
    K = c_gas
        + c_spot/10000 * abs(Q1-Q0)*p1
        + c_fut/10000 * abs(H1-H0)*zeta*p1.
    """
    return float(
        params.c_gas
        + params.c_spot / 10000.0 * abs(Q1 - Q0) * p1
        + params.c_fut / 10000.0 * abs(H1 - H0) * params.zeta * p1
    )


# =============================================================================
# Synthetic prices
# =============================================================================


def generate_gbm_paths(
    params: ModelParams,
    n_paths: int = 1000,
    T: Optional[float] = None,
    dt: Optional[float] = None,
    p0: Optional[float] = None,
    seed: int = 777,
    n_jobs: int = -1,
    backend: Literal["loky", "threading"] = "threading",
) -> Tuple[Array, Array]:
    """Generate synthetic GBM price paths in parallel chunks."""
    T = params.simulation_horizon if T is None else T
    dt = params.dt if dt is None else dt
    p0 = params.p0 if p0 is None else p0
    n_steps = int(np.ceil(T / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Make chunks deterministic across job counts.
    n_chunks = max(1, min(abs(n_jobs) if n_jobs != -1 else 8, n_paths))
    chunk_sizes = [n_paths // n_chunks] * n_chunks
    for i in range(n_paths % n_chunks):
        chunk_sizes[i] += 1
    seeds = np.random.SeedSequence(seed).spawn(n_chunks)

    def one_chunk(size: int, ss: np.random.SeedSequence) -> Array:
        rng = np.random.default_rng(ss)
        z = rng.standard_normal((size, n_steps))
        log_ret = (params.mu - 0.5 * params.sigma**2) * dt + params.sigma * math.sqrt(dt) * z
        log_price = np.cumsum(log_ret, axis=1)
        prices = np.empty((size, n_steps + 1), dtype=float)
        prices[:, 0] = p0
        prices[:, 1:] = p0 * np.exp(log_price)
        return prices

    chunks = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(one_chunk)(sz, ss) for sz, ss in zip(chunk_sizes, seeds) if sz > 0
    )
    return times, np.vstack(chunks)


# =============================================================================
# MC one-episode simulation for Bellman value iteration
# =============================================================================


def simulate_episode_multipliers(
    alpha: float,
    r_bounds: Tuple[float, float],
    params: ModelParams,
    n_paths: int,
    seed: int,
) -> Dict[str, Array]:
    """
    Simulate one post-rebalance holding episode under fixed alpha.

    Output is normalized by D and p0 whenever possible. The position formula is
    the same one used later in full portfolio simulation.
    """
    rng = np.random.default_rng(seed)
    r_l, r_u = r_bounds
    n_steps = int(np.ceil(params.h / params.dt))
    dt = params.h / n_steps

    ratio = np.ones(n_paths, dtype=float)
    integral_ratio = np.zeros(n_paths, dtype=float)
    tau = np.full(n_paths, params.h, dtype=float)
    done = np.zeros(n_paths, dtype=bool)
    reason_code = np.zeros(n_paths, dtype=np.int8)  # -1 lower, +1 upper, 0 horizon

    drift = (params.mu - 0.5 * params.sigma**2) * dt
    vol = params.sigma * math.sqrt(dt)

    for step in range(1, n_steps + 1):
        active = ~done
        if not active.any():
            break
        prev = ratio[active].copy()
        new = prev * np.exp(drift + vol * rng.standard_normal(active.sum()))
        ratio[active] = new
        integral_ratio[active] += 0.5 * (prev + new) * dt

        lower_hit = new <= r_l
        upper_hit = new >= r_u
        horizon = step == n_steps
        hit = lower_hit | upper_hit | horizon
        if np.any(hit):
            active_idx = np.where(active)[0]
            idx = active_idx[hit]
            tau[idx] = step * dt
            done[idx] = True
            reason = np.zeros(hit.sum(), dtype=np.int8)
            reason[lower_hit[hit]] = -1
            reason[upper_hit[hit]] = +1
            reason_code[idx] = reason

    # Normalize with p0=1 and D=1.
    Q = 1.0 - alpha
    H = Q / params.zeta
    fut_value_mult = (
        alpha
        + H * params.zeta * (ratio - 1.0)
        - H * params.kappa_tilda * (params.zeta - 1.0) * integral_ratio
    )
    spot_value_mult = Q * ratio
    capital_mult = np.maximum(spot_value_mult + fut_value_mult, 1e-12)

    # Reward-only part for the Bellman running payoff. This is the carry cashflow
    # term implied by the futures-position formula.
    reward_mult = -H * params.kappa_tilda * (params.zeta - 1.0) * integral_ratio

    return {
        "capital_mult": capital_mult,
        "reward_mult": reward_mult,
        "exit_ratio": ratio,
        "tau": tau,
        "disc": np.exp(-params.rho * tau),
        "reason_code": reason_code.astype(float),
    }


# =============================================================================
# Value table and CSV persistence
# =============================================================================


@dataclass
class ValueTable:
    D_grid: Array
    alpha_grid: Array
    V: Array
    alpha_star_initial: Array
    bounds: Dict[float, Tuple[float, float]]
    params: ModelParams

    def value(self, D: float | Array, alpha: float) -> Array:
        """Interpolate V(D, alpha) in D for the nearest alpha-grid point."""
        j = int(np.argmin(np.abs(self.alpha_grid - alpha)))
        D_arr = np.asarray(D, dtype=float)
        val = np.interp(D_arr, self.D_grid, self.V[:, j], left=self.V[0, j], right=self.V[-1, j])
        return val

    def initial_alpha(self, D: float) -> float:
        vals = np.array([self.value(D, a).item() for a in self.alpha_grid])
        return float(self.alpha_grid[int(np.argmax(vals))])

    def alpha_from_saved_star(self, D: float) -> float:
        return float(np.interp(D, self.D_grid, self.alpha_star_initial))

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        data = {"D": self.D_grid, "alpha_star_initial": self.alpha_star_initial}
        for j, a in enumerate(self.alpha_grid):
            data[f"V_alpha_{a:.8f}"] = self.V[:, j]
        df = pd.DataFrame(data)
        # Store metadata as a JSON sidecar for robust reload.
        df.to_csv(path, index=False)
        meta = {
            "alpha_grid": self.alpha_grid.tolist(),
            "bounds": {f"{k:.12g}": [v[0], v[1]] for k, v in self.bounds.items()},
            "params": asdict(self.params),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_csv(cls, path: str | Path, params: Optional[ModelParams] = None) -> "ValueTable":
        path = Path(path)
        df = pd.read_csv(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata sidecar: {meta_path}")
        meta = json.loads(meta_path.read_text())
        alpha_grid = np.array(meta["alpha_grid"], dtype=float)
        V = np.column_stack([df[f"V_alpha_{a:.8f}"].to_numpy(float) for a in alpha_grid])
        bounds = {float(k): (float(v[0]), float(v[1])) for k, v in meta["bounds"].items()}
        if params is None:
            params = ModelParams(**meta["params"])
        return cls(
            D_grid=df["D"].to_numpy(float),
            alpha_grid=alpha_grid,
            V=V,
            alpha_star_initial=df["alpha_star_initial"].to_numpy(float),
            bounds=bounds,
            params=params,
        )


def _interp_value_for_alpha(V: Array, D_grid: Array, D_query: Array, alpha_idx: int) -> Array:
    return np.interp(D_query, D_grid, V[:, alpha_idx], left=V[0, alpha_idx], right=V[-1, alpha_idx])


def _transaction_cost_normalized(
    D0: float,
    D1: Array,
    r1: Array,
    alpha_old: float,
    alpha_new: float,
    params: ModelParams,
) -> Array:
    """Cost with normalized p0=1 and p1=r1 for value iteration."""
    Q0 = (1.0 - alpha_old) * D0
    H0 = Q0 / params.zeta
    Q1 = (1.0 - alpha_new) * D1 / r1
    H1 = Q1 / params.zeta
    return (
        params.c_gas
        + params.c_spot / 10000.0 * np.abs(Q1 - Q0) * r1
        + params.c_fut / 10000.0 * np.abs(H1 - H0) * params.zeta * r1
    )


def _bellman_update_for_old_alpha(
    j_old: int,
    V: Array,
    D_grid: Array,
    alpha_grid: Array,
    alpha_old: float,
    episode: Dict[str, Array],
    params: ModelParams,
) -> Tuple[int, Array, Array]:
    """One Bellman update block for a fixed old alpha."""
    nD = len(D_grid)
    nA = len(alpha_grid)
    n_paths = len(episode["capital_mult"])

    cap_mult = episode["capital_mult"]
    reward_mult = episode["reward_mult"]
    r_exit = episode["exit_ratio"]
    disc = episode["disc"]

    V_col = np.empty(nD, dtype=float)
    policy_col = np.empty(nD, dtype=int)

    for iD, D0 in enumerate(D_grid):
        D1 = np.maximum(D0 * cap_mult, 1e-12)
        reward = D0 * reward_mult
        candidates = np.empty((n_paths, nA), dtype=float)

        for j_new, alpha_new in enumerate(alpha_grid):
            continuation = _interp_value_for_alpha(V, D_grid, D1, j_new)
            K = _transaction_cost_normalized(D0, D1, r_exit, float(alpha_old), float(alpha_new), params)
            candidates[:, j_new] = continuation - K

        best_idx = np.argmax(candidates, axis=1)
        best_val = candidates[np.arange(n_paths), best_idx]
        V_col[iD] = float(np.mean(reward + disc * best_val))
        policy_col[iD] = int(np.argmax(np.bincount(best_idx, minlength=nA)))

    return j_old, V_col, policy_col


def precompute_value_table(
    params: ModelParams,
    D_grid: Optional[Array] = None,
    alpha_grid: Optional[Array] = None,
    n_paths: int = 2000,
    max_iter: int = 100,
    tol: float = 1e-4,
    relaxation: float = 0.65,
    seed: int = 123,
    n_jobs: int = -1,
    backend: Literal["loky", "threading"] = "threading",
    csv_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> ValueTable:
    """
    Precompute V(D, alpha_old) and alpha*(D), then optionally save to CSV.

    This is the expensive step. It uses MC paths for one continuation episode and
    then value iteration with K subtracted inside the intervention operator.
    """
    if D_grid is None:
        D_grid = np.geomspace(100.0, 10_000.0, 60)
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.95, 31)

    D_grid = np.asarray(D_grid, dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    nD, nA = len(D_grid), len(alpha_grid)

    if verbose:
        print(f"zeta = {params.zeta:.8f}")
        print(f"Precomputing relative P_L/P_U for {nA} alpha values...")
    t0 = time.perf_counter()
    bounds = precompute_relative_bounds_parallel(alpha_grid, params, n_jobs=n_jobs, backend=backend)

    if verbose:
        print(f"Bounds done in {time.perf_counter() - t0:.2f}s")
        print(f"Simulating one-episode MC blocks: n_paths={n_paths}, backend={backend}, n_jobs={n_jobs}")

    seed_seq = np.random.SeedSequence(seed).spawn(nA)
    episode_items = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(simulate_episode_multipliers)(
            alpha=float(a),
            r_bounds=bounds[float(a)],
            params=params,
            n_paths=n_paths,
            seed=int(ss.generate_state(1)[0]),
        )
        for a, ss in zip(alpha_grid, seed_seq)
    )
    episodes = {float(a): ep for a, ep in zip(alpha_grid, episode_items)}

    V = np.zeros((nD, nA), dtype=float)
    policy_idx = np.zeros((nD, nA), dtype=int)

    if verbose:
        print("Starting Bellman value iteration...")
    prev_err = np.inf
    for it in range(1, max_iter + 1):
        iter_start = time.perf_counter()
        blocks = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_bellman_update_for_old_alpha)(
                j_old=j,
                V=V,
                D_grid=D_grid,
                alpha_grid=alpha_grid,
                alpha_old=float(a),
                episode=episodes[float(a)],
                params=params,
            )
            for j, a in enumerate(alpha_grid)
        )

        V_new = np.empty_like(V)
        policy_new = np.empty_like(policy_idx)
        for j, V_col, pol_col in blocks:
            V_new[:, j] = V_col
            policy_new[:, j] = pol_col

        err = float(np.max(np.abs(V_new - V)))
        V = (1.0 - relaxation) * V + relaxation * V_new
        policy_idx = policy_new

        if verbose:
            print(
                f"iter={it:03d} | error={err:.6e} | "
                f"real_time={time.perf_counter() - iter_start:.2f}s | "
                f"ratio={err / prev_err if np.isfinite(prev_err) and prev_err > 0 else np.nan:.3f}"
            )
        if err < tol:
            if verbose:
                print(f"Converged: error {err:.6e} < tol {tol:.1e}")
            break
        prev_err = err

    alpha_star_initial = np.empty(nD, dtype=float)
    for i, D in enumerate(D_grid):
        vals = np.array([_interp_value_for_alpha(V, D_grid, np.array([D]), j)[0] for j in range(nA)])
        alpha_star_initial[i] = float(alpha_grid[int(np.argmax(vals))])

    table = ValueTable(
        D_grid=D_grid,
        alpha_grid=alpha_grid,
        V=V,
        alpha_star_initial=alpha_star_initial,
        bounds=bounds,
        params=params,
    )
    if csv_path is not None:
        table.to_csv(csv_path)
        if verbose:
            print(f"Saved value table: {csv_path}")
            print(f"Saved metadata: {Path(csv_path).with_suffix(Path(csv_path).suffix + '.meta.json')}")
    return table


# =============================================================================
# Rebalancing function and portfolio simulation
# =============================================================================


def choose_alpha_with_intervention_operator(
    D_pre: float,
    p1: float,
    Q0: float,
    H0: float,
    value_table: ValueTable,
    params: ModelParams,
) -> Tuple[float, float, float, float, float]:
    """
    Choose alpha_new by maximizing V(D_pre, alpha_new) - K.

    Returns: alpha_new, score, K, Q1_pre_cost, H1_pre_cost.
    """
    best = (-np.inf, np.nan, np.nan, np.nan, np.nan)  # score, alpha, K, Q1, H1
    for alpha_new in value_table.alpha_grid:
        Q1, H1 = positions_from_alpha(D_pre, p1, float(alpha_new), params)
        K = transaction_cost(Q0, H0, Q1, H1, p1, params)
        score = float(value_table.value(D_pre, float(alpha_new))) - K
        if score > best[0]:
            best = (score, float(alpha_new), K, Q1, H1)
    score, alpha, K, Q1, H1 = best
    return alpha, score, K, Q1, H1


def rebalance_portfolio(
    t: float,
    p1: float,
    D_pre: float,
    old_position: PositionState,
    value_table: ValueTable,
    params: ModelParams,
    reason: RebalanceReason,
) -> Tuple[PositionState, Dict[str, float | str]]:
    """
    Rebalance function requested in the prompt.

    It returns a new PositionState with new Q, H, D, optimal alpha, and new
    P_L/P_U bounds. The alpha choice uses the intervention operator, i.e. K is
    subtracted in the score. The realized account D is then reduced by K.
    """
    alpha_new, score, K, Q1_pre_cost, H1_pre_cost = choose_alpha_with_intervention_operator(
        D_pre=D_pre,
        p1=p1,
        Q0=old_position.Q,
        H0=old_position.H,
        value_table=value_table,
        params=params,
    )

    D_after_cost = max(D_pre - K, 1e-12)
    if params.size_positions_after_cost:
        Q1, H1 = positions_from_alpha(D_after_cost, p1, alpha_new, params)
    else:
        Q1, H1 = Q1_pre_cost, H1_pre_cost

    P_L, P_U, r_L, r_U = absolute_bounds_after_rebalance(
        p_rebalance=p1,
        alpha=alpha_new,
        params=params,
        bounds_cache=value_table.bounds,
    )
    new_pos = PositionState(
        t_rebalance=t,
        p_rebalance=p1,
        D_rebalance=D_after_cost,
        alpha=alpha_new,
        Q=Q1,
        H=H1,
        P_L=P_L,
        P_U=P_U,
        r_L=r_L,
        r_U=r_U,
    )
    event = {
        "time": t,
        "price": p1,
        "reason": reason,
        "D_pre": D_pre,
        "D_after": D_after_cost,
        "alpha_new": alpha_new,
        "K": K,
        "score": score,
        "Q_new": Q1,
        "H_new": H1,
        "P_L_new": P_L,
        "P_U_new": P_U,
    }
    return new_pos, event


def initial_position(
    D0: float,
    p0: float,
    value_table: ValueTable,
    params: ModelParams,
    alpha0: Optional[float] = None,
) -> PositionState:
    """Open initial position using alpha0 or alpha*(D0) from the value table."""
    alpha = value_table.initial_alpha(D0) if alpha0 is None else float(alpha0)
    Q, H = positions_from_alpha(D0, p0, alpha, params)
    P_L, P_U, r_L, r_U = absolute_bounds_after_rebalance(p0, alpha, params, value_table.bounds)
    return PositionState(
        t_rebalance=0.0,
        p_rebalance=p0,
        D_rebalance=D0,
        alpha=alpha,
        Q=Q,
        H=H,
        P_L=P_L,
        P_U=P_U,
        r_L=r_L,
        r_U=r_U,
    )


def simulate_portfolio_on_price_path(
    times: Array,
    prices: Array,
    value_table: ValueTable,
    params: ModelParams,
    D0: Optional[float] = None,
    alpha0: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run portfolio with rebalances on one synthetic price path."""
    D0 = params.D0 if D0 is None else D0
    pos = initial_position(D0=D0, p0=float(prices[0]), value_table=value_table, params=params, alpha0=alpha0)

    integral_since_rebalance = 0.0
    rows: List[Dict[str, float]] = []
    events: List[Dict[str, float | str]] = []

    current_value = D0
    rows.append(
        {
            "time": float(times[0]),
            "price": float(prices[0]),
            "portfolio_value": current_value,
            "return": 0.0,
            "alpha": pos.alpha,
            "Q": pos.Q,
            "H": pos.H,
            "P_L": pos.P_L,
            "P_U": pos.P_U,
            "is_rebalance": 0.0,
        }
    )

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        p_prev = float(prices[i - 1])
        p = float(prices[i])
        t = float(times[i])
        integral_since_rebalance += 0.5 * (p_prev + p) * dt

        D_pre = portfolio_value_from_state(pos, p, integral_since_rebalance, params, I_no_liq=1.0)
        current_value = D_pre
        rebalance_flag = 0.0

        reason: Optional[RebalanceReason] = None
        if p <= pos.P_L:
            reason = "lower_yield"
        elif p >= pos.P_U:
            reason = "upper_liq_prob"

        if reason is not None:
            new_pos, event = rebalance_portfolio(
                t=t,
                p1=p,
                D_pre=D_pre,
                old_position=pos,
                value_table=value_table,
                params=params,
                reason=reason,
            )
            events.append(event)
            pos = new_pos
            integral_since_rebalance = 0.0
            current_value = pos.D_rebalance
            rebalance_flag = 1.0

        rows.append(
            {
                "time": t,
                "price": p,
                "portfolio_value": current_value,
                "return": current_value / D0 - 1.0,
                "alpha": pos.alpha,
                "Q": pos.Q,
                "H": pos.H,
                "P_L": pos.P_L,
                "P_U": pos.P_U,
                "is_rebalance": rebalance_flag,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(events)


def simulate_many_portfolios(
    times: Array,
    price_paths: Array,
    value_table: ValueTable,
    params: ModelParams,
    D0: Optional[float] = None,
    alpha0: Optional[float] = None,
    n_jobs: int = -1,
    backend: Literal["loky", "threading"] = "threading",
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]:
    """Run strategy on many price paths in parallel."""

    def one(i: int) -> Tuple[int, pd.DataFrame, pd.DataFrame, float, int]:
        path_df, events_df = simulate_portfolio_on_price_path(
            times=times,
            prices=price_paths[i],
            value_table=value_table,
            params=params,
            D0=D0,
            alpha0=alpha0,
        )
        final_value = float(path_df["portfolio_value"].iloc[-1])
        initial_value = (params.D0 if D0 is None else D0)
        final_return = final_value / initial_value - 1.0
        return i, path_df, events_df, final_return, len(events_df)

    out = Parallel(n_jobs=n_jobs, backend=backend)(delayed(one)(i) for i in range(price_paths.shape[0]))
    out = sorted(out, key=lambda x: x[0])
    paths = [x[1] for x in out]
    events = [x[2] for x in out]
    summary = pd.DataFrame(
        {
            "path_id": [x[0] for x in out],
            "final_return": [x[3] for x in out],
            "n_rebalances": [x[4] for x in out],
        }
    )
    return summary, paths, events


# =============================================================================
# Visualization
# =============================================================================


def plot_single_path_with_rebalances(
    path_df: pd.DataFrame,
    events_df: pd.DataFrame,
    title: str = "Portfolio and price path with rebalances",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot portfolio value and price trajectory.

    Squares mark P_L/yield rebalances. Triangles mark P_U/liquidation-probability rebalances.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(path_df["time"], path_df["portfolio_value"], label="Portfolio value")
    ax1.set_xlabel("Time, years")
    ax1.set_ylabel("Portfolio value")

    ax2 = ax1.twinx()
    ax2.plot(path_df["time"], path_df["price"], linestyle="--", label="Price")
    ax2.plot(path_df["time"], path_df["P_L"], linestyle=":", label="P_L")
    ax2.plot(path_df["time"], path_df["P_U"], linestyle=":", label="P_U")
    ax2.set_ylabel("Price")

    if not events_df.empty:
        for reason, marker, label in [
            ("lower_yield", "s", "Rebalance by yield / P_L"),
            ("upper_liq_prob", "^", "Rebalance by liquidation probability / P_U"),
        ]:
            sub = events_df[events_df["reason"] == reason]
            if not sub.empty:
                y = np.interp(sub["time"].to_numpy(float), path_df["time"], path_df["portfolio_value"])
                ax1.scatter(sub["time"], y, marker=marker, s=70, label=label)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    return fig, ax1


def plot_alpha_vs_D(value_table: ValueTable) -> Tuple[plt.Figure, plt.Axes]:
    """Plot dependence of optimal initial alpha on D."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(value_table.D_grid, value_table.alpha_star_initial, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("Optimal alpha")
    ax.set_title("Optimal alpha as a function of D")
    fig.tight_layout()
    return fig, ax


def plot_return_histogram(summary_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot final-return distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(summary_df["final_return"], bins=40)
    ax.set_xlabel("Final return")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of simulated portfolio returns")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# End-to-end runner
# =============================================================================


def run_end_to_end_example(
    value_table_csv: str | Path = "impulse_value_table.csv",
    force_recompute_table: bool = False,
    n_paths_table: int = 2000,
    n_paths_sim: int = 1000,
    n_jobs: int = -1,
    backend: Literal["loky", "threading"] = "threading",
    show_plots: bool = True,
) -> Dict[str, object]:
    """Run table precompute, simulation, summary, and visualizations."""
    params = default_params()
    csv_path = Path(value_table_csv)

    D_grid = np.geomspace(100.0, 10_000.0, 60)
    alpha_grid = np.linspace(0.05, 0.95, 31)

    if force_recompute_table or not csv_path.exists() or not csv_path.with_suffix(csv_path.suffix + ".meta.json").exists():
        value_table = precompute_value_table(
            params=params,
            D_grid=D_grid,
            alpha_grid=alpha_grid,
            n_paths=n_paths_table,
            max_iter=100,
            tol=1e-4,
            relaxation=0.65,
            seed=123,
            n_jobs=n_jobs,
            backend=backend,
            csv_path=csv_path,
            verbose=True,
        )
    else:
        value_table = ValueTable.from_csv(csv_path, params=params)
        print(f"Loaded value table from {csv_path}")

    print(f"Initial optimal alpha at D={params.D0:.2f}: {value_table.initial_alpha(params.D0):.6f}")

    print(f"Generating {n_paths_sim} synthetic price paths...")
    times, price_paths = generate_gbm_paths(
        params=params,
        n_paths=n_paths_sim,
        T=params.simulation_horizon,
        dt=params.dt,
        p0=params.p0,
        seed=777,
        n_jobs=n_jobs,
        backend=backend,
    )

    print("Running portfolio simulations...")
    t0 = time.perf_counter()
    summary, path_dfs, event_dfs = simulate_many_portfolios(
        times=times,
        price_paths=price_paths,
        value_table=value_table,
        params=params,
        D0=params.D0,
        alpha0=None,
        n_jobs=n_jobs,
        backend=backend,
    )
    print(f"Simulation done in {time.perf_counter() - t0:.2f}s")
    print(f"Mean final return: {summary['final_return'].mean():.6f}")
    print(f"Variance of final returns: {summary['final_return'].var(ddof=1):.6f}")
    print(f"Mean number of rebalances: {summary['n_rebalances'].mean():.3f}")

    figures = {}
    if show_plots:
        figures["path"] = plot_single_path_with_rebalances(path_dfs[0], event_dfs[0])
        figures["alpha"] = plot_alpha_vs_D(value_table)
        figures["returns"] = plot_return_histogram(summary)
        plt.show()

    return {
        "params": params,
        "value_table": value_table,
        "times": times,
        "price_paths": price_paths,
        "summary": summary,
        "path_dfs": path_dfs,
        "event_dfs": event_dfs,
        "figures": figures,
    }


if __name__ == "__main__":
    run_end_to_end_example(
        value_table_csv="impulse_value_table.csv",
        force_recompute_table=True,
        n_paths_table=2000,
        n_paths_sim=1000,
        n_jobs=-1,
        backend="threading",
        show_plots=True,
    )
