"""Simplified fixed-wing aircraft dynamics and model-based trajectory optimization.

Provides a nominal point-mass aircraft model and a trajectory optimizer that
quantifies how physically plausible a reconstructed flight trajectory is by
solving for optimal control inputs subject to the dynamics model.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize
from scipy import sparse

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

from .feature_mapping import PhysicsFeatureMap


# Aircraft parameter presets
AIRCRAFT_PRESETS = {
    "cessna172s": {
        "max_roll": 60.0,
        "max_pitch": 30.0,
        "max_roll_rate": 15.0,
        "max_pitch_rate": 10.0,
        "max_accel": 5.0,
        "max_fuel_flow": 0.005,  # gal/s, ~18 gal/hr
    },
    "pa28": {
        "max_roll": 60.0,
        "max_pitch": 25.0,
        "max_roll_rate": 12.0,
        "max_pitch_rate": 8.0,
        "max_accel": 4.0,
        "max_fuel_flow": 0.004,  # ~14.4 gal/hr
    },
    "pa44": {
        "max_roll": 45.0,
        "max_pitch": 20.0,
        "max_roll_rate": 10.0,
        "max_pitch_rate": 6.0,
        "max_accel": 3.5,
        "max_fuel_flow": 0.008,  # twin engine, ~28.8 gal/hr
    },
}

# Default state weights: inverse-square-range normalization so each
# state contributes roughly equally to the loss regardless of units.
DEFAULT_STATE_WEIGHTS_6 = np.array([
    1.0 / 60.0 ** 2,    # roll: typical range ~120 deg
    1.0 / 30.0 ** 2,    # pitch: typical range ~60 deg
    1.0 / 5000.0 ** 2,  # altitude: typical range ~10000 ft
    1.0 / 180.0 ** 2,   # heading: typical range ~360 deg
    1.0 / 100.0 ** 2,   # airspeed: typical range ~160 kts
    1.0 / 50.0 ** 2,    # fuel: typical range ~60 gal
])
DEFAULT_STATE_WEIGHTS_5 = DEFAULT_STATE_WEIGHTS_6[:5]


class AircraftDynamics:
    """Simplified fixed-wing aircraft dynamics model.

    State vector: [roll(deg), pitch(deg), altitude(ft), heading(deg),
                   airspeed(kts), fuel(gal)]
    Control vector: [roll_rate(deg/s), pitch_rate(deg/s), accel(kts/s),
                     fuel_flow(gal/s)]

    Fuel state and control are optional (controlled by has_fuel flag).
    """

    # Constants
    G = 32.174        # gravity, ft/s^2
    KTS_TO_FPS = 1.6878  # knots to ft/s

    def __init__(
        self,
        dt: float = 1.0,
        has_fuel: bool = True,
        min_airspeed: float = 40.0,
        max_roll: float = 60.0,
        max_pitch: float = 30.0,
        max_roll_rate: float = 15.0,
        max_pitch_rate: float = 10.0,
        max_accel: float = 5.0,
        max_fuel_flow: float = 0.005,
    ):
        self.dt = dt
        self.has_fuel = has_fuel
        self.min_airspeed = min_airspeed
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.max_roll_rate = max_roll_rate
        self.max_pitch_rate = max_pitch_rate
        self.max_accel = max_accel
        self.max_fuel_flow = max_fuel_flow

    @property
    def num_states(self) -> int:
        return 6 if self.has_fuel else 5

    @property
    def num_controls(self) -> int:
        return 4 if self.has_fuel else 3

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Euler-integrate one timestep.

        Args:
            state: [roll, pitch, alt, hdg, airspeed, (fuel)].
            control: [u_roll, u_pitch, u_accel, (u_fuel)].

        Returns:
            Next state array.
        """
        roll, pitch, alt, hdg, airspeed = state[0], state[1], state[2], state[3], state[4]
        u_roll, u_pitch, u_accel = control[0], control[1], control[2]

        # Clamp airspeed to avoid division by zero in coordinated turn
        V_safe = max(airspeed, self.min_airspeed)
        V_fps = V_safe * self.KTS_TO_FPS

        # Roll dynamics
        roll_new = roll + u_roll * self.dt
        # Clamp to avoid tan singularity
        roll_new = np.clip(roll_new, -89.0, 89.0)

        # Pitch dynamics
        pitch_new = pitch + u_pitch * self.dt

        # Altitude: h_dot = V * sin(pitch)
        alt_new = alt + V_fps * np.sin(np.radians(pitch)) * self.dt

        # Heading: coordinated turn psi_dot = (g / V) * tan(roll)
        roll_rad = np.radians(roll)
        hdg_rate_dps = (self.G / V_fps) * np.tan(roll_rad) * (180.0 / np.pi)
        hdg_new = (hdg + hdg_rate_dps * self.dt) % 360.0

        # Airspeed
        airspeed_new = airspeed + u_accel * self.dt

        next_state = np.array([roll_new, pitch_new, alt_new, hdg_new, airspeed_new])

        if self.has_fuel and len(state) == 6:
            u_fuel = control[3] if len(control) >= 4 else 0.0
            fuel_new = max(state[5] - u_fuel * self.dt, 0.0)
            next_state = np.append(next_state, fuel_new)

        return next_state

    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Simulate N steps from initial state.

        Args:
            x0: Initial state, shape (num_states,).
            controls: Control sequence, shape (N, num_controls).

        Returns:
            Trajectory including initial state, shape (N+1, num_states).
        """
        N = controls.shape[0]
        trajectory = np.zeros((N + 1, x0.shape[0]))
        trajectory[0] = x0.copy()
        for t in range(N):
            trajectory[t + 1] = self.step(trajectory[t], controls[t])
        return trajectory

    def control_bounds(self, N: int) -> List[Tuple[float, float]]:
        """Build scipy bounds list for N timesteps of controls."""
        bounds = []
        for _ in range(N):
            bounds.append((-self.max_roll_rate, self.max_roll_rate))
            bounds.append((-self.max_pitch_rate, self.max_pitch_rate))
            bounds.append((-self.max_accel, self.max_accel))
            if self.has_fuel:
                bounds.append((0.0, self.max_fuel_flow))
        return bounds

    def linearize(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around state x0 with zero control.

        Returns A, B matrices for: x_{t+1} = A @ x_t + B @ u_t
        """
        num_states = self.num_states
        num_controls = self.num_controls

        # Numerical linearization via finite differences
        eps = 1e-5

        A = np.zeros((num_states, num_states))
        B = np.zeros((num_states, num_controls))

        u0 = np.zeros(num_controls)
        f0 = self.step(x0, u0)

        # Compute A (df/dx)
        for i in range(num_states):
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = self.step(x_plus, u0)
            A[:, i] = (f_plus - f0) / eps

        # Compute B (df/du)
        for i in range(num_controls):
            u_plus = u0.copy()
            u_plus[i] += eps
            f_plus = self.step(x0, u_plus)
            B[:, i] = (f_plus - f0) / eps

        return A, B

    def get_control_bounds_arrays(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bound arrays for N timesteps."""
        num_controls = self.num_controls
        lower = np.zeros(N * num_controls)
        upper = np.zeros(N * num_controls)

        for t in range(N):
            idx = t * num_controls
            lower[idx] = -self.max_roll_rate
            upper[idx] = self.max_roll_rate
            lower[idx + 1] = -self.max_pitch_rate
            upper[idx + 1] = self.max_pitch_rate
            lower[idx + 2] = -self.max_accel
            upper[idx + 2] = self.max_accel
            if self.has_fuel:
                lower[idx + 3] = 0.0
                upper[idx + 3] = self.max_fuel_flow

        return lower, upper


def _heading_diff(diff: np.ndarray) -> np.ndarray:
    """Wrap heading differences to [-180, 180]."""
    return (diff + 180.0) % 360.0 - 180.0


def trajectory_physics_baseline(
    dynamics: AircraftDynamics,
    x0: np.ndarray,
    x_reconstructed: np.ndarray,
    state_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fast physics plausibility metric using zero-control baseline.

    Simply rolls out the dynamics with zero control and measures how far
    the reconstruction deviates from the physically natural trajectory.
    Very fast (O(N)) - no optimization needed.

    Args:
        dynamics: AircraftDynamics instance.
        x0: Initial state.
        x_reconstructed: Target trajectory, shape (N, num_states).
        state_weights: Per-state weighting.

    Returns:
        Dict with physics_loss and related metrics.
    """
    N = x_reconstructed.shape[0]
    num_states = x0.shape[0]
    num_controls = dynamics.num_controls

    if state_weights is None:
        if num_states == 6:
            state_weights = DEFAULT_STATE_WEIGHTS_6.copy()
        else:
            state_weights = DEFAULT_STATE_WEIGHTS_5.copy()

    HDG_IDX = 3

    # Zero-control rollout
    zero_controls = np.zeros((N, num_controls))
    trajectory = dynamics.rollout(x0, zero_controls)

    state_diff = trajectory[1:] - x_reconstructed
    state_diff[:, HDG_IDX] = _heading_diff(state_diff[:, HDG_IDX])

    per_state_mse = np.mean(state_diff ** 2, axis=0)
    state_tracking = float(np.sum(state_weights * per_state_mse))

    return {
        "physics_loss": state_tracking,
        "state_tracking_loss": state_tracking,
        "control_penalty": 0.0,
        "optimal_controls": zero_controls,
        "physics_trajectory": trajectory,
        "per_state_mse": per_state_mse,
        "success": True,
        "optimizer_message": "zero-control baseline",
    }


def trajectory_optimization_loss(
    dynamics: AircraftDynamics,
    x0: np.ndarray,
    x_reconstructed: np.ndarray,
    lambda_control: float = 0.01,
    state_weights: Optional[np.ndarray] = None,
    max_iter: int = 20,
    method: str = "baseline",
    ftol: float = 1e-3,
    gtol: float = 1e-3,
) -> Dict[str, Any]:
    """Compute physics loss via model-based trajectory optimization.

    Measures how physically plausible a reconstructed trajectory is by
    comparing it to what the dynamics model predicts.

    Args:
        dynamics: AircraftDynamics instance.
        x0: Initial state (last true state before masked region).
        x_reconstructed: Reconstructed trajectory for masked region, shape (N, num_states).
        lambda_control: Control regularization weight.
        state_weights: Per-state weighting; defaults to inverse-square-range normalization.
        max_iter: Maximum optimizer iterations (for optimization methods).
        method: 'baseline' (fastest), 'trf' (least_squares), or 'L-BFGS-B'.
        ftol: Tolerance for termination.
        gtol: Tolerance for termination.

    Returns:
        Dict with physics_loss, state_tracking_loss, control_penalty,
        optimal_controls, physics_trajectory, per_state_mse, success.
    """
    # Fast baseline: zero-control rollout comparison
    if method == "baseline":
        return trajectory_physics_baseline(
            dynamics, x0, x_reconstructed, state_weights
        )
    N = x_reconstructed.shape[0]
    num_states = x0.shape[0]
    num_controls = dynamics.num_controls

    if state_weights is None:
        if num_states == 6:
            state_weights = DEFAULT_STATE_WEIGHTS_6.copy()
        else:
            state_weights = DEFAULT_STATE_WEIGHTS_5.copy()

    # Sqrt of weights for residual formulation
    sqrt_state_weights = np.sqrt(state_weights)
    sqrt_lambda = np.sqrt(lambda_control)

    HDG_IDX = 3  # heading is always index 3 in the physics state

    def residuals(u_flat: np.ndarray) -> np.ndarray:
        """Return residuals for least_squares (faster than scalar objective)."""
        controls = u_flat.reshape(N, num_controls)
        trajectory = dynamics.rollout(x0, controls)
        state_diff = trajectory[1:] - x_reconstructed

        # Handle heading wraparound
        state_diff[:, HDG_IDX] = _heading_diff(state_diff[:, HDG_IDX])

        # Weight the state residuals
        weighted_state_residuals = (state_diff * sqrt_state_weights).flatten()

        # Control regularization residuals
        control_residuals = (sqrt_lambda * controls).flatten()

        return np.concatenate([weighted_state_residuals, control_residuals])

    # Initial guess: zero controls
    u0 = np.zeros(N * num_controls)

    # Build bounds for least_squares (lower, upper arrays)
    bounds_list = dynamics.control_bounds(N)
    lower_bounds = np.array([b[0] for b in bounds_list])
    upper_bounds = np.array([b[1] for b in bounds_list])

    result = scipy.optimize.least_squares(
        residuals,
        u0,
        method=method,
        bounds=(lower_bounds, upper_bounds),
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_iter * 10,  # least_squares counts function evals differently
        verbose=0,
    )

    # Extract final results
    optimal_controls = result.x.reshape(N, num_controls)
    trajectory = dynamics.rollout(x0, optimal_controls)

    state_diff = trajectory[1:] - x_reconstructed
    state_diff[:, HDG_IDX] = _heading_diff(state_diff[:, HDG_IDX])

    per_state_mse = np.mean(state_diff ** 2, axis=0)
    state_tracking = float(np.sum(state_weights * per_state_mse))
    control_pen = float(lambda_control * np.mean(optimal_controls ** 2))

    # least_squares returns status > 0 for success
    success = result.status > 0

    return {
        "physics_loss": state_tracking + control_pen,
        "state_tracking_loss": state_tracking,
        "control_penalty": control_pen,
        "optimal_controls": optimal_controls,
        "physics_trajectory": trajectory,
        "per_state_mse": per_state_mse,
        "success": bool(success),
        "optimizer_message": result.message,
    }


def find_contiguous_masked_regions(
    mask_1d: np.ndarray,
) -> List[Tuple[int, int]]:
    """Find contiguous regions where mask is False (masked).

    Args:
        mask_1d: 1D boolean array, True=keep, False=masked.

    Returns:
        List of (start_idx, end_idx) tuples.
    """
    regions = []
    in_region = False
    start = 0
    for i in range(len(mask_1d)):
        if not mask_1d[i] and not in_region:
            start = i
            in_region = True
        elif mask_1d[i] and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(mask_1d)))
    return regions


def physics_trajectory_optimization(
    original_sequence: np.ndarray,
    reconstructed_sequence: np.ndarray,
    mask: np.ndarray,
    feature_map: PhysicsFeatureMap,
    dynamics: AircraftDynamics,
    lambda_control: float = 0.01,
    min_segment_length: int = 5,
    max_iter: int = 50,
    state_weights: Optional[np.ndarray] = None,
    compute_gt_baseline: bool = True,
) -> Optional[Dict[str, Any]]:
    """Extract masked region, find initial state, and run trajectory optimization.

    Identifies the masked region, extracts the initial state from the last
    unmasked timestep, and solves for optimal controls that best reproduce
    the reconstruction subject to the aircraft dynamics model.

    Works with joint masks (from sequential_mask_transform) where all features
    share the same mask pattern.

    Args:
        original_sequence: Shape (seq_len, feat_dim), original data in original scale.
        reconstructed_sequence: Shape (seq_len, feat_dim), model output in original scale.
        mask: Shape (seq_len, feat_dim) or (seq_len,), True=keep, False=masked.
        feature_map: PhysicsFeatureMap for column mapping.
        dynamics: AircraftDynamics instance.
        lambda_control: Control regularization weight.
        min_segment_length: Minimum contiguous masked segment to evaluate.
        max_iter: Maximum optimizer iterations.
        state_weights: Per-state weighting for trajectory optimization loss.

    Returns:
        Dict with physics loss results plus metadata, or None if no valid segment.
    """
    # Reduce mask to 1D (use first column, assuming joint masking)
    if mask.ndim == 2:
        mask_1d = mask[:, 0].astype(bool)
    else:
        mask_1d = mask.astype(bool)

    regions = find_contiguous_masked_regions(mask_1d)

    # Filter by minimum length
    regions = [(s, e) for s, e in regions if (e - s) >= min_segment_length]

    if not regions:
        return None

    # Pick the longest region (for middle mode there may be one,
    # for forecast it should be the last one)
    longest = max(regions, key=lambda r: r[1] - r[0])
    mask_start, mask_end = longest

    # Need at least one unmasked timestep before the region for initial state
    if mask_start == 0:
        return None

    # Extract physics states
    x0 = feature_map.extract_state(original_sequence[mask_start - 1])
    x_rec = feature_map.extract_state(reconstructed_sequence[mask_start:mask_end])
    x_true = feature_map.extract_state(original_sequence[mask_start:mask_end])

    # Run trajectory optimization on reconstruction
    recon_result = trajectory_optimization_loss(
        dynamics, x0, x_rec,
        lambda_control=lambda_control,
        state_weights=state_weights,
        max_iter=max_iter,
    )

    # Optionally run trajectory optimization on ground truth as calibration baseline
    if compute_gt_baseline:
        gt_result = trajectory_optimization_loss(
            dynamics, x0, x_true,
            lambda_control=lambda_control,
            state_weights=state_weights,
            max_iter=max_iter,
        )
        recon_result["gt_physics_loss"] = gt_result["physics_loss"]
        recon_result["gt_state_tracking_loss"] = gt_result["state_tracking_loss"]
        recon_result["gt_per_state_mse"] = gt_result["per_state_mse"]
        recon_result["gt_physics_trajectory"] = gt_result["physics_trajectory"]
    else:
        recon_result["gt_physics_loss"] = 0.0
        recon_result["gt_state_tracking_loss"] = 0.0
        recon_result["gt_per_state_mse"] = np.zeros_like(recon_result["per_state_mse"])
        recon_result["gt_physics_trajectory"] = None
    recon_result["mask_start"] = mask_start
    recon_result["mask_end"] = mask_end
    recon_result["x0"] = x0
    recon_result["x_reconstructed"] = x_rec
    recon_result["x_true"] = x_true

    return recon_result
