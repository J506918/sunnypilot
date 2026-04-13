"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import numpy as np
from collections import deque
from enum import IntEnum

from opendbc.car.lateral import FRICTION_THRESHOLD
from opendbc.sunnypilot.car.interfaces import LatControlInputs
from opendbc.sunnypilot.car.lateral_ext import get_friction as get_friction_in_torque_space
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]

# Stage D learning constants
D_LEARNING_RATE = 0.02           # EMA learning rate for friction / steer_ratio
D_WINDOW_SIZE = 50               # rolling window for variance / convergence check
D_VARIANCE_THRESHOLD = 1e-5      # variance threshold — below this → converged
D_MIN_SPEED = 5.0                # m/s  — don't learn at very low speed
D_MIN_CURVATURE = 0.002          # 1/m  — ignore near-straight segments
D_PERSIST_INTERVAL = 100         # frames between Params writes
D_FRICTION_MIN = 0.005
D_FRICTION_MAX = 0.30
# Allow ±30% of the fingerprint-time steer_ratio — this range covers the typical
# real-world variation from tire wear, load, and temperature across vehicles.
D_STEER_RATIO_SCALE_MIN = 0.70   # learned steer_ratio lower bound (fraction of CP value)
D_STEER_RATIO_SCALE_MAX = 1.30   # learned steer_ratio upper bound (fraction of CP value)
# Bias in the friction estimator: a well-tracking car has ~0.02 normalised torque error;
# deviations above/below this steer the EMA update toward higher/lower friction.
D_FRICTION_BIAS = 0.02
# Variance ratio above which D is considered unstable and A is paused
D_INSTABILITY_VARIANCE_MULTIPLIER = 10.0

# Stage A learning constants
A_CORRECTION_RATE = 0.005        # how quickly A correction tracks error
A_CORRECTION_MAX = 0.20          # max |additive correction| → factor in [0.8, 1.2]
A_WEIGHT_RAMP_RATE = 0.0002      # per-frame ramp rate from 0→A_WEIGHT_MAX
A_WEIGHT_MAX = 0.50              # A's maximum blending weight
A_WARMUP_FRAMES = 500            # frames after D convergence before A is allowed
# During warmup, A is allowed to pre-ramp to this fraction of A_WEIGHT_MAX
A_WARMUP_WEIGHT_FRACTION = 0.20
# Rate to reduce A weight when D becomes unstable (multiples of ramp rate)
A_DEGRADE_RATE_MULTIPLIER = 5.0
# A error is pre-scaled before accumulation to keep correction values conservative
A_ERROR_SCALE = 0.10
# A reaches FULLY_ADAPTIVE state once its blend weight exceeds this fraction of max
A_FULLY_ADAPTIVE_WEIGHT_FRACTION = 0.50


class ATCState(IntEnum):
    OFF = 0               # RTTC disabled
    D_LEARNING = 1        # Stage D: learning friction / steer_ratio
    D_CONVERGED_A_WARMING = 2  # D done; A accumulating
    FULLY_ADAPTIVE = 3    # D + A both active
    A_PAUSED = 4          # D drifting; A paused until D re-converges


def roll_pitch_adjust(roll, pitch):
    return roll * math.cos(pitch)


def smooth_friction(error, deadzone, threshold, torque_params):
    error_adj = abs(error) - deadzone
    if error_adj <= 0.0:
        return 0.0
    x = error_adj / max(threshold, 0.01)
    friction_mag = torque_params.friction * math.tanh(2.0 * x)
    return friction_mag if error > 0.0 else -friction_mag


class RealTimeTorqueCorrection(LatControlTorqueExtBase):
    def __init__(self, lac_torque, CP, CP_SP, CI):
        super().__init__(lac_torque, CP, CP_SP, CI)
        self.params = Params()
        self.enabled = self.params.get_bool("RealTimeTorqueCorrection")

        # =============================================================
        # All car-specific params from CP — set at fingerprint time
        # =============================================================
        self.steer_ratio = CP.steerRatio                                    # e.g. 13.7 ~ 17.0
        self.steer_actuator_delay = CP.steerActuatorDelay                   # e.g. 0.08 ~ 0.30
        self.car_mass = CP.mass                                             # e.g. 1200 ~ 2500 kg
        self.wheelbase = CP.wheelbase                                       # e.g. 2.5 ~ 3.0 m
        self.center_to_front = CP.centerToFront
        self.tire_stiffness_front = CP.tireStiffnessFront
        self.tire_stiffness_rear = CP.tireStiffnessRear
        self.car_friction = CP.lateralTuning.torque.friction                # offline friction
        self.car_lat_accel_factor = CP.lateralTuning.torque.latAccelFactor  # offline torque sensitivity
        self.steering_angle_deadzone = CP.lateralTuning.torque.steeringAngleDeadzoneDeg

        # =============================================================
        # Adaptive learning state — Stage D
        # =============================================================
        self._d_friction = self.car_friction          # learned friction (starts at CP value)
        self._d_steer_ratio = self.steer_ratio        # learned steer_ratio (starts at CP value)
        self._d_friction_window: deque = deque(maxlen=D_WINDOW_SIZE)
        self._d_steer_ratio_window: deque = deque(maxlen=D_WINDOW_SIZE)
        self._d_converged = False
        self._d_persist_counter = 0

        # Stage A state
        self._a_correction = 0.0      # additive torque correction (signed)
        self._a_weight = 0.0          # current blend weight 0→A_WEIGHT_MAX
        self._a_warmup_frames = 0     # frames since D first converged
        self._a_active = False        # True once A is fully warming

        self._atc_state = ATCState.OFF
        self._status_counter = 0      # throttle Params writes for status

        # Restore learned state from persistent storage
        self._load_learned_state()

        # =============================================================
        # Derive all gains from CP — zero magic numbers
        # =============================================================

        # -- Understeer gradient from tire stiffness & mass distribution --
        # K_us > 0 means understeer; bigger K_us → needs more correction
        center_to_rear = self.wheelbase - self.center_to_front
        weight_front = self.car_mass * 9.81 * center_to_rear / self.wheelbase
        weight_rear = self.car_mass * 9.81 * self.center_to_front / self.wheelbase
        k_us = (weight_front / max(self.tire_stiffness_front, 1.0)
                - weight_rear / max(self.tire_stiffness_rear, 1.0))
        k_us_factor = 1.0 + 5.0 * max(k_us, 0.0)  # understeer amplifies correction need

        # -- Yaw inertia factor: heavier & longer → more sluggish yaw response --
        yaw_inertia = self.car_mass * self.wheelbase ** 2 / 12.0  # approx Iz
        yaw_inertia_ref = 1800.0 * 2.7 ** 2 / 12.0
        inertia_scale = yaw_inertia / yaw_inertia_ref

        # -- Steering sensitivity: high ratio & low latAccelFactor → needs more gain --
        steering_sensitivity = self.steer_ratio / max(self.car_lat_accel_factor, 0.5)
        steering_sensitivity_ref = 15.0 / 2.5
        sensitivity_scale = steering_sensitivity / steering_sensitivity_ref

        # -- Master car scale: geometric mean of all factors --
        car_scale = (inertia_scale * sensitivity_scale * k_us_factor) ** (1.0 / 3.0)
        car_scale = float(np.clip(car_scale, 0.5, 2.5))

        # -- Layer 3: position correction --
        self.position_error_gain = 0.015 * car_scale
        self.position_error_max = 0.003 * car_scale

        # -- Layer 2: heading rate correction --
        # scale inversely with wheelbase: short car yaws faster
        wheelbase_scale = 2.7 / max(self.wheelbase, 1.5)
        self.heading_error_gain = 0.12 * car_scale * wheelbase_scale

        # -- Curvature-adaptive boost --
        # more friction → less extra boost needed in bends
        self._update_curvature_boost()

        # -- Straight-line damping --
        # high steer ratio → twitchier on-center → damp more
        damp_base = float(np.clip(0.55 + 0.015 * self.steer_ratio, 0.6, 0.85))
        self.straight_damp_bp = [0.0, 0.0005, 0.001]
        self.straight_damp_v = [damp_base, 0.9, 1.0]

        # -- Jerk anticipation --
        # more actuator delay → look further ahead & apply more
        delay_factor = self.steer_actuator_delay / 0.12  # normalized to typical delay
        self.jerk_anticipation_gain = float(np.clip(0.20 * delay_factor, 0.08, 0.50))
        self.jerk_anticipation_lookahead = [
            0.3 + self.steer_actuator_delay,
            0.6 + self.steer_actuator_delay,
            1.0 + self.steer_actuator_delay,
        ]

        # -- EPS saturation protection --
        self.eps_saturation_threshold = 0.95
        self.eps_degrade_bp = [0.95, 1.0]
        self.eps_degrade_v = [1.0, 0.6]

        # =============================================================
        # Filters — time constants derived from car dynamics
        # =============================================================
        self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
        self.pitch_last = 0.0

        # heading filter: slower steering → lower cutoff to avoid noise
        heading_filter_hz = 2.0 * (15.0 / max(self.steer_ratio, 10.0))
        self.heading_error_filter = FirstOrderFilter(
            0.0, 1.0 / (2.0 * math.pi * heading_filter_hz), 0.01
        )

        # position filter: heavier car → more smoothing
        position_filter_tau = 0.6 + 0.3 * (self.car_mass / 1800.0)
        self.position_error_filter = FirstOrderFilter(0.0, position_filter_tau, 0.01)

        self.jerk_filter = FirstOrderFilter(0.0, 0.1, 0.01)
        self.curvature_bias = 0.0
        self.eps_saturated_count = 0

        # Past data buffers
        self.past_times = [-0.3, -0.2, -0.1]
        history_check_frames = [int(abs(i) * 100) for i in self.past_times]
        self.history_frame_offsets = [
            history_check_frames[0] - i for i in history_check_frames
        ]
        self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
        self.roll_deque = deque(maxlen=history_check_frames[0])

    # ==================================================================
    # Curvature-adaptive boost (recomputed when friction is updated)
    # ==================================================================
    def _update_curvature_boost(self):
        friction_ratio = max(self._d_friction, 0.01) / 0.1
        curve_scale = 1.0 / max(math.sqrt(friction_ratio), 0.5)
        curve_scale = float(np.clip(curve_scale, 0.6, 1.8))
        self.curvature_boost_bp = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
        self.curvature_boost_v = [
            1.0,
            1.0,
            1.0 + 0.05 * curve_scale,
            1.0 + 0.15 * curve_scale,
            1.0 + 0.30 * curve_scale,
            1.0 + 0.50 * curve_scale,
            1.0 + 0.75 * curve_scale,
            1.0 + 1.00 * curve_scale,
        ]

    # ==================================================================
    # Persistence helpers
    # ==================================================================
    def _load_learned_state(self):
        try:
            raw = self.params.get("ATCLearnedState", encoding="utf-8")
            if raw:
                state = json.loads(raw)
                friction = float(state.get("friction", self.car_friction))
                sr = float(state.get("steer_ratio", self.steer_ratio))
                self._d_friction = float(np.clip(friction, D_FRICTION_MIN, D_FRICTION_MAX))
                sr_min = self.steer_ratio * D_STEER_RATIO_SCALE_MIN
                sr_max = self.steer_ratio * D_STEER_RATIO_SCALE_MAX
                self._d_steer_ratio = float(np.clip(sr, sr_min, sr_max))
                self._d_converged = bool(state.get("d_converged", False))
                self._a_correction = float(np.clip(state.get("a_correction", 0.0),
                                                   -A_CORRECTION_MAX, A_CORRECTION_MAX))
                self._a_weight = float(np.clip(state.get("a_weight", 0.0), 0.0, A_WEIGHT_MAX))
        except Exception:
            pass  # if anything goes wrong, start fresh

    def _save_learned_state(self):
        try:
            state = {
                "friction": round(self._d_friction, 6),
                "steer_ratio": round(self._d_steer_ratio, 4),
                "d_converged": self._d_converged,
                "a_correction": round(self._a_correction, 6),
                "a_weight": round(self._a_weight, 6),
            }
            self.params.put("ATCLearnedState", json.dumps(state))
        except Exception:
            pass

    def _reset_learned_state(self):
        self._d_friction = self.car_friction
        self._d_steer_ratio = self.steer_ratio
        self._d_friction_window.clear()
        self._d_steer_ratio_window.clear()
        self._d_converged = False
        self._a_correction = 0.0
        self._a_weight = 0.0
        self._a_warmup_frames = 0
        self._a_active = False
        self._update_curvature_boost()
        try:
            self.params.remove("ATCLearnedState")
            self.params.put_bool("AdaptiveTorqueControlReset", False)
        except Exception:
            pass

    # ==================================================================
    # Stage D: adaptive parameter learning
    # ==================================================================
    def _should_learn_d(self, CS) -> bool:
        if CS.vEgo < D_MIN_SPEED:
            return False
        if abs(self._desired_curvature) < D_MIN_CURVATURE:
            return False
        if CS.steeringPressed:
            return False
        if self._steer_limited_by_safety:
            return False
        if self.eps_saturated_count > 50:
            return False
        # Require meaningful lateral excitation
        if abs(self._desired_lateral_accel) < 0.5:
            return False
        return True

    def _update_stage_d(self, CS):
        if not self._should_learn_d(CS):
            return

        # --- friction sample ---
        # Estimate observed friction from the torque error signal:
        # when error is small (car tracking well), friction estimate is close
        # to current value; adapt toward it using EMA.
        torque_error = abs(self._pid_log.error)
        # A small positive error → actual friction slightly higher; negative → lower
        friction_obs = self._d_friction + D_LEARNING_RATE * (torque_error - D_FRICTION_BIAS)
        friction_obs = float(np.clip(friction_obs, D_FRICTION_MIN, D_FRICTION_MAX))
        self._d_friction = (1.0 - D_LEARNING_RATE) * self._d_friction + D_LEARNING_RATE * friction_obs
        self._d_friction_window.append(self._d_friction)

        # --- steer_ratio sample ---
        # Use the ratio of desired vs actual yaw rate as a proxy for steer_ratio error
        desired_yaw = (self._desired_curvature + self.curvature_bias) * CS.vEgo
        actual_yaw = -CS.yawRate
        if abs(desired_yaw) > 0.01:
            sr_correction = actual_yaw / desired_yaw
            sr_correction = float(np.clip(sr_correction, D_STEER_RATIO_SCALE_MIN, D_STEER_RATIO_SCALE_MAX))
            sr_obs = self._d_steer_ratio * sr_correction
            sr_min = self.steer_ratio * D_STEER_RATIO_SCALE_MIN
            sr_max = self.steer_ratio * D_STEER_RATIO_SCALE_MAX
            sr_obs = float(np.clip(sr_obs, sr_min, sr_max))
            self._d_steer_ratio = (1.0 - D_LEARNING_RATE) * self._d_steer_ratio + D_LEARNING_RATE * sr_obs
            self._d_steer_ratio_window.append(self._d_steer_ratio)

        # --- convergence check ---
        if len(self._d_friction_window) >= D_WINDOW_SIZE:
            arr = np.array(self._d_friction_window)
            var = float(np.var(arr))
            was_converged = self._d_converged
            self._d_converged = var < D_VARIANCE_THRESHOLD
            if self._d_converged:
                self._update_curvature_boost()
            elif was_converged:
                # D drifted — pause A
                self._a_active = False
                self._a_warmup_frames = 0

    # ==================================================================
    # Stage A: lightweight adaptive correction
    # ==================================================================
    def _update_stage_a(self, CS):
        if not self._d_converged:
            return

        self._a_warmup_frames += 1

        # Check if D has become unstable again (high variance → pause A)
        if len(self._d_friction_window) >= D_WINDOW_SIZE:
            arr = np.array(self._d_friction_window)
            var = float(np.var(arr))
            if var > D_VARIANCE_THRESHOLD * D_INSTABILITY_VARIANCE_MULTIPLIER:
                # D drifting — degrade A weight back toward 0
                self._a_weight = max(0.0, self._a_weight - A_WEIGHT_RAMP_RATE * A_DEGRADE_RATE_MULTIPLIER)
                self._a_active = False
                return

        # Ramp A weight in once we've warmed up long enough
        if self._a_warmup_frames >= A_WARMUP_FRAMES:
            self._a_active = True
            self._a_weight = min(A_WEIGHT_MAX, self._a_weight + A_WEIGHT_RAMP_RATE)
        else:
            # Still warming — allow some gentle ramp up during warmup
            warmup_fraction = self._a_warmup_frames / A_WARMUP_FRAMES
            target = A_WEIGHT_MAX * A_WARMUP_WEIGHT_FRACTION * warmup_fraction
            if self._a_weight < target:
                self._a_weight = target

        # Compute a small correction based on tracking error
        error = self._pid_log.error  # torque-space error
        # EMA-based correction: moves toward reducing persistent error
        self._a_correction = ((1.0 - A_CORRECTION_RATE) * self._a_correction
                              + A_CORRECTION_RATE * error * A_ERROR_SCALE)
        self._a_correction = float(np.clip(self._a_correction, -A_CORRECTION_MAX, A_CORRECTION_MAX))

    # ==================================================================
    # Apply Stage A correction to output torque
    # ==================================================================
    def _apply_stage_a_correction(self):
        if self._a_weight <= 0.0 or not self._d_converged:
            return
        # Blend: final = (1 - w) * base + w * (base + correction)
        #             = base + w * correction
        self._output_torque += self._a_weight * self._a_correction

    # ==================================================================
    # ATC state machine
    # ==================================================================
    def _compute_atc_state(self) -> ATCState:
        if not self._rttc_enabled:
            return ATCState.OFF
        if not self._d_converged:
            return ATCState.D_LEARNING
        if self._d_converged and not self._a_active:
            # Check if A was previously active and is now paused
            if self._a_weight > 0.0:
                return ATCState.A_PAUSED
            return ATCState.D_CONVERGED_A_WARMING
        if self._a_active and self._a_weight >= A_WEIGHT_MAX * A_FULLY_ADAPTIVE_WEIGHT_FRACTION:
            return ATCState.FULLY_ADAPTIVE
        return ATCState.D_CONVERGED_A_WARMING

    def _update_atc_status(self):
        self._status_counter += 1
        if self._status_counter < 10:
            return
        self._status_counter = 0
        new_state = self._compute_atc_state()
        if new_state != self._atc_state:
            self._atc_state = new_state
        try:
            self.params.put("ATCStatus", str(int(self._atc_state)))
        except Exception:
            pass

    @property
    def _rttc_enabled(self):
        return self.enabled and self.model_valid

    def update_limits(self):
        if not self._rttc_enabled:
            return
        self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)

    def update_lateral_lag(self, lag):
        super().update_lateral_lag(lag)

    # ==================================================================
    # Layer 3: path position error → curvature bias
    # ==================================================================
    def _compute_position_correction(self, CS):
        if (
            self.model_v2 is None
            or not hasattr(self.model_v2, "position")
            or len(self.model_v2.position.y) == 0
        ):
            self.curvature_bias = 0.0
            return

        if CS.vEgo < 5.0:
            self.curvature_bias = 0.0
            return

        raw_position_error = float(self.model_v2.position.y[0])
        filtered_error = self.position_error_filter.update(raw_position_error)
        correction = self.position_error_gain * filtered_error
        self.curvature_bias = float(
            np.clip(correction, -self.position_error_max, self.position_error_max)
        )

    # ==================================================================
    # Layer 2: heading rate error → feedforward boost
    # ==================================================================
    def _compute_heading_correction(self, CS):
        if CS.vEgo < 1.0:
            self.heading_error_filter.x = 0.0
            return 0.0

        desired_yaw_rate = (self._desired_curvature + self.curvature_bias) * CS.vEgo
        actual_yaw_rate = -CS.yawRate
        heading_error_raw = desired_yaw_rate - actual_yaw_rate
        heading_error = self.heading_error_filter.update(heading_error_raw)
        speed_scale = float(np.interp(CS.vEgo, [3.0, 15.0], [0.0, 1.0]))
        return self.heading_error_gain * heading_error * speed_scale

    # ==================================================================
    # EPS saturation protection
    # ==================================================================
    def _compute_eps_protection(self, output_torque):
        if abs(output_torque) > self.eps_saturation_threshold:
            self.eps_saturated_count = min(self.eps_saturated_count + 1, 200)
        else:
            self.eps_saturated_count = max(self.eps_saturated_count - 2, 0)

        if self.eps_saturated_count > 50:
            return float(
                np.interp(abs(output_torque), self.eps_degrade_bp, self.eps_degrade_v)
            )
        return 1.0

    # ==================================================================
    # Jerk anticipation
    # ==================================================================
    def _compute_jerk_anticipation(self, CS):
        if (
            self.model_v2 is None
            or not hasattr(self.model_v2, "acceleration")
            or len(self.model_v2.acceleration.y) < 2
        ):
            return 0.0

        future_jerks = []
        for t in self.jerk_anticipation_lookahead:
            future_lat_accel = float(
                np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y)
            )
            jerk = (future_lat_accel - self._desired_lateral_accel) / max(t, 0.01)
            future_jerks.append(jerk)

        if len(future_jerks) == 0:
            return 0.0

        signs = [1 if j > 0 else (-1 if j < 0 else 0) for j in future_jerks]
        if len(set(signs) - {0}) > 1:
            return 0.0

        min_abs_jerk = min(future_jerks, key=lambda x: abs(x))
        filtered_jerk = self.jerk_filter.update(min_abs_jerk)
        return self.jerk_anticipation_gain * filtered_jerk

    # ==================================================================
    # Curvature-adaptive boost
    # ==================================================================
    def _get_curvature_boost(self):
        abs_curv = abs(self._desired_curvature)
        return float(np.interp(abs_curv, self.curvature_boost_bp, self.curvature_boost_v))

    # ==================================================================
    # Straight-line damping
    # ==================================================================
    def _get_straight_damping(self):
        abs_curv = abs(self._desired_curvature)
        return float(np.interp(abs_curv, self.straight_damp_bp, self.straight_damp_v))

    # ==================================================================
    # Main entry: called from latcontrol_torque_ext.py
    # ==================================================================
    def update_rttc_feedforward(self, CS, params, calibrated_pose):
        if not self._rttc_enabled:
            return

        # --- Check for reset request ---
        try:
            if self.params.get_bool("AdaptiveTorqueControlReset"):
                self._reset_learned_state()
        except Exception:
            pass

        # --- roll & pitch ---
        roll = params.roll
        if calibrated_pose is not None:
            pitch = self.pitch.update(calibrated_pose.orientation.pitch)
            roll = roll_pitch_adjust(roll, pitch)
            self.pitch_last = pitch
        self.roll_deque.append(roll)
        self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

        # --- Layer 3: position correction ---
        self._compute_position_correction(CS)

        # --- Low speed curvature injection ---
        low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
        self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
        self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature

        # --- Apply curvature bias from Layer 3 ---
        if CS.vEgo > 5.0:
            bias_as_lat_accel = self.curvature_bias * CS.vEgo ** 2
            self._setpoint += bias_as_lat_accel

        # --- Layer 1: lateral accel error in torque space ---
        torque_from_setpoint = self.torque_from_lateral_accel_in_torque_space(
            LatControlInputs(self._setpoint, self._roll_compensation, CS.vEgo, CS.aEgo),
            self.lac_torque.torque_params, gravity_adjusted=False,
        )
        torque_from_measurement = self.torque_from_lateral_accel_in_torque_space(
            LatControlInputs(self._measurement, self._roll_compensation, CS.vEgo, CS.aEgo),
            self.lac_torque.torque_params, gravity_adjusted=False,
        )
        base_error = float(torque_from_setpoint - torque_from_measurement)

        # --- Curvature boost & straight damping ---
        curvature_boost = self._get_curvature_boost()
        straight_damp = self._get_straight_damping()
        self._pid_log.error = base_error * curvature_boost * straight_damp

        # --- Stage D: adaptive parameter learning (updates after error is set) ---
        self._update_stage_d(CS)

        # --- Feedforward ---
        self._ff = self.torque_from_lateral_accel_in_torque_space(
            LatControlInputs(
                self._gravity_adjusted_lateral_accel,
                self._roll_compensation,
                CS.vEgo,
                CS.aEgo,
            ),
            self.lac_torque.torque_params, gravity_adjusted=True,
        )

        # --- Smooth friction ---
        friction_input = self.update_friction_input(self._setpoint, self._measurement)
        self._ff += smooth_friction(
            friction_input,
            self._lateral_accel_deadzone,
            FRICTION_THRESHOLD,
            self.lac_torque.torque_params,
        )

        # --- Layer 2: heading correction ---
        heading_ff = self._compute_heading_correction(CS)
        self._ff += heading_ff

        # --- Jerk anticipation ---
        jerk_ff = self._compute_jerk_anticipation(CS)
        self._ff += jerk_ff

        # --- PID output ---
        freeze_integrator = (
            self._steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
        )
        self._output_torque = self._pid.update(
            self._pid_log.error,
            feedforward=self._ff,
            speed=CS.vEgo,
            freeze_integrator=freeze_integrator,
        )

        # --- EPS saturation protection ---
        # Skip EPS protection during large curvature turns - sustained high torque is normal
        if abs(self._desired_curvature) <= 0.02:
            eps_factor = self._compute_eps_protection(self._output_torque)
            self._output_torque *= eps_factor
        else:
            # Reset counter so EPS protection doesn't activate upon returning to straight
            self.eps_saturated_count = 0

        # --- Stage A: adaptive correction layer (after D learning, after EPS protection) ---
        self._update_stage_a(CS)
        self._apply_stage_a_correction()

        # --- Persist learned state periodically ---
        self._d_persist_counter += 1
        if self._d_persist_counter >= D_PERSIST_INTERVAL:
            self._d_persist_counter = 0
            self._save_learned_state()

        # --- Update ATC status param ---
        self._update_atc_status()
