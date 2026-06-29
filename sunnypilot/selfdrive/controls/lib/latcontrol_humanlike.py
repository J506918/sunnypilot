"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD
from opendbc.sunnypilot.car.interfaces import LatControlInputs
from opendbc.sunnypilot.car.lateral_ext import get_friction as get_friction_in_torque_space
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.modeld.constants import ModelConstants

try:
  from openpilot.common.params import Params
except ImportError:  # pragma: no cover - allows pure-Python unit tests in limited environments
  Params = None

VERSION = 2

PREVIEW_SPEED_BP = [0.0, 5.0, 15.0, 30.0]
NEAR_PREVIEW_T = [0.12, 0.15, 0.20, 0.28]
FAR_PREVIEW_T = [0.38, 0.45, 0.58, 0.72]
DESIRED_CURVATURE_WEIGHT = [0.45, 0.40, 0.32, 0.24]
FAR_CURVATURE_WEIGHT = [0.10, 0.14, 0.20, 0.26]

OUTER_KP = [0.38, 0.30, 0.20, 0.12]
OUTER_KI = [0.09, 0.07, 0.045, 0.025]
OUTER_LIMIT = [0.30, 0.24, 0.18, 0.12]

INNER_KP = [0.90, 0.72, 0.52, 0.34]
INNER_KI = [0.12, 0.09, 0.06, 0.03]
INNER_KD = [0.12, 0.09, 0.06, 0.04]

LOW_SPEED_AUTHORITY = [1.85, 1.60, 1.25, 1.00]
RATE_LIMIT = [0.12, 0.09, 0.055, 0.030]

LP_FILTER_CUTOFF_HZ = 5.0


class HumanLikeTorqueParamsOverride:
  def __init__(self):
    self.params = Params() if Params is not None else None
    self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl") if self.params is not None else False
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled") if self.params is not None else False
    self.frame = -1

  def update(self, torque_params) -> bool:
    if self.params is None or not self.enforce_torque_control_toggle:
      return False

    self.frame += 1
    if self.frame % 300 == 0:
      self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")

      if not self.torque_override_enabled:
        return False

      torque_params.latAccelFactor = float(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
      torque_params.friction = float(self.params.get("TorqueParamsOverrideFriction", return_default=True))
      return True

    return False


class HumanLikePreview:
  def __init__(self):
    self.model_v2 = None
    self.model_valid = False
    self.lat_delay = 0.2
    self.near_preview_time = self.lat_delay
    self.far_preview_time = self.lat_delay

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    velocity = getattr(getattr(self.model_v2, "velocity", None), "x", [])
    orientation_rate = getattr(getattr(self.model_v2, "orientationRate", None), "z", [])
    lateral_accel = getattr(getattr(self.model_v2, "acceleration", None), "y", [])
    has_velocity = self.model_v2 is not None and len(velocity) >= CONTROL_N
    has_orientation_rate = has_velocity and len(orientation_rate) >= CONTROL_N
    has_lat_accel = has_velocity and len(lateral_accel) >= CONTROL_N
    self.model_valid = has_orientation_rate or has_lat_accel

  def update_lateral_lag(self, lag):
    self.lat_delay = max(0.01, float(lag))

  def update_limits(self):
    pass

  def _curvature_plan(self):
    if not self.model_valid:
      return None

    velocities = getattr(getattr(self.model_v2, "velocity", None), "x", [])
    orientation_rate = getattr(getattr(self.model_v2, "orientationRate", None), "z", [])
    lateral_accel = getattr(getattr(self.model_v2, "acceleration", None), "y", [])

    speeds = np.maximum(np.asarray(velocities[:len(ModelConstants.T_IDXS)], dtype=float), 0.1)
    if len(orientation_rate) >= len(ModelConstants.T_IDXS):
      yaw_rates = np.asarray(orientation_rate[:len(ModelConstants.T_IDXS)], dtype=float)
      return yaw_rates / speeds

    lat_accels = np.asarray(lateral_accel[:len(ModelConstants.T_IDXS)], dtype=float)
    return lat_accels / np.square(speeds)

  def preview_curvature(self, v_ego, desired_curvature):
    self.near_preview_time = self.lat_delay + float(np.interp(v_ego, PREVIEW_SPEED_BP, NEAR_PREVIEW_T))
    self.far_preview_time = self.lat_delay + float(np.interp(v_ego, PREVIEW_SPEED_BP, FAR_PREVIEW_T))

    curvature_plan = self._curvature_plan()
    if curvature_plan is None:
      return desired_curvature, desired_curvature, desired_curvature

    near_curvature = float(np.interp(self.near_preview_time, ModelConstants.T_IDXS, curvature_plan))
    far_curvature = float(np.interp(self.far_preview_time, ModelConstants.T_IDXS, curvature_plan))

    desired_weight = float(np.interp(v_ego, PREVIEW_SPEED_BP, DESIRED_CURVATURE_WEIGHT))
    far_weight = float(np.interp(v_ego, PREVIEW_SPEED_BP, FAR_CURVATURE_WEIGHT))
    near_weight = max(0.0, 1.0 - desired_weight - far_weight)
    blended_curvature = desired_weight * desired_curvature + near_weight * near_curvature + far_weight * far_curvature

    return blended_curvature, near_curvature, far_curvature


class LatControlHumanLike(LatControl):
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.torque_from_lateral_accel_in_torque_space = CI.torque_from_lateral_accel_in_torque_space()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    self.outer_loop = PIDController([PREVIEW_SPEED_BP, OUTER_KP], [PREVIEW_SPEED_BP, OUTER_KI], rate=1 / self.dt)
    self.inner_loop = PIDController([PREVIEW_SPEED_BP, INNER_KP], [PREVIEW_SPEED_BP, INNER_KI], [PREVIEW_SPEED_BP, INNER_KD],
                                    pos_limit=self.steer_max, neg_limit=-self.steer_max, rate=1 / self.dt)
    self.yaw_rate_error_rate = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.last_yaw_rate_error = 0.0
    self.last_output_torque = 0.0

    self.extension = HumanLikePreview()
    self.torque_override = HumanLikeTorqueParamsOverride()

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def reset(self):
    super().reset()
    self.outer_loop.reset()
    self.inner_loop.reset()
    self.yaw_rate_error_rate.x = 0.0
    self.last_yaw_rate_error = 0.0
    self.last_output_torque = 0.0

  def _rate_limit(self, output_torque, v_ego):
    max_delta = float(np.interp(v_ego, PREVIEW_SPEED_BP, RATE_LIMIT))
    output_torque = np.clip(output_torque, self.last_output_torque - max_delta, self.last_output_torque + max_delta)
    output_torque = float(np.clip(output_torque, -self.steer_max, self.steer_max))
    self.last_output_torque = output_torque
    return output_torque

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    self.extension.update_lateral_lag(lat_delay)
    self.torque_override.update(self.torque_params)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measured_yaw_rate = measured_curvature * CS.vEgo
    actual_lateral_accel = measured_curvature * CS.vEgo ** 2

    preview_curvature, near_curvature, far_curvature = self.extension.preview_curvature(CS.vEgo, desired_curvature)
    desired_yaw_rate_ff = preview_curvature * CS.vEgo
    desired_lateral_jerk = ((far_curvature - near_curvature) * CS.vEgo ** 2) / max(self.extension.far_preview_time - self.extension.near_preview_time, self.dt)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    if not active:
      self.reset()
      output_torque = 0.0
      pid_log.active = False
    else:
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 0.3
      path_tracking_yaw_error = (desired_curvature - measured_curvature) * CS.vEgo

      outer_limit = float(np.interp(CS.vEgo, PREVIEW_SPEED_BP, OUTER_LIMIT))
      self.outer_loop.set_limits(outer_limit, -outer_limit)
      yaw_rate_ref = desired_yaw_rate_ff + self.outer_loop.update(path_tracking_yaw_error, speed=CS.vEgo, freeze_integrator=freeze_integrator)

      desired_lateral_accel = yaw_rate_ref * CS.vEgo
      yaw_rate_error = yaw_rate_ref - measured_yaw_rate
      raw_yaw_rate_error_rate = (yaw_rate_error - self.last_yaw_rate_error) / self.dt
      filtered_yaw_rate_error_rate = self.yaw_rate_error_rate.update(raw_yaw_rate_error_rate)
      self.last_yaw_rate_error = yaw_rate_error

      low_speed_gain = float(np.interp(CS.vEgo, PREVIEW_SPEED_BP, LOW_SPEED_AUTHORITY))
      torque_error = low_speed_gain * self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(yaw_rate_error * max(CS.vEgo, 1.0), 0.0, CS.vEgo, CS.aEgo),
        self.torque_params,
        False,
      )
      torque_error_rate = low_speed_gain * self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(filtered_yaw_rate_error_rate * max(CS.vEgo, 1.0), 0.0, CS.vEgo, CS.aEgo),
        self.torque_params,
        False,
      )

      gravity_adjusted_preview_accel = preview_curvature * CS.vEgo ** 2 - roll_compensation - self.torque_params.latAccelOffset
      ff_torque = low_speed_gain * self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(gravity_adjusted_preview_accel, roll_compensation, CS.vEgo, CS.aEgo),
        self.torque_params,
        True,
      )
      friction_torque = low_speed_gain * get_friction_in_torque_space(
        desired_lateral_accel - actual_lateral_accel,
        lateral_accel_deadzone,
        FRICTION_THRESHOLD,
        self.torque_params,
      )

      output_torque = self.inner_loop.update(
        torque_error,
        error_rate=torque_error_rate,
        speed=CS.vEgo,
        feedforward=ff_torque + friction_torque,
        freeze_integrator=freeze_integrator,
      )
      output_torque = self._rate_limit(output_torque, CS.vEgo)

      pid_log.active = True
      pid_log.error = float(torque_error)
      pid_log.errorRate = float(torque_error_rate)
      pid_log.p = float(self.inner_loop.p + self.outer_loop.p)
      pid_log.i = float(self.inner_loop.i + self.outer_loop.i)
      pid_log.d = float(self.inner_loop.d)
      pid_log.f = float(self.inner_loop.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(desired_lateral_accel)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return -output_torque, 0.0, pid_log
