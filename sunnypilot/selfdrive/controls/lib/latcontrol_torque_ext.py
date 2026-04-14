"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl
from openpilot.sunnypilot.selfdrive.controls.lib.rttc.rttc import RealTimeTorqueCorrection
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride


class LatControlTorqueExt(NeuralNetworkLateralControl, LatControlTorqueExtOverride):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    NeuralNetworkLateralControl.__init__(self, lac_torque, CP, CP_SP, CI)
    LatControlTorqueExtOverride.__init__(self, CP)
    self._rttc = RealTimeTorqueCorrection(lac_torque, CP, CP_SP, CI)

  def update_model_v2(self, model_v2):
    super().update_model_v2(model_v2)
    self._rttc.update_model_v2(model_v2)

  def update_limits(self):
    super().update_limits()
    self._rttc.update_limits()

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self._rttc.update_lateral_lag(lag)

  def update(self, CS, VM, pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
             desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
             desired_curvature, actual_curvature, steer_limited_by_safety, output_torque):
    # --- Populate shared input state for NNLC (self) ---
    self._ff = ff
    self._pid = pid
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._roll_compensation = roll_compensation
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel
    self._desired_curvature = desired_curvature
    self._actual_curvature = actual_curvature
    self._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
    self._steer_limited_by_safety = steer_limited_by_safety
    self._output_torque = output_torque

    if self._rttc._rttc_enabled:
      # --- ATC/RTTC path: fully isolated from NNLC state ---
      # update_calculations is called ONLY on the RTTC instance, using its own
      # internal state. The NNLC (self) update_calculations is intentionally
      # skipped to prevent state pollution between the two controllers.
      self._rttc._ff = ff
      self._rttc._pid = pid
      self._rttc._pid_log = pid_log
      self._rttc._setpoint = setpoint
      self._rttc._measurement = measurement
      self._rttc._roll_compensation = roll_compensation
      self._rttc._lateral_accel_deadzone = lateral_accel_deadzone
      self._rttc._desired_lateral_accel = desired_lateral_accel
      self._rttc._actual_lateral_accel = actual_lateral_accel
      self._rttc._desired_curvature = desired_curvature
      self._rttc._actual_curvature = actual_curvature
      self._rttc._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
      self._rttc._steer_limited_by_safety = steer_limited_by_safety
      self._rttc._output_torque = output_torque
      self._rttc.update_calculations(CS, VM, desired_lateral_accel)
      self._rttc.update_rttc_feedforward(CS, params, calibrated_pose)
      return self._rttc._pid_log, self._rttc._output_torque
    else:
      # --- NNLC path: only runs when ATC is disabled ---
      # update_calculations is called on NNLC (self) only in this branch,
      # ensuring NNLC internal state (lateral jerk, lookahead, etc.) is never
      # dirtied by ATC runs.
      self.update_calculations(CS, VM, desired_lateral_accel)
      self.update_neural_network_feedforward(CS, params, calibrated_pose)
      return self._pid_log, self._output_torque
