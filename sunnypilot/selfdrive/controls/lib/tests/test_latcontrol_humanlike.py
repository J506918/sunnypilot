"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_humanlike import LatControlHumanLike

LOW_SPEED_MPS = 4.0
HIGH_SPEED_MPS = 20.0
MAX_ZERO_CROSSING_DELTA = 0.3


class TorqueParams:
  def __init__(self):
    self.latAccelFactor = 2.5
    self.latAccelOffset = 0.0
    self.friction = 0.1
    self.steeringAngleDeadzoneDeg = 0.0

  def as_builder(self):
    clone = TorqueParams()
    clone.latAccelFactor = self.latAccelFactor
    clone.latAccelOffset = self.latAccelOffset
    clone.friction = self.friction
    clone.steeringAngleDeadzoneDeg = self.steeringAngleDeadzoneDeg
    return clone


class FakeCI:
  @staticmethod
  def torque_from_lateral_accel():
    return lambda lateral_accel, torque_params: lateral_accel / float(torque_params.latAccelFactor)

  @staticmethod
  def lateral_accel_from_torque():
    return lambda torque, torque_params: torque * float(torque_params.latAccelFactor)

  @staticmethod
  def torque_from_lateral_accel_in_torque_space():
    return lambda inputs, torque_params, gravity_adjusted: inputs.lateral_acceleration / float(torque_params.latAccelFactor)


class FakeVM:
  @staticmethod
  def calc_curvature(angle_rad, v_ego, _roll):
    return angle_rad / max(v_ego, 1.0)


def build_controller():
  torque_params = TorqueParams()
  cp = SimpleNamespace(
    steerLimitTimer=0.8,
    lateralTuning=SimpleNamespace(torque=torque_params),
  )
  return LatControlHumanLike(cp, None, FakeCI(), 0.01)


def build_car_state(v_ego):
  return SimpleNamespace(
    vEgo=v_ego,
    aEgo=0.0,
    steeringAngleDeg=0.0,
    steeringPressed=False,
  )


def build_live_params():
  return SimpleNamespace(angleOffsetDeg=0.0, roll=0.0)


def build_model(curvature):
  horizon = 33
  speed = 10.0
  yaw_rate = curvature * speed
  return SimpleNamespace(
    velocity=SimpleNamespace(x=[speed] * horizon),
    orientationRate=SimpleNamespace(z=[yaw_rate] * horizon),
    acceleration=SimpleNamespace(y=[curvature * speed * speed] * horizon),
  )


class TestLatControlHumanLike:
  def test_low_speed_authority_is_stronger(self):
    controller = build_controller()
    vm = FakeVM()
    params = build_live_params()
    desired_lateral_accel = 0.32

    low_speed_output = 0.0
    high_speed_output = 0.0

    for _ in range(20):
      low_speed_output, _, _ = controller.update(True, build_car_state(LOW_SPEED_MPS), vm, params, False,
                                                 desired_lateral_accel / (LOW_SPEED_MPS ** 2), None, False, 0.2)

    controller.reset()

    for _ in range(20):
      high_speed_output, _, _ = controller.update(True, build_car_state(HIGH_SPEED_MPS), vm, params, False,
                                                  desired_lateral_accel / (HIGH_SPEED_MPS ** 2), None, False, 0.2)

    assert abs(low_speed_output) > abs(high_speed_output)

  def test_zero_crossing_stays_smooth(self):
    controller = build_controller()
    vm = FakeVM()
    params = build_live_params()
    cs = build_car_state(6.0)

    first_output, _, _ = controller.update(True, cs, vm, params, False, 0.003, None, False, 0.2)
    second_output, _, _ = controller.update(True, cs, vm, params, False, -0.003, None, False, 0.2)

    assert abs(second_output - first_output) < MAX_ZERO_CROSSING_DELTA

  def test_preview_model_can_bias_turn_in(self):
    controller = build_controller()
    vm = FakeVM()
    params = build_live_params()
    cs = build_car_state(10.0)

    no_preview_output, _, _ = controller.update(True, cs, vm, params, False, 0.0, None, False, 0.2)

    controller.reset()
    controller.extension.update_model_v2(build_model(0.02))

    preview_output = 0.0
    for _ in range(8):
      preview_output, _, _ = controller.update(True, cs, vm, params, False, 0.0, None, False, 0.2)

    assert abs(preview_output) > abs(no_preview_output)
