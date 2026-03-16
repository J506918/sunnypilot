#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Settings Schema Generator
=========================

Generates a SettingsSchema JSON structure at runtime. The schema defines:
- Panels (logical groupings of settings)
- Items (individual settings with widget types)
- Visibility/enablement rules (declarative dependency logic)
- Vehicle-specific settings (per brand)

The schema is consumed by the sunnylink frontend to render a fully
data-driven settings UI. It merges titles, descriptions, and options
from params_metadata.json automatically.

This module is imported by sunnylinkd at startup and generates the
schema in-memory — nothing is written to disk or git.
"""
from __future__ import annotations  # Required for self-referencing SchemaItem dataclass

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC

from openpilot.sunnypilot.sunnylink.capabilities import CAPABILITY_FIELDS

SCHEMA_VERSION = "1.0"
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params_metadata.json")

# ---------------------------------------------------------------------------
# Rule builders
# ---------------------------------------------------------------------------

def offroad_only() -> dict:
  return {"type": "offroad_only"}

def cap(field_name: str, equals: object = True) -> dict:
  return {"type": "capability", "field": field_name, "equals": equals}

def param_eq(key: str, equals: object = True) -> dict:
  return {"type": "param", "key": key, "equals": equals}

def param_cmp(key: str, op: str, value: int | float) -> dict:
  return {"type": "param_compare", "key": key, "op": op, "value": value}

def not_rule(condition: dict) -> dict:
  return {"type": "not", "condition": condition}

def any_of(*conditions: dict) -> dict:
  return {"type": "any", "conditions": list(conditions)}

def all_of(*conditions: dict) -> dict:
  return {"type": "all", "conditions": list(conditions)}


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SchemaItem:
  key: str
  widget: str  # toggle, option, multiple_button, button, info
  title: str | None = None
  description: str | None = None
  options: list[dict] | None = None
  min: int | float | None = None
  max: int | float | None = None
  step: int | float | None = None
  unit: str | None = None
  value_map: dict[str, int] | None = None
  visibility: list[dict] | None = None
  enablement: list[dict] | None = None
  sub_items: list[SchemaItem] | None = None
  action: str | None = None

  def to_dict(self) -> dict:
    d: dict = {"key": self.key, "widget": self.widget}
    for attr in ("title", "description", "options", "min", "max", "step", "unit",
                 "value_map", "visibility", "enablement", "action"):
      val = getattr(self, attr)
      if val is not None:
        d[attr] = val
    if self.sub_items:
      d["sub_items"] = [si.to_dict() for si in self.sub_items]
    return d


@dataclass
class SubPanel:
  id: str
  label: str
  trigger_key: str
  trigger_condition: dict | None = None
  items: list[SchemaItem] = field(default_factory=list)

  def to_dict(self) -> dict:
    d: dict = {"id": self.id, "label": self.label, "trigger_key": self.trigger_key}
    if self.trigger_condition:
      d["trigger_condition"] = self.trigger_condition
    d["items"] = [item.to_dict() for item in self.items]
    return d


@dataclass
class Panel:
  id: str
  label: str
  icon: str
  order: int
  remote_configurable: bool = True
  items: list[SchemaItem] = field(default_factory=list)
  sub_panels: list[SubPanel] = field(default_factory=list)

  def to_dict(self) -> dict:
    d: dict = {
      "id": self.id,
      "label": self.label,
      "icon": self.icon,
      "order": self.order,
      "remote_configurable": self.remote_configurable,
      "items": [item.to_dict() for item in self.items],
    }
    if self.sub_panels:
      d["sub_panels"] = [sp.to_dict() for sp in self.sub_panels]
    return d


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

def _steering_panel() -> Panel:
  return Panel(
    id="steering", label="Steering", icon="steering_wheel", order=1,
    items=[
      SchemaItem(key="Mads", widget="toggle",
                 enablement=[offroad_only()]),
      SchemaItem(key="BlinkerPauseLateralControl", widget="toggle",
                 sub_items=[
                   SchemaItem(key="BlinkerMinLateralControlSpeed", widget="option",
                              visibility=[param_eq("BlinkerPauseLateralControl", True)]),
                   SchemaItem(key="BlinkerLateralReengageDelay", widget="option",
                              visibility=[param_eq("BlinkerPauseLateralControl", True)]),
                 ]),
      SchemaItem(key="EnforceTorqueControl", widget="toggle",
                 visibility=[not_rule(cap("steer_control_type", "angle"))],
                 enablement=[offroad_only(), cap("torque_allowed", True),
                             param_eq("NeuralNetworkLateralControl", False)]),
      SchemaItem(key="NeuralNetworkLateralControl", widget="toggle",
                 visibility=[not_rule(cap("steer_control_type", "angle"))],
                 enablement=[offroad_only(), cap("torque_allowed", True),
                             param_eq("EnforceTorqueControl", False)]),
    ],
    sub_panels=[
      SubPanel(
        id="mads_settings", label="MADS Settings", trigger_key="Mads",
        trigger_condition=param_eq("Mads", True),
        items=[
          SchemaItem(key="MadsMainCruiseAllowed", widget="toggle",
                     enablement=[offroad_only()]),
          SchemaItem(key="MadsUnifiedEngagementMode", widget="toggle",
                     enablement=[offroad_only()]),
          SchemaItem(key="MadsSteeringMode", widget="multiple_button",
                     enablement=[offroad_only()]),
        ],
      ),
      SubPanel(
        id="torque_settings", label="Torque Settings", trigger_key="EnforceTorqueControl",
        trigger_condition=param_eq("EnforceTorqueControl", True),
        items=[
          SchemaItem(key="LiveTorqueParamsToggle", widget="toggle",
                     enablement=[offroad_only()]),
          SchemaItem(key="LiveTorqueParamsRelaxedToggle", widget="toggle",
                     enablement=[offroad_only()],
                     visibility=[param_eq("LiveTorqueParamsToggle", True)]),
          SchemaItem(key="CustomTorqueParams", widget="toggle",
                     enablement=[offroad_only()]),
          SchemaItem(key="TorqueParamsOverrideEnabled", widget="toggle",
                     enablement=[offroad_only()],
                     visibility=[param_eq("CustomTorqueParams", True)]),
          SchemaItem(key="TorqueParamsOverrideLatAccelFactor", widget="option",
                     visibility=[param_eq("CustomTorqueParams", True)]),
          SchemaItem(key="TorqueParamsOverrideFriction", widget="option",
                     visibility=[param_eq("CustomTorqueParams", True)]),
          SchemaItem(key="TorqueControlTune", widget="multiple_button",
                     enablement=[offroad_only()]),
        ],
      ),
      SubPanel(
        id="lane_change_settings", label="Lane Change Settings", trigger_key="AutoLaneChangeTimer",
        items=[
          SchemaItem(key="AutoLaneChangeTimer", widget="multiple_button"),
          SchemaItem(key="AutoLaneChangeBsmDelay", widget="toggle",
                     enablement=[
                       cap("enable_bsm", True),
                       param_cmp("AutoLaneChangeTimer", ">", 0),
                     ]),
        ],
      ),
    ],
  )


def _cruise_panel() -> Panel:
  return Panel(
    id="cruise", label="Cruise", icon="cruise_control", order=2,
    items=[
      SchemaItem(key="LongitudinalPersonality", widget="multiple_button"),
      SchemaItem(key="IntelligentCruiseButtonManagement", widget="toggle",
                 visibility=[cap("icbm_available", True)],
                 enablement=[offroad_only()]),
      SchemaItem(key="CustomAccIncrementsEnabled", widget="toggle",
                 enablement=[offroad_only(),
                             any_of(cap("has_longitudinal_control", True), cap("has_icbm", True))],
                 sub_items=[
                   SchemaItem(key="CustomAccShortPressIncrement", widget="option",
                              visibility=[param_eq("CustomAccIncrementsEnabled", True)]),
                   SchemaItem(key="CustomAccLongPressIncrement", widget="option",
                              visibility=[param_eq("CustomAccIncrementsEnabled", True)]),
                 ]),
      SchemaItem(key="DynamicExperimentalControl", widget="toggle",
                 visibility=[any_of(cap("has_longitudinal_control", True), cap("has_icbm", True))],
                 enablement=[cap("has_longitudinal_control", True)]),
      SchemaItem(key="SmartCruiseControlVision", widget="toggle",
                 visibility=[any_of(cap("has_longitudinal_control", True), cap("has_icbm", True))]),
      SchemaItem(key="SmartCruiseControlMap", widget="toggle",
                 visibility=[any_of(cap("has_longitudinal_control", True), cap("has_icbm", True))]),
    ],
    sub_panels=[
      SubPanel(
        id="speed_limit_settings", label="Speed Limit Settings",
        trigger_key="SpeedLimitMode",
        items=[
          SchemaItem(key="SpeedLimitMode", widget="multiple_button"),
          SchemaItem(key="SpeedLimitPolicy", widget="multiple_button",
                     visibility=[param_cmp("SpeedLimitMode", ">", 0)]),
          SchemaItem(key="SpeedLimitOffsetType", widget="multiple_button",
                     visibility=[param_cmp("SpeedLimitMode", ">", 0)]),
          SchemaItem(key="SpeedLimitValueOffset", widget="option",
                     visibility=[all_of(
                       param_cmp("SpeedLimitMode", ">", 0),
                       param_cmp("SpeedLimitOffsetType", ">", 0),
                     )]),
        ],
      ),
    ],
  )


def _display_panel() -> Panel:
  return Panel(
    id="display", label="Display", icon="display", order=3,
    items=[
      SchemaItem(key="OnroadScreenOffBrightness", widget="multiple_button"),
      SchemaItem(key="OnroadScreenOffTimer", widget="multiple_button",
                 enablement=[
                   # Timer only enabled when brightness is NOT Auto (0) or Auto Dark (1)
                   not_rule(any_of(param_eq("OnroadScreenOffBrightness", 0),
                                   param_eq("OnroadScreenOffBrightness", 1))),
                 ]),
      SchemaItem(key="Brightness", widget="option"),
      SchemaItem(key="InteractivityTimeout", widget="multiple_button"),
    ],
  )


def _visuals_panel() -> Panel:
  return Panel(
    id="visuals", label="Visuals", icon="visuals", order=4,
    items=[
      SchemaItem(key="BlindSpot", widget="toggle"),
      SchemaItem(key="TorqueBar", widget="toggle"),
      SchemaItem(key="ShowTurnSignals", widget="toggle"),
      SchemaItem(key="RoadNameToggle", widget="toggle"),
      SchemaItem(key="StandstillTimer", widget="toggle"),
      SchemaItem(key="RocketFuel", widget="toggle"),
      SchemaItem(key="ChevronInfo", widget="multiple_button",
                 enablement=[cap("has_longitudinal_control", True)]),
      SchemaItem(key="DevUIInfo", widget="multiple_button"),
      SchemaItem(key="TrueVEgoUI", widget="toggle"),
      SchemaItem(key="HideVEgoUI", widget="toggle"),
      SchemaItem(key="GreenLightAlert", widget="toggle"),
      SchemaItem(key="LeadDepartAlert", widget="toggle"),
      SchemaItem(key="RainbowMode", widget="toggle"),
    ],
  )


def _device_panel() -> Panel:
  return Panel(
    id="device", label="Device", icon="device", order=5,
    items=[
      SchemaItem(key="OffroadMode", widget="toggle"),
      SchemaItem(key="DeviceBootMode", widget="multiple_button"),
      SchemaItem(key="IsMetric", widget="toggle"),
      SchemaItem(key="QuietMode", widget="toggle"),
      SchemaItem(key="OnroadUploads", widget="toggle"),
      SchemaItem(key="LanguageSetting", widget="info"),
      SchemaItem(key="RecordFront", widget="toggle"),
      SchemaItem(key="RecordAudio", widget="toggle"),
      SchemaItem(key="MaxTimeOffroad", widget="option"),
      SchemaItem(key="DisengageOnAccelerator", widget="toggle"),
    ],
  )


def _software_panel() -> Panel:
  """software.py: DisableUpdates only."""
  return Panel(
    id="software", label="Software", icon="software", order=6,
    items=[
      SchemaItem(key="DisableUpdates", widget="toggle",
                 enablement=[offroad_only()],
                 visibility=[param_eq("ShowAdvancedControls", True)]),
    ],
  )


def _sunnylink_panel() -> Panel:
  """sunnylink.py: SunnylinkEnabled, EnableSunnylinkUploader."""
  return Panel(
    id="sunnylink", label="sunnylink", icon="sunnylink", order=7,
    remote_configurable=False,
    items=[
      SchemaItem(key="SunnylinkEnabled", widget="toggle",
                 enablement=[offroad_only()]),
      SchemaItem(key="EnableSunnylinkUploader", widget="toggle",
                 enablement=[param_eq("SunnylinkEnabled", True)]),
    ],
  )


def _developer_panel() -> Panel:
  """Upstream DeveloperLayout + DeveloperLayoutSP combined.

  Upstream (selfdrive/ui/layouts/settings/developer.py):
    AdbEnabled, SshEnabled, JoystickDebugMode, LongitudinalManeuverMode,
    AlphaLongitudinalEnabled, ShowDebugInfo

  SP overlay (selfdrive/ui/sunnypilot/layouts/settings/developer.py):
    ShowAdvancedControls, EnableGithubRunner, EnableCopyparty, QuickBootToggle
  """
  return Panel(
    id="developer", label="Developer", icon="developer", order=8,
    items=[
      # ── Upstream items ──
      SchemaItem(key="AdbEnabled", widget="toggle",
                 enablement=[offroad_only()]),
      SchemaItem(key="SshEnabled", widget="toggle"),
      SchemaItem(key="JoystickDebugMode", widget="toggle",
                 visibility=[not_rule(cap("is_release", True))],
                 enablement=[offroad_only()]),
      SchemaItem(key="LongitudinalManeuverMode", widget="toggle",
                 visibility=[not_rule(cap("is_release", True))],
                 enablement=[offroad_only(), cap("has_longitudinal_control", True)]),
      SchemaItem(key="AlphaLongitudinalEnabled", widget="toggle",
                 visibility=[all_of(
                   cap("alpha_long_available", True),
                   not_rule(cap("is_release", True)),
                 )]),
      SchemaItem(key="ShowDebugInfo", widget="toggle"),
      # ── SP overlay items ──
      SchemaItem(key="ShowAdvancedControls", widget="toggle"),
      SchemaItem(key="EnableGithubRunner", widget="toggle",
                 visibility=[all_of(
                   param_eq("ShowAdvancedControls", True),
                   not_rule(cap("is_release", True)),
                 )]),
      SchemaItem(key="EnableCopyparty", widget="toggle",
                 visibility=[param_eq("ShowAdvancedControls", True)]),
      SchemaItem(key="QuickBootToggle", widget="toggle",
                 visibility=[all_of(
                   param_eq("ShowAdvancedControls", True),
                   not_rule(cap("is_release", True)),
                   not_rule(cap("is_development", True)),
                 )],
                 enablement=[param_eq("DisableUpdates", True)]),
    ],
  )


def _models_panel() -> Panel:
  """models.py: CameraOffset, LagdToggle, LagdToggleDelay, LaneTurnDesire, LaneTurnValue.

  Not remote-configurable (dedicated frontend UI for model selection).
  """
  return Panel(
    id="models", label="Models", icon="models", order=9,
    remote_configurable=False,
    items=[
      SchemaItem(key="LaneTurnDesire", widget="toggle"),
      SchemaItem(key="LaneTurnValue", widget="option",
                 visibility=[all_of(
                   param_eq("LaneTurnDesire", True),
                   param_eq("ShowAdvancedControls", True),
                 )]),
      SchemaItem(key="LagdToggle", widget="toggle"),
      SchemaItem(key="LagdToggleDelay", widget="option",
                 visibility=[all_of(
                   not_rule(param_eq("LagdToggle", True)),
                   param_eq("ShowAdvancedControls", True),
                 )]),
      SchemaItem(key="CameraOffset", widget="option"),
    ],
  )


# ---------------------------------------------------------------------------
# Vehicle-specific settings (per brand)
# ---------------------------------------------------------------------------

def _vehicle_settings() -> dict[str, list[dict]]:
  return {
    "hyundai": [
      SchemaItem(
        key="HyundaiLongitudinalTuning", widget="multiple_button",
        visibility=[cap("alpha_long_available", True)],
        enablement=[offroad_only(), cap("has_longitudinal_control", True)],
      ).to_dict(),
    ],
    "toyota": [
      SchemaItem(
        key="ToyotaEnforceStockLongitudinal", widget="toggle",
        enablement=[offroad_only()],
      ).to_dict(),
      SchemaItem(
        key="ToyotaStopAndGoHack", widget="toggle",
        enablement=[
          offroad_only(),
          cap("has_longitudinal_control", True),
          param_eq("ToyotaEnforceStockLongitudinal", False),
        ],
      ).to_dict(),
    ],
    "tesla": [
      SchemaItem(
        key="TeslaCoopSteering", widget="toggle",
        enablement=[offroad_only()],
      ).to_dict(),
    ],
    "subaru": [
      SchemaItem(
        key="SubaruStopAndGo", widget="toggle",
        enablement=[offroad_only(), cap("has_stop_and_go", True)],
      ).to_dict(),
      SchemaItem(
        key="SubaruStopAndGoManualParkingBrake", widget="toggle",
        enablement=[offroad_only(), cap("has_stop_and_go", True)],
      ).to_dict(),
    ],
  }


# ---------------------------------------------------------------------------
# Metadata loading from params_metadata.json
#
# Titles, descriptions, options, and numeric constraints live in
# params_metadata.json — a JSON file alongside this module.
# The generator reads it at runtime and merges into schema items.
# ---------------------------------------------------------------------------

def _load_metadata() -> dict:
  """Load params_metadata.json for title/description/options enrichment."""
  if not os.path.exists(METADATA_PATH):
    return {}
  try:
    with open(METADATA_PATH) as f:
      return json.load(f)
  except (json.JSONDecodeError, OSError):
    return {}


def _enrich_item(item_dict: dict, metadata: dict) -> None:
  """Merge metadata into a schema item dict (mutates in place)."""
  key = item_dict.get("key", "")
  meta = metadata.get(key, {})

  # Only fill in fields not already set by the panel definition
  if "title" not in item_dict and "title" in meta:
    item_dict["title"] = meta["title"]
  if "description" not in item_dict and "description" in meta and meta["description"]:
    item_dict["description"] = meta["description"]
  if "options" not in item_dict and "options" in meta:
    item_dict["options"] = meta["options"]
  if "min" not in item_dict and "min" in meta:
    item_dict["min"] = meta["min"]
  if "max" not in item_dict and "max" in meta:
    item_dict["max"] = meta["max"]
  if "step" not in item_dict and "step" in meta:
    item_dict["step"] = meta["step"]
  if "unit" not in item_dict and "unit" in meta:
    item_dict["unit"] = meta["unit"]

  # Recurse into sub_items
  for sub in item_dict.get("sub_items", []):
    _enrich_item(sub, metadata)


def _enrich_schema(schema: dict, metadata: dict) -> None:
  """Enrich all items in the schema with metadata."""
  for panel in schema.get("panels", []):
    for item in panel.get("items", []):
      _enrich_item(item, metadata)
    for sub_panel in panel.get("sub_panels", []):
      for item in sub_panel.get("items", []):
        _enrich_item(item, metadata)

  for brand_items in schema.get("vehicle_settings", {}).values():
    for item in brand_items:
      _enrich_item(item, metadata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_schema() -> dict:
  """Generate the complete SettingsSchema dict."""
  panels = [
    _steering_panel(),
    _cruise_panel(),
    _display_panel(),
    _visuals_panel(),
    _device_panel(),
    _software_panel(),
    _sunnylink_panel(),
    _developer_panel(),
    _models_panel(),
  ]

  schema = {
    "schema_version": SCHEMA_VERSION,
    "generated_at": datetime.now(UTC).isoformat(),
    "panels": [p.to_dict() for p in panels],
    "vehicle_settings": _vehicle_settings(),
    "capability_fields": list(CAPABILITY_FIELDS),
  }

  metadata = _load_metadata()
  _enrich_schema(schema, metadata)

  return schema


def generate_schema_json() -> str:
  """Generate SettingsSchema as a JSON string."""
  return json.dumps(generate_schema(), separators=(",", ":"))


def _walk_rules(rules: list[dict] | None, visitor: Callable[[dict], None]) -> None:
  """Recursively walk all rules, calling visitor on each leaf rule."""
  if not rules:
    return
  for rule in rules:
    visitor(rule)
    if rule.get("type") == "not" and "condition" in rule:
      _walk_rules([rule["condition"]], visitor)
    elif rule.get("type") in ("any", "all") and "conditions" in rule:
      _walk_rules(rule["conditions"], visitor)


def _walk_all_items(schema: dict, visitor: Callable[[dict], None]) -> None:
  """Walk every item in the schema (panels, sub_panels, sub_items, vehicle_settings)."""
  def _visit_item(item: dict) -> None:
    visitor(item)
    for sub in item.get("sub_items", []):
      _visit_item(sub)

  for panel in schema.get("panels", []):
    for item in panel.get("items", []):
      _visit_item(item)
    for sp in panel.get("sub_panels", []):
      for item in sp.get("items", []):
        _visit_item(item)

  for brand_items in schema.get("vehicle_settings", {}).values():
    for item in brand_items:
      _visit_item(item)


def collect_all_keys(schema: dict) -> set[str]:
  """Collect all param keys referenced in the schema (items + rules)."""
  keys: set[str] = set()

  def _visit_rule(rule: dict) -> None:
    if rule.get("type") in ("param", "param_compare") and "key" in rule:
      keys.add(rule["key"])

  def _visit_item(item: dict) -> None:
    if "key" in item:
      keys.add(item["key"])
    _walk_rules(item.get("visibility"), _visit_rule)
    _walk_rules(item.get("enablement"), _visit_rule)

  _walk_all_items(schema, _visit_item)
  return keys


def collect_capability_refs(schema: dict) -> set[str]:
  """Collect all capability field names referenced in rules."""
  refs: set[str] = set()

  def _visit_rule(rule: dict) -> None:
    if rule.get("type") == "capability" and "field" in rule:
      refs.add(rule["field"])

  def _visit_item(item: dict) -> None:
    _walk_rules(item.get("visibility"), _visit_rule)
    _walk_rules(item.get("enablement"), _visit_rule)

  _walk_all_items(schema, _visit_item)
  return refs


if __name__ == "__main__":
  # CLI: print schema for inspection
  schema = generate_schema()
  print(json.dumps(schema, indent=2))
  print(f"\nTotal panels: {len(schema['panels'])}")
  print(f"Total capability fields: {len(schema['capability_fields'])}")
  print(f"Total vehicle brands: {len(schema['vehicle_settings'])}")
  keys = collect_all_keys(schema)
  print(f"Total unique param keys: {len(keys)}")
