"""
Anti-regression test for system/version.py.

Ensures that system.version is importable and exports all symbols required by
manager.py and other startup-chain modules. A previous regression (commit 7485ab6)
reduced this file to a comment-only stub, which caused manager.py to fail at
import time and left devices stuck on the boot logo with no ability to self-heal.
"""
import importlib
import inspect
import json
import types


class TestVersionImports:
  """Verify system.version is importable and exports all required public symbols."""

  def _load_module(self):
    return importlib.import_module("openpilot.system.version")

  def test_module_is_importable(self):
    """system.version must import without raising any exception."""
    mod = self._load_module()
    assert isinstance(mod, types.ModuleType)

  def test_required_functions_present(self):
    """Functions used by manager.py and other startup modules must exist and be callable."""
    mod = self._load_module()
    for name in ("get_build_metadata", "get_version", "get_release_notes", "is_prebuilt", "is_dirty"):
      assert callable(getattr(mod, name, None)), f"system.version must export callable '{name}'"

  def test_required_classes_present(self):
    """Dataclasses used by startup modules must be present."""
    mod = self._load_module()
    for name in ("BuildMetadata", "OpenpilotMetadata"):
      cls = getattr(mod, name, None)
      assert cls is not None, f"system.version must export class '{name}'"
      assert inspect.isclass(cls), f"'{name}' must be a class"

  def test_required_constants_present(self):
    """Module-level constants consumed by various callers must exist with correct types."""
    mod = self._load_module()

    str_constants = (
      "training_version",
      "terms_version",
      "terms_version_sp",
      "sunnylink_consent_version",
      "sunnylink_consent_declined",
      "BUILD_METADATA_FILENAME",
    )
    for name in str_constants:
      value = getattr(mod, name, None)
      assert isinstance(value, str) and value, \
        f"system.version.{name} must be a non-empty str, got {value!r}"

    list_constants = (
      "RELEASE_BRANCHES",
      "TESTED_BRANCHES",
      "RELEASE_SP_BRANCHES",
      "TESTED_SP_BRANCHES",
      "MASTER_SP_BRANCHES",
    )
    for name in list_constants:
      value = getattr(mod, name, None)
      assert isinstance(value, list), f"system.version.{name} must be a list"

    assert isinstance(getattr(mod, "SP_BRANCH_MIGRATIONS", None), dict), \
      "system.version.SP_BRANCH_MIGRATIONS must be a dict"


class TestGetBuildMetadata:
  """Verify get_build_metadata() behaves correctly in all deployment scenarios."""

  def test_fallback_when_no_git_and_no_build_json(self, tmp_path):
    """get_build_metadata() must not raise when neither .git nor build.json is present.

    This is the tarball/snapshot deployment scenario that previously caused
    'Exception: invalid build metadata' and prevented the device from booting.
    """
    # Create a minimal directory that looks like a deployed openpilot tree
    # (no .git, no build.json, but version.h and CHANGELOG.md present)
    sp_common = tmp_path / "sunnypilot" / "common"
    sp_common.mkdir(parents=True)
    (sp_common / "version.h").write_text('#define SUNNYPILOT_VERSION "0.0.0-test"\n')
    (tmp_path / "CHANGELOG.md").write_text("Test version 0.0.0\n\nSome notes")

    from openpilot.system.version import get_build_metadata, BuildMetadata
    # Must not raise
    meta = get_build_metadata(str(tmp_path))
    assert isinstance(meta, BuildMetadata)
    assert meta.channel == "unknown"
    assert meta.openpilot.version == "0.0.0-test"
    assert meta.openpilot.git_commit == "unknown"

  def test_build_json_takes_precedence(self, tmp_path):
    """When build.json exists, its values must be used regardless of .git presence."""
    build_json = {
      "channel": "staging-zh",
      "openpilot": {
        "version": "1.2.3",
        "release_notes": "notes",
        "git_commit": "abc123",
        "git_origin": "https://github.com/example/repo",
        "git_commit_date": "2026-01-01T00:00:00Z",
        "build_style": "unknown",
      }
    }
    (tmp_path / "build.json").write_text(json.dumps(build_json))

    from openpilot.system.version import get_build_metadata, BuildMetadata
    meta = get_build_metadata(str(tmp_path))
    assert isinstance(meta, BuildMetadata)
    assert meta.channel == "staging-zh"
    assert meta.openpilot.version == "1.2.3"
    assert meta.openpilot.git_commit == "abc123"
