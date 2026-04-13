#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import sys

import pyray as rl

FONT_DIR = Path(__file__).resolve().parent
SELFDRIVE_DIR = FONT_DIR.parents[1]
TRANSLATIONS_DIR = SELFDRIVE_DIR / "ui" / "translations"
LANGUAGES_FILE = TRANSLATIONS_DIR / "languages.json"

GLYPH_PADDING = 6
EXTRA_CHARS = "–‑✓×°§•X⚙✕◀▶✔⌫⇧␣○●↳çêüñ–‑✓×°§•€£¥"
UNIFONT_LANGUAGES = {"th", "zh-CHT", "zh-CHS", "ko", "ja"}

# Hash file lives next to the generated font assets (not tracked in git)
TRANSLATION_HASH_FILE = FONT_DIR / ".translation_hash"


def _languages():
  if not LANGUAGES_FILE.exists():
    return {}
  with LANGUAGES_FILE.open(encoding="utf-8") as f:
    return json.load(f)


def _char_sets():
  base = set(map(chr, range(32, 127))) | set(EXTRA_CHARS)
  unifont = set(base)

  for language, code in _languages().items():
    unifont.update(language)
    po_path = TRANSLATIONS_DIR / f"app_{code}.po"
    try:
      chars = set(po_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      continue
    (unifont if code in UNIFONT_LANGUAGES else base).update(chars)

  return tuple(sorted(ord(c) for c in base)), tuple(sorted(ord(c) for c in unifont))


def compute_translation_hash() -> str:
  """Compute a stable hash of the translation character sets used to build the font atlases.

  The hash covers the sorted unique codepoints from all .po files (via _char_sets),
  plus the set of language codes and their names, so any addition of a new language or
  new translation string will invalidate the stored hash and trigger font regeneration.
  """
  base_cp, unifont_cp = _char_sets()
  h = hashlib.sha256()
  h.update(repr(base_cp).encode())
  h.update(repr(unifont_cp).encode())
  return h.hexdigest()


def _font_outputs_exist() -> bool:
  """Return True only when every expected .fnt and .png output file is present."""
  fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
  for font in fonts:
    if "emoji" in font.name.lower():
      continue
    if not (FONT_DIR / f"{font.stem}.fnt").exists():
      return False
    if not (FONT_DIR / f"{font.stem}.png").exists():
      return False
  return True


def ensure_fonts_up_to_date() -> None:
  """Regenerate font atlases when the translation character set has changed or when
  any output file is missing.  Writes a hash file after a successful build so that
  subsequent calls are fast no-ops as long as nothing has changed.

  Safe to call at every startup: the common-path cost is one hash computation plus
  a single file read, which completes in milliseconds.
  """
  current_hash = compute_translation_hash()

  stored_hash: str | None = None
  if TRANSLATION_HASH_FILE.exists():
    try:
      stored_hash = TRANSLATION_HASH_FILE.read_text(encoding="utf-8").strip()
    except OSError:
      pass

  if stored_hash == current_hash and _font_outputs_exist():
    return

  if stored_hash != current_hash:
    print("Translation character set changed, regenerating font atlases...")
  else:
    print("Font atlas outputs missing, regenerating font atlases...")

  main()

  try:
    TRANSLATION_HASH_FILE.write_text(current_hash, encoding="utf-8")
  except OSError as e:
    print(f"Warning: could not write font translation hash file: {e}")


def _glyph_metrics(glyphs, rects, codepoints):
  entries = []
  min_offset_y, max_extent = None, 0
  for idx, codepoint in enumerate(codepoints):
    glyph = glyphs[idx]
    rect = rects[idx]
    width = int(round(rect.width))
    height = int(round(rect.height))
    offset_y = int(round(glyph.offsetY))
    min_offset_y = offset_y if min_offset_y is None else min(min_offset_y, offset_y)
    max_extent = max(max_extent, offset_y + height)
    entries.append({
      "id": codepoint,
      "x": int(round(rect.x)),
      "y": int(round(rect.y)),
      "width": width,
      "height": height,
      "xoffset": int(round(glyph.offsetX)),
      "yoffset": offset_y,
      "xadvance": int(round(glyph.advanceX)),
    })

  if min_offset_y is None:
    raise RuntimeError("No glyphs were generated")

  line_height = int(round(max_extent - min_offset_y))
  base = int(round(max_extent))
  return entries, line_height, base


def _write_bmfont(path: Path, font_size: int, face: str, atlas_name: str, line_height: int, base: int, atlas_size, entries):
  # TODO: why doesn't raylib calculate these metrics correctly?
  if line_height != font_size:
    print("using font size for line height", atlas_name)
    line_height = font_size
  lines = [
    f"info face=\"{face}\" size=-{font_size} bold=0 italic=0 charset=\"\" unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=0,0 outline=0",
    f"common lineHeight={line_height} base={base} scaleW={atlas_size[0]} scaleH={atlas_size[1]} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4",
    f"page id=0 file=\"{atlas_name}\"",
    f"chars count={len(entries)}",
  ]
  for entry in entries:
    lines.append(
      ("char id={id:<4} x={x:<5} y={y:<5} width={width:<5} height={height:<5} " +
       "xoffset={xoffset:<5} yoffset={yoffset:<5} xadvance={xadvance:<5} page=0  chnl=15").format(**entry)
    )
  path.write_text("\n".join(lines) + "\n")


def _process_font(font_path: Path, codepoints: tuple[int, ...]):
  print(f"Processing {font_path.name}...")

  font_size = {
    "unifont.otf": 16,  # unifont is only 16x8 or 16x16 pixels per glyph
  }.get(font_path.name, 200)

  data = font_path.read_bytes()
  file_buf = rl.ffi.new("unsigned char[]", data)
  cp_buffer = rl.ffi.new("int[]", codepoints)
  cp_ptr = rl.ffi.cast("int *", cp_buffer)
  glyphs = rl.load_font_data(rl.ffi.cast("unsigned char *", file_buf), len(data), font_size, cp_ptr, len(codepoints), rl.FontType.FONT_DEFAULT)
  if glyphs == rl.ffi.NULL:
    raise RuntimeError("raylib failed to load font data")

  rects_ptr = rl.ffi.new("Rectangle **")
  image = rl.gen_image_font_atlas(glyphs, rects_ptr, len(codepoints), font_size, GLYPH_PADDING, 0)
  if image.width == 0 or image.height == 0:
    raise RuntimeError("raylib returned an empty atlas")

  rects = rects_ptr[0]
  atlas_name = f"{font_path.stem}.png"
  atlas_path = FONT_DIR / atlas_name
  entries, line_height, base = _glyph_metrics(glyphs, rects, codepoints)

  if not rl.export_image(image, atlas_path.as_posix()):
    raise RuntimeError("Failed to export atlas image")

  _write_bmfont(FONT_DIR / f"{font_path.stem}.fnt", font_size, font_path.stem, atlas_name, line_height, base, (image.width, image.height), entries)


def main():
  base_cp, unifont_cp = _char_sets()
  fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
  for font in fonts:
    if "emoji" in font.name.lower():
      continue
    glyphs = unifont_cp if font.stem.lower().startswith("unifont") else base_cp
    _process_font(font, glyphs)
  return 0


if __name__ == "__main__":
  if "--ensure-up-to-date" in sys.argv:
    ensure_fonts_up_to_date()
  else:
    raise SystemExit(main())
