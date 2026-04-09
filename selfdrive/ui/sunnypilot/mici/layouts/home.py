"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, BigDialog
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.selfdrive.ui.ui_state import ui_state

class MiciHomeLayoutSP(MiciHomeLayout):
  ITEM_SPACING = 18
  Y_CENTER = 28

  def __init__(self):
    super().__init__()
    self._offroad_mode = False
    self._last_param_update = 0.0
    self._font = gui_app.font(FontWeight.BOLD)
    self._font_size = 32
    self._padding_x = 10
    self._padding_y = 10
    self._icon_lkas = gui_app.texture("icons_mici/settings/device/lkas.png", 110, 110)
    self._offroad_btn_rect = rl.Rectangle(0, 0, 0, 0)

  def _update_state(self):
    super()._update_state()
    now = rl.get_time()
    if now - self._last_param_update > 1.0:
      self._offroad_mode = ui_state.params.get_bool("OffroadMode")
      self._last_param_update = now

  def _render(self, _):
    super()._render(_)

    if self._offroad_mode:
      text = "FORCE OFFROAD"
      bg_color = rl.Color(245, 56, 56, 175)
    else:
      text = "FORCE ONROAD"
      bg_color = rl.Color(60, 160, 80, 175)

    text_size = measure_text_cached(self._font, text, self._font_size)
    rect_width = text_size.x + self._padding_x * 2
    rect_height = self._font_size + self._padding_y * 2

    pos_x = self._rect.x + self.rect.width - self.ITEM_SPACING - rect_width - 5
    pos_y = self._rect.y + self.rect.height - rect_height / 2 - self.Y_CENTER

    self._offroad_btn_rect = rl.Rectangle(pos_x, pos_y, rect_width, rect_height)
    rl.draw_rectangle_rounded(self._offroad_btn_rect, 0.5, 10, bg_color)

    text_x = pos_x + self._padding_x
    text_y = pos_y + self._padding_y / 1.4
    rl.draw_text_ex(self._font, text, rl.Vector2(int(text_x), int(text_y)), self._font_size, 0, rl.WHITE)

  def _handle_always_offroad(self):
    if ui_state.engaged:
      gui_app.push_widget(BigDialog(tr("disengage"), tr("\nto toggle always offroad")))
      return

    def _enable_always_offroad():
      if not ui_state.engaged:
        ui_state.params.put_bool("OffroadMode", True)
        self._offroad_mode = True
        self._last_param_update = rl.get_time()

    def _disable_always_offroad():
      ui_state.params.put_bool("OffroadMode", False)
      self._offroad_mode = False
      self._last_param_update = rl.get_time()

    if self._offroad_mode:
      dlg = BigConfirmationDialog(tr("exit always offroad"), self._icon_lkas, red=True,
                                  confirm_callback=_disable_always_offroad)
    else:
      dlg = BigConfirmationDialog(tr("enable always offroad"), self._icon_lkas, red=True,
                                  confirm_callback=_enable_always_offroad)
    gui_app.push_widget(dlg)

  def _handle_mouse_release(self, mouse_pos):
    if rl.check_collision_point_rec(mouse_pos, self._offroad_btn_rect):
      self._handle_always_offroad()
      return
    super()._handle_mouse_release(mouse_pos)
