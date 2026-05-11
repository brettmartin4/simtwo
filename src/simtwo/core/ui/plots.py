from __future__ import annotations

from typing import Sequence

import glfw 
import imgui 
from imgui.integrations.glfw import GlfwRenderer 
from OpenGL import GL


def draw_line_plot(label: str, xs: Sequence[float], ys: Sequence[float], size: tuple[float, float] = (700, 260), pad: float = 10.0) -> None:
    """
    Uses the window draw list to draw a simple line plot in an ImGui child regio
    """

    imgui.text(label)
    child_id = "##" + "".join(ch if ch.isalnum() else "_" for ch in label) + "_plot"
    imgui.begin_child(child_id, width=size[0], height=size[1], border=True)

    draw_list = imgui.get_window_draw_list()
    x0, y0 = imgui.get_cursor_screen_pos()
    w, h = size

    left = x0 + pad
    right = x0 + w - pad
    top = y0 + pad
    bottom = y0 + h - pad

    draw_list.add_rect_filled(
        x0, y0, x0 + w, y0 + h,
        imgui.get_color_u32_rgba(0.10, 0.10, 0.10, 1.0)
    )
    draw_list.add_rect(
        x0, y0, x0 + w, y0 + h,
        imgui.get_color_u32_rgba(0.60, 0.60, 0.60, 1.0)
    )

    if len(xs) >= 2 and len(ys) >= 2:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if xmax == xmin:
            xmax = xmin + 1
        if ymax == ymin:
            ymax = ymin + 1e-12

        def to_screen(x: float, y: float) -> tuple[float, float]:
            sx = left + (x - xmin) / (xmax - xmin) * (right - left)
            sy = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
            return sx, sy

        draw_list.add_line(
            left, bottom, right, bottom,
            imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0
        )
        draw_list.add_line(
            left, top, left, bottom,
            imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0
        )

        color = imgui.get_color_u32_rgba(0.20, 0.70, 1.00, 1.0)
        thickness = 2.0

        for i in range(1, len(xs)):
            x_a, y_a = to_screen(xs[i - 1], ys[i - 1])
            x_b, y_b = to_screen(xs[i], ys[i])
            draw_list.add_line(x_a, y_a, x_b, y_b, color, thickness)

        imgui.set_cursor_screen_pos((x0 + pad, y0 + h - pad - 18))
        imgui.text(f"x:[{xmin},{xmax}]   y:[{ymin:.3e},{ymax:.3e}]")
    else:
        imgui.set_cursor_screen_pos((x0 + pad, y0 + pad))
        imgui.text("No data yet...")

    imgui.dummy(w, h)
    imgui.end_child()
