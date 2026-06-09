from __future__ import annotations

from typing import Any, Sequence

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL
from qutip import Bloch
import numpy as np
import matplotlib


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


_POINCARE_TEXTURE_CACHE: dict[str, Any] = {
    "key": None,
    "texture_id": None,
    "width": 0,
    "height": 0,
}


def draw_poincare_bloch_plot(label: str, states: Sequence[Any], size: tuple[float, float] = (700, 360)) -> None:
    """
    Draws qutip Bloch sphere configured as a Poincare sphere (see previous notebook from Dr. B's class for implementation details)

    can also handle some errors (refine later)
    """

    imgui.text(label)
    child_id = "##" + "".join(ch if ch.isalnum() else "_" for ch in label) + "_poincare"
    imgui.begin_child(child_id, width=size[0], height=size[1], border=True)

    vectors = _extract_stokes_vectors(states)
    if not vectors:
        imgui.spacing()
        imgui.text_disabled("No polarization state data to display yet.")
        imgui.text_disabled("Press Start after applying a polarization model.")
        imgui.end_child()
        return

    try:
        tex_id, tex_w, tex_h = _get_or_create_poincare_texture(vectors, size)
        if tex_id is None:
            raise RuntimeError("Texture creation returned no OpenGL texture.")
        draw_w = min(float(tex_w), max(64.0, size[0] - 12.0))
        draw_h = min(float(tex_h), max(64.0, size[1] - 34.0))
        imgui.image(tex_id, draw_w, draw_h)
        latest = vectors[-1]
        imgui.text(f"S1={latest[0]: .3f}   S2={latest[1]: .3f}   S3={latest[2]: .3f}")
    except Exception as exc:
        _draw_poincare_fallback(vectors, size, str(exc))

    imgui.end_child()


def _extract_stokes_vectors(states: Sequence[Any], max_points: int = 150) -> list[tuple[float, float, float]]:
    vectors: list[tuple[float, float, float]] = []
    for state in states[-max_points:]:
        vec = _state_to_stokes_vector(state)
        if vec is not None:
            vectors.append(vec)
    return vectors


def _state_to_stokes_vector(state: Any) -> tuple[float, float, float] | None:
    if state is None:
        return None

    if isinstance(state, dict):
        for key in ("stokes", "stokes_vector", "poincare", "bloch"):
            if key in state:
                return _sequence_to_stokes(state[key])
        if all(k in state for k in ("S1", "S2", "S3")):
            return _sequence_to_stokes([state["S1"], state["S2"], state["S3"]])

    return _sequence_to_stokes(state)


def _sequence_to_stokes(value: Any) -> tuple[float, float, float] | None:
    try:
        vals = list(value)
    except TypeError:
        return None

    if len(vals) >= 3 and not any(isinstance(vals[i], complex) for i in range(3)):
        try:
            s1, s2, s3 = float(vals[0]), float(vals[1]), float(vals[2])
            return _normalize_stokes((s1, s2, s3))
        except (TypeError, ValueError):
            return None

    # accept a two-amplitude qubit/polarization ket vec [a, b] and convert it to a stokes vector
    # This keep compatibility with the older callback that was already here
    if len(vals) >= 2:
        try:
            a = complex(vals[0])
            b = complex(vals[1])
            norm = (abs(a) ** 2 + abs(b) ** 2) ** 0.5
            if norm <= 0.0:
                return None
            a /= norm
            b /= norm
            s1 = 2.0 * (a.conjugate() * b).real
            s2 = 2.0 * (a.conjugate() * b).imag
            s3 = abs(a) ** 2 - abs(b) ** 2
            return _normalize_stokes((float(s1), float(s2), float(s3)))
        except Exception:
            return None

    return None


def _normalize_stokes(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    import math

    s1, s2, s3 = vec
    norm = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    if norm <= 0.0 or not math.isfinite(norm):
        return (1.0, 0.0, 0.0)
    return (s1 / norm, s2 / norm, s3 / norm)


def _get_or_create_poincare_texture(vectors: list[tuple[float, float, float]], size: tuple[float, float]) -> tuple[int | None, int, int]:
    width = int(max(256, min(768, size[0] - 18)))
    height = int(max(256, min(768, size[1] - 48)))
    key = (width, height, tuple((round(x, 4), round(y, 4), round(z, 4)) for x, y, z in vectors))

    if _POINCARE_TEXTURE_CACHE.get("key") == key:
        return (
            _POINCARE_TEXTURE_CACHE.get("texture_id"),
            int(_POINCARE_TEXTURE_CACHE.get("width") or width),
            int(_POINCARE_TEXTURE_CACHE.get("height") or height),
        )

    rgba = _render_qutip_poincare_rgba(vectors, width=width, height=height)
    texture_id = _rgba_to_texture(rgba)

    old_tex = _POINCARE_TEXTURE_CACHE.get("texture_id")
    if old_tex is not None:
        try:
            GL.glDeleteTextures([old_tex])
        except Exception:
            pass

    _POINCARE_TEXTURE_CACHE.update(
        {
            "key": key,
            "texture_id": texture_id,
            "width": width,
            "height": height,
        }
    )
    return texture_id, width, height


def _render_qutip_poincare_rgba(vectors: list[tuple[float, float, float]], *, width: int, height: int):

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = fig.add_axes([0.04, 0.03, 0.92, 0.82], projection="3d")
    bloch = Bloch(fig=fig, axes=axes)
    bloch.title = ""

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.60)

    bloch.xlabel = [r"$S_1$", ""]
    bloch.ylabel = [r"$S_2$", ""]
    bloch.zlabel = [r"$S_3$", ""]

    if len(vectors) > 1:
        xs = [v[0] for v in vectors]
        ys = [v[1] for v in vectors]
        zs = [v[2] for v in vectors]
        bloch.add_points([xs, ys, zs], meth="l")

    bloch.add_vectors(list(vectors[-1]))
    bloch.render()
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    #rgba = np.flipud(rgba)
    plt.close(fig)
    return rgba


def _rgba_to_texture(rgba) -> int:
    height, width = int(rgba.shape[0]), int(rgba.shape[1])
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA,
        width,
        height,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        rgba.tobytes(),
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return int(texture_id)


def _draw_poincare_fallback(vectors: list[tuple[float, float, float]], size: tuple[float, float], reason: str) -> None:
    draw_list = imgui.get_window_draw_list()
    x0, y0 = imgui.get_cursor_screen_pos()
    w, h = size
    plot_h = max(240.0, h - 70.0)
    cx = x0 + w * 0.5
    cy = y0 + plot_h * 0.5
    r = min(w, plot_h) * 0.38

    draw_list.add_circle(cx, cy, r, imgui.get_color_u32_rgba(0.65, 0.65, 0.65, 1.0), 64, 1.5)
    draw_list.add_line(cx - r, cy, cx + r, cy, imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0)
    draw_list.add_line(cx, cy + r, cx, cy - r, imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 1.0), 1.0)
    draw_list.add_line(cx - 0.7 * r, cy + 0.7 * r, cx + 0.7 * r, cy - 0.7 * r, imgui.get_color_u32_rgba(0.45, 0.45, 0.45, 1.0), 1.0)

    imgui.set_cursor_screen_pos((cx + r + 8, cy - 8))
    imgui.text("S1")
    imgui.set_cursor_screen_pos((cx + 8, cy - r - 18))
    imgui.text("S2")
    imgui.set_cursor_screen_pos((cx + 0.7 * r + 8, cy - 0.7 * r - 8))
    imgui.text("S3")

    def project(vec: tuple[float, float, float]) -> tuple[float, float]:
        s1, s2, s3 = vec
        px = cx + r * (s1 + 0.35 * s3)
        py = cy - r * (s2 + 0.35 * s3)
        return px, py

    if len(vectors) >= 2:
        for a, b in zip(vectors[:-1], vectors[1:]):
            ax, ay = project(a)
            bx, by = project(b)
            draw_list.add_line(ax, ay, bx, by, imgui.get_color_u32_rgba(0.20, 0.70, 1.00, 1.0), 1.0)

    lx, ly = project(vectors[-1])
    draw_list.add_line(cx, cy, lx, ly, imgui.get_color_u32_rgba(1.00, 0.85, 0.20, 1.0), 2.0)
    draw_list.add_circle_filled(lx, ly, 4.0, imgui.get_color_u32_rgba(1.00, 0.85, 0.20, 1.0))

    imgui.dummy(w, plot_h)
    imgui.text_disabled("using fallback Poincare projection.")
    imgui.text_disabled(reason[:120])
    latest = vectors[-1]
    imgui.text(f"S1={latest[0]: .3f}   S2={latest[1]: .3f}   S3={latest[2]: .3f}")
