"""Render timing and polarization plots as OpenGL textures and exportable MPL figures."""

from __future__ import annotations

from typing import Any, Sequence

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL
from qutip import Bloch
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math
from matplotlib import cm

_DEFAULT_IMGUI_FONT_SIZE = 13.0


def _font_scale(font_size: float) -> float:
    """Sets image font size with error checking."""
    try:
        value = float(font_size)
    except (TypeError, ValueError):
        value = _DEFAULT_IMGUI_FONT_SIZE
    return max(0.35, min(4.0, value / _DEFAULT_IMGUI_FONT_SIZE))


def _draw_scaled_text(text: str, font_size: float) -> None:
    """Draws text at a given font size."""
    try:
        imgui.set_window_font_scale(_font_scale(font_size))
        imgui.text_unformatted(str(text))
    finally:
        try:
            imgui.set_window_font_scale(1.0)
        except Exception:
            pass


def _draw_scaled_text_at(pos: tuple[float, float], text: str, font_size: float) -> None:
    """Draws text at a specified position with a given font size."""
    old_pos = imgui.get_cursor_screen_pos()
    try:
        imgui.set_cursor_screen_pos(pos)
        _draw_scaled_text(str(text), font_size)
    finally:
        imgui.set_cursor_screen_pos(old_pos)


def _format_tick(value: float) -> str:
    """Applies tick formatting for timing plots."""
    if not math.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 10000 or (0 < abs_value < 0.001):
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _build_axis_ticks(vmin: float, vmax: float, frequency: float = 0.0, count: int = 5) -> list[float]:
    """Draws axis ticks with given parameters for timing plots."""
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return []
    if vmax == vmin:
        return [vmin]
    if frequency and frequency > 0:
        start = math.ceil(vmin / frequency) * frequency
        ticks: list[float] = []
        current = start
        limit = 0
        while current <= vmax and limit < 200:
            ticks.append(current)
            current += frequency
            limit += 1
        return ticks or [vmin, vmax]
    count = max(2, int(count))
    step = (vmax - vmin) / float(count - 1)
    return [vmin + step * i for i in range(count)]


def create_timing_plot_texture(xs: Sequence[float], ys: Sequence[float], *, width: int, height: int, title: str, title_font_size: float, x_axis_label: str, x_axis_font_size: float, y_axis_label: str, y_axis_font_size: float, tick_frequency: float, tick_font_size: float, target_xs: Sequence[float] | None = None, target_ys: Sequence[float] | None = None, target_label: str = "Target", target_y_axis_label: str = "Target", target_y_axis_font_size: float = 13.0) -> tuple[int, int, int]:
    """Render a timing plot to an OpenGL texture.
    
    Args:
        xs: X-axis values to plot.
        ys: Y-axis values to plot.
        width (int): Requested texture or figure width in pixels.
        height (int): Requested texture or figure height in pixels.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
        x_axis_label (str): Text displayed on the x-axis.
        x_axis_font_size (float): Font size used for the x-axis label.
        y_axis_label (str): Text displayed on the y-axis.
        y_axis_font_size (float): Font size used for the y-axis label.
        tick_frequency (float): Optional x-axis tick interval; zero keeps mpl automatic ticks.
        tick_font_size (float): Font size used for axis tick labels.
        target_xs: Optional x-axis values for the target overlay.
        target_ys: Optional target values plotted on the right y-axis.
        target_label (str): Legend label used for target values.
        target_y_axis_label (str): Label for the right y-axis when target values are shown.
        target_y_axis_font_size (float): Font size used for the right y-axis label.
    
    Returns:
        The OpenGL texture id and the rendered texture dimensions.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state."""
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("No timing plot data is available to render.")
    rgba = _render_timing_rgba(
        xs,
        ys,
        width=int(width),
        height=int(height),
        title=title,
        title_font_size=title_font_size,
        x_axis_label=x_axis_label,
        x_axis_font_size=x_axis_font_size,
        y_axis_label=y_axis_label,
        y_axis_font_size=y_axis_font_size,
        tick_frequency=tick_frequency,
        tick_font_size=tick_font_size,
        target_xs=target_xs,
        target_ys=target_ys,
        target_label=target_label,
        target_y_axis_label=target_y_axis_label,
        target_y_axis_font_size=target_y_axis_font_size,
    )
    return _rgba_to_texture(rgba), int(width), int(height)


def create_poincare_plot_texture(states: Sequence[Any], *, width: int, height: int, start_index: int = 0, window_size: int = 200) -> tuple[int, int, int]:
    """Render a Poincare sphere polarization plot to an OpenGL texture.
    
    Args:
        states: Polarization states to render or convert.
        width (int): Requested texture or figure width in pixels.
        height (int): Requested texture or figure height in pixels.
        start_index (int): First observation index included in the polarization distribution window.
        window_size (int): Number of observations included in the polarization distribution window.
    
    Returns:
        The OpenGL texture id and the rendered texture dimensions.
    
    Raises:
        ValueError: If the operation cannot be completed with the current inputs or state.
    """
    vectors = _window_stokes_vectors(states, start_index=start_index, window_size=window_size)
    if not vectors:
        raise ValueError("No polarization plot data is available to render.")
    rgba = _render_poincare_distribution_rgba(vectors, width=int(width), height=int(height))
    return _rgba_to_texture(rgba), int(width), int(height)


def delete_plot_texture(texture_id: int | None) -> None:
    """Delete an OpenGL texture based on id if it exists."""
    if texture_id is None:
        return
    try:
        GL.glDeleteTextures([int(texture_id)])
    except Exception:
        pass


def draw_static_plot_texture(label: str, texture_id: int | None, texture_size: tuple[int, int], size: tuple[float, float], *, title: str | None = None, title_font_size: float = 18.0, empty_message: str = "No plot has been generated yet.", footer_text: str = "") -> None:
    """Draw an existing plot texture without rebuilding it.
    
    Args:
        label (str): Identifier used for the ImGui child window or plot label.
        texture_id (int): OpenGL texture identifier to display or delete.
        texture_size: Pixel width and height of the texture.
        size: Requested display size for the ImGui child region.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
        empty_message (str): Message shown when no texture is available.
        footer_text (str): Optional text displayed beneath the plot.
    """
    if title:
        _draw_scaled_text(title, title_font_size)
    child_id = "##" + "".join(ch if ch.isalnum() else "_" for ch in label) + "_static_texture"
    imgui.begin_child(child_id, width=size[0], height=size[1], border=True)
    tex_w, tex_h = int(texture_size[0] or 0), int(texture_size[1] or 0)
    if texture_id is None or tex_w <= 0 or tex_h <= 0:
        imgui.spacing()
        imgui.text_disabled(empty_message)
        imgui.end_child()
        return
    avail_w = max(64.0, float(size[0]) - 12.0)
    avail_h = max(64.0, float(size[1]) - 36.0)
    scale = min(avail_w / float(tex_w), avail_h / float(tex_h))
    draw_w = float(tex_w) * scale
    draw_h = float(tex_h) * scale
    imgui.image(int(texture_id), draw_w, draw_h)
    if footer_text:
        imgui.text_unformatted(str(footer_text))
    imgui.end_child()


def draw_line_plot(label: str, xs: Sequence[float], ys: Sequence[float], size: tuple[float, float] = (700, 260), pad: float = 10.0, *, title: str | None = None, title_font_size: float = 18.0, x_axis_label: str = "Epoch", x_axis_font_size: float = 13.0, y_axis_label: str = "Prediction", y_axis_font_size: float = 13.0, tick_frequency: float = 0.0, tick_font_size: float = 11.0, target_xs: Sequence[float] | None = None, target_ys: Sequence[float] | None = None, target_label: str = "Target", target_y_axis_label: str = "Target", target_y_axis_font_size: float = 13.0) -> None:
    """Draw a timing plot through the legacy per frame texture cache.
    
    Args:
        label (str): Identifier used for the ImGui child window or plot label.
        xs: X-axis values to plot.
        ys: Y-axis values to plot.
        size: Requested display size for the ImGui child region.
        pad (float): Value used for pad.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
        x_axis_label (str): Text displayed on the x-axis.
        x_axis_font_size (float): Font size used for the x-axis label.
        y_axis_label (str): Text displayed on the y-axis.
        y_axis_font_size (float): Font size used for the y-axis label.
        tick_frequency (float): Optional x-axis tick interval; zero keeps mpl automatic ticks.
        tick_font_size (float): Font size used for axis tick labels.
        target_xs: Optional x-axis values for the target overlay.
        target_ys: Optional target values plotted on the right y-axis.
        target_label (str): Legend label used for target values.
        target_y_axis_label (str): Label for the right y-axis when target values are shown.
        target_y_axis_font_size (float): Font size used for the right y-axis label.
    
    Raises:
        RuntimeError: If the operation cannot be completed with the current inputs or state.
    """
    child_id = "##" + "".join(ch if ch.isalnum() else "_" for ch in label) + "_plot"
    imgui.begin_child(child_id, width=size[0], height=size[1], border=True)

    if len(xs) < 2 or len(ys) < 2:
        imgui.spacing()
        imgui.text("No data yet...")
        imgui.end_child()
        return

    try:
        tex_id, tex_w, tex_h = _get_or_create_timing_texture(
            label,
            xs,
            ys,
            size,
            title=title or label,
            title_font_size=title_font_size,
            x_axis_label=x_axis_label,
            x_axis_font_size=x_axis_font_size,
            y_axis_label=y_axis_label,
            y_axis_font_size=y_axis_font_size,
            tick_frequency=tick_frequency,
            tick_font_size=tick_font_size,
            target_xs=target_xs,
            target_ys=target_ys,
            target_label=target_label,
            target_y_axis_label=target_y_axis_label,
            target_y_axis_font_size=target_y_axis_font_size,
        )
        if tex_id is None:
            raise RuntimeError("Texture creation returned no OpenGL texture.")
        draw_w = min(float(tex_w), max(64.0, size[0] - 12.0))
        draw_h = min(float(tex_h), max(64.0, size[1] - 12.0))
        imgui.image(tex_id, draw_w, draw_h)
    except Exception as exc:
        imgui.spacing()
        imgui.text_disabled("Matplotlib timing plot rendering failed.")
        imgui.text_disabled(str(exc)[:160])

    imgui.end_child()


_TIMING_TEXTURE_CACHE: dict[str, Any] = {
    "key": None,
    "texture_id": None,
    "width": 0,
    "height": 0,
}


def _series_key(values: Sequence[float] | None, precision: int = 6) -> tuple[float, ...]:
    """Round floating point values in a sequence to a given precision"""
    if values is None:
        return ()
    return tuple(round(float(value), precision) for value in values)


def _matplotlib_series_color(index: int = 1) -> str | None:
    try:
        import matplotlib.pyplot as plt

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(colors) > index:
            return colors[index]
    except Exception:
        return None
    return None


def _get_or_create_timing_texture(label: str, xs: Sequence[float], ys: Sequence[float], size: tuple[float, float], *, title: str, title_font_size: float, x_axis_label: str, x_axis_font_size: float, y_axis_label: str, y_axis_font_size: float, tick_frequency: float, tick_font_size: float, target_xs: Sequence[float] | None, target_ys: Sequence[float] | None, target_label: str, target_y_axis_label: str, target_y_axis_font_size: float) -> tuple[int | None, int, int]:
    """Return or create timing texture for internal use.
    
    Args:
        label (str): Identifier used for the ImGui child window or plot label.
        xs: X-axis values to plot.
        ys: Y-axis values to plot.
        size: Requested display size for the ImGui child region.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
        x_axis_label (str): Text displayed on the x-axis.
        x_axis_font_size (float): Font size used for the x-axis label.
        y_axis_label (str): Text displayed on the y-axis.
        y_axis_font_size (float): Font size used for the y-axis label.
        tick_frequency (float): Optional x-axis tick interval; zero keeps mpl automatic ticks.
        tick_font_size (float): Font size used for axis tick labels.
        target_xs: Optional x-axis values for the target overlay.
        target_ys: Optional target values plotted on the right y-axis.
        target_label (str): Legend label used for target values.
        target_y_axis_label (str): Label for the right y-axis when target values are shown.
        target_y_axis_font_size (float): Font size used for the right y-axis label.
    
    Returns:
        The computed texture value
    """
    width = int(max(256, min(1200, size[0] - 12)))
    height = int(max(180, min(800, size[1] - 12)))
    key = (
        label,
        width,
        height,
        str(title),
        round(float(title_font_size), 3),
        str(x_axis_label),
        round(float(x_axis_font_size), 3),
        str(y_axis_label),
        round(float(y_axis_font_size), 3),
        round(float(tick_frequency or 0.0), 6),
        round(float(tick_font_size), 3),
        str(target_label),
        str(target_y_axis_label),
        round(float(target_y_axis_font_size), 3),
        _series_key(xs),
        _series_key(ys),
        _series_key(target_xs),
        _series_key(target_ys),
    )

    if _TIMING_TEXTURE_CACHE.get("key") == key:
        return (
            _TIMING_TEXTURE_CACHE.get("texture_id"),
            int(_TIMING_TEXTURE_CACHE.get("width") or width),
            int(_TIMING_TEXTURE_CACHE.get("height") or height),
        )

    rgba = _render_timing_rgba(
        xs,
        ys,
        width=width,
        height=height,
        title=title,
        title_font_size=title_font_size,
        x_axis_label=x_axis_label,
        x_axis_font_size=x_axis_font_size,
        y_axis_label=y_axis_label,
        y_axis_font_size=y_axis_font_size,
        tick_frequency=tick_frequency,
        tick_font_size=tick_font_size,
        target_xs=target_xs,
        target_ys=target_ys,
        target_label=target_label,
        target_y_axis_label=target_y_axis_label,
        target_y_axis_font_size=target_y_axis_font_size,
    )
    texture_id = _rgba_to_texture(rgba)

    old_tex = _TIMING_TEXTURE_CACHE.get("texture_id")
    if old_tex is not None:
        try:
            GL.glDeleteTextures([old_tex])
        except Exception:
            pass

    _TIMING_TEXTURE_CACHE.update(
        {
            "key": key,
            "texture_id": texture_id,
            "width": width,
            "height": height,
        }
    )
    return texture_id, width, height
    

def _render_timing_rgba( xs: Sequence[float], ys: Sequence[float], *, width: int, height: int, title: str, title_font_size: float, x_axis_label: str, x_axis_font_size: float, y_axis_label: str, y_axis_font_size: float, tick_frequency: float, tick_font_size: float, target_xs: Sequence[float] | None, target_ys: Sequence[float] | None, target_label: str, target_y_axis_label: str, target_y_axis_font_size: float):
    """Generates timing plot.
    
    Args:
        xs: X-axis values to plot.
        ys: Y-axis values to plot.
        width (int): Requested texture or figure width in pixels.
        height (int): Requested texture or figure height in pixels.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
        x_axis_label (str): Text displayed on the x-axis.
        x_axis_font_size (float): Font size used for the x-axis label.
        y_axis_label (str): Text displayed on the y-axis.
        y_axis_font_size (float): Font size used for the y-axis label.
        tick_frequency (float): Optional x-axis tick interval; zero keeps mpl automatic ticks.
        tick_font_size (float): Font size used for axis tick labels.
        target_xs: Optional x-axis values for the target overlay.
        target_ys: Optional target values plotted on the right y-axis.
        target_label (str): Legend label used for target values.
        target_y_axis_label (str): Label for the right y-axis when target values are shown.
        target_y_axis_font_size (float): Font size used for the right y-axis label.
    
    Returns:
        The computed value for the requested operation.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.plot(list(xs), list(ys), linewidth=1.8, label="Prediction")
    ax.set_title(str(title), fontsize=float(title_font_size))
    ax.set_xlabel(str(x_axis_label), fontsize=float(x_axis_font_size))
    ax.set_ylabel(str(y_axis_label), fontsize=float(y_axis_font_size))
    ax.tick_params(axis="both", labelsize=float(tick_font_size))

    handles, labels = ax.get_legend_handles_labels()
    has_target = target_xs is not None and target_ys is not None and len(target_xs) >= 2 and len(target_ys) >= 2
    if has_target:
        ax_right = ax.twinx()
        target_color = _matplotlib_series_color(1) or "C1"
        ax_right.plot(list(target_xs), list(target_ys), linewidth=1.5, color=target_color, label=str(target_label))
        ax_right.set_ylabel(str(target_y_axis_label), fontsize=float(target_y_axis_font_size), color=target_color)
        ax_right.tick_params(axis="y", labelsize=float(tick_font_size), colors=target_color)
        right_handles, right_labels = ax_right.get_legend_handles_labels()
        handles.extend(right_handles)
        labels.extend(right_labels)

    if tick_frequency and float(tick_frequency) > 0:
        ax.xaxis.set_major_locator(MultipleLocator(float(tick_frequency)))
    if handles:
        ax.legend(handles, labels, fontsize=max(6.0, float(tick_font_size)))
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return rgba


_POINCARE_TEXTURE_CACHE: dict[str, Any] = {
    "key": None,
    "texture_id": None,
    "width": 0,
    "height": 0,
}


def draw_poincare_bloch_plot(label: str, states: Sequence[Any], size: tuple[float, float] = (700, 360), *, title: str | None = None, title_font_size: float = 18.0) -> None:
    """Draw a Poincare sphere through the legacy per frame texture cache.
    
    Args:
        label (str): Identifier used for the ImGui child window or plot label.
        states: Polarization states to render or convert.
        size: Requested display size for the ImGui child region.
        title (str): Plot or window title text.
        title_font_size (float): Font size used for the plot title.
    
    Raises:
        RuntimeError: If the operation cannot be completed with the current inputs or state.
    """
    _draw_scaled_text(title or label, title_font_size)
    child_id = "##" + "".join(ch if ch.isalnum() else "_" for ch in label) + "_poincare"
    imgui.begin_child(child_id, width=size[0], height=size[1], border=True)

    vectors = _extract_stokes_vectors(states)
    if not vectors:
        imgui.spacing()
        imgui.text_disabled("No polarization state data to display yet.")
        imgui.text_disabled("Press Generate after applying a polarization model.")
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


def _extract_stokes_vectors(states: Sequence[Any], max_points: int | None = None) -> list[tuple[float, float, float]]:
    """Converts states to stokes vectors.
    
    Args:
        states: Polarization states to render or convert.
        max_points (int): Value used for max points.
    
    Returns:
        The computed list of Stokes vector values.
    """
    vectors: list[tuple[float, float, float]] = []
    source = states if max_points is None else states[-max_points:]
    for state in source:
        vec = _state_to_stokes_vector(state)
        if vec is not None:
            vectors.append(vec)
    return vectors


def _window_stokes_vectors(states: Sequence[Any], *, start_index: int, window_size: int) -> list[tuple[float, float, float]]:
    """Gets all stokes values from within a specified window.
    
    Args:
        states: Polarization states to render or convert.
        start_index (int): First observation index included in the polarization distribution window.
        window_size (int): Number of observations included in the polarization distribution window.
    
    Returns:
        Windowed Stokes value list.
    """
    all_vectors = _extract_stokes_vectors(states, max_points=None)
    if not all_vectors:
        return []
    start = max(0, min(int(start_index), len(all_vectors) - 1))
    size = max(1, int(window_size))
    end = min(len(all_vectors), start + size)
    return all_vectors[start:end]


def _state_to_stokes_vector(state: Any) -> tuple[float, float, float] | None:
    """Converts state to stokes vector."""
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
    """Converts a sequence to stokes values."""
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
    """Applies normalization to stokes vector."""
    s1, s2, s3 = vec
    norm = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    if norm <= 0.0 or not math.isfinite(norm):
        return (1.0, 0.0, 0.0)
    return (s1 / norm, s2 / norm, s3 / norm)


def _get_or_create_poincare_texture(vectors: list[tuple[float, float, float]], size: tuple[float, float]) -> tuple[int | None, int, int]:
    """Return or create poincare texture for internal use.
    
    Args:
        vectors: Normalized Stokes vectors on the Poincare sphere.
        size: Requested display size for the ImGui child region.
    
    Returns:
        Poincare texture.
    """
    width = int(max(256, min(768, size[0] - 18)))
    height = int(max(256, min(768, size[1] - 48)))
    key = (width, height, tuple((round(x, 4), round(y, 4), round(z, 4)) for x, y, z in vectors))

    if _POINCARE_TEXTURE_CACHE.get("key") == key:
        return (
            _POINCARE_TEXTURE_CACHE.get("texture_id"),
            int(_POINCARE_TEXTURE_CACHE.get("width") or width),
            int(_POINCARE_TEXTURE_CACHE.get("height") or height),
        )

    rgba = _render_poincare_distribution_rgba(vectors, width=width, height=height)
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


def _ensure_3d_axes_compatibility(axes: Any) -> None:
    """Ensure 3d axes compatibility is available or valid."""
    try:
        current_dist = getattr(axes, "_dist", None)
        axes._dist = 10.0 if current_dist is None else float(current_dist)
    except Exception:
        try:
            axes._dist = 10.0
        except Exception:
            pass
    try:
        current_dist = getattr(axes, "dist", None)
        if current_dist is None:
            axes.dist = 10.0
    except Exception:
        pass
    try:
        vertical_axis = getattr(axes, "_vertical_axis", None)
        if vertical_axis is None:
            axes._vertical_axis = 2
    except Exception:
        pass


def _safe_set_3d_box_aspect(axes: Any, aspect: tuple[float, float, float]) -> None:
    """Safely sets the Poincare sphere box aspect to fit within the panel."""
    _ensure_3d_axes_compatibility(axes)
    try:
        axes.set_box_aspect(aspect)
    except TypeError as exc:
        message = str(exc)
        if "unary -" not in message or "NoneType" not in message:
            raise
        try:
            axes._vertical_axis = 2
            axes.set_box_aspect(aspect)
        except Exception:
            pass


def _render_poincare_distribution_rgba(vectors: list[tuple[float, float, float]], *, width: int, height: int):
    """Internal helper that calls _render_poincare_distribution_points_rgba; retained for backwars compatibility."""
    return _render_poincare_distribution_points_rgba(vectors, width=width, height=height)


def _point_density_values(data):
    """Calculates point density from stokes vector."""
    s1 = data[:, 0]
    s2 = data[:, 1]
    s3 = data[:, 2]
    phi = np.arctan2(s2, s1)
    theta = np.arccos(np.clip(s3, -1.0, 1.0))
    phi_bins = np.linspace(-np.pi, np.pi, 49)
    theta_bins = np.linspace(0.0, np.pi, 25)
    hist, _, _ = np.histogram2d(phi, theta, bins=(phi_bins, theta_bins))
    phi_idx = np.clip(np.digitize(phi, phi_bins) - 1, 0, hist.shape[0] - 1)
    theta_idx = np.clip(np.digitize(theta, theta_bins) - 1, 0, hist.shape[1] - 1)
    density = hist[phi_idx, theta_idx].astype(float)
    max_density = float(density.max()) if density.size else 0.0
    if max_density > 0.0:
        density = density / max_density
    return density


def _normalized_distribution_data(vectors: list[tuple[float, float, float]]):
    """Creates a normalized Poincare distribution."""
    data = np.asarray(vectors, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    data = data[:, :3]
    norms = np.linalg.norm(data, axis=1)
    valid = np.isfinite(data).all(axis=1) & (norms > 0.0)
    data = data[valid] / norms[valid, None]
    if data.size == 0:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    return data


def _render_poincare_distribution_points_rgba(vectors: list[tuple[float, float, float]], *, width: int, height: int):
    """
    Render a qutip poincare sphere with density colored polarization samples and return its RGBA image.

    Args:
        vectors (list[tuple[float, float, float]]): Normalized Stokes vectors on the Poincare sphere.
        width (int): Width of the ImGui child region in pixels.
        height (int): Height of the ImGui child region in pixels.

    Returns:
        numpy.ndarray: RGBA image buffer of the point based Poincare distribution.
    """
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = fig.add_axes([0.04, 0.03, 0.92, 0.82], projection="3d")
    bloch = Bloch(fig=fig, axes=axes)
    bloch.title = ""
    bloch.xlabel = [r"$S_1$", ""]
    bloch.ylabel = [r"$S_2$", ""]
    bloch.zlabel = [r"$S_3$", ""]
    bloch.render()

    data = _normalized_distribution_data(vectors)
    density = _point_density_values(data)
    point_sizes = 16.0 + 48.0 * density
    axes.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=density,
        cmap="viridis",
        s=point_sizes,
        alpha=0.88,
        depthshade=False,
        linewidths=0.0,
    )

    axes.view_init(elev=24, azim=38)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return rgba

def _rgba_to_texture(rgba) -> int:
    """
    Upload an RGBA Numpy image to OpenGL and return the newly allocated texture identifier.

    Args:
        rgba (numpy.ndarray): Height-by-width RGBA image buffer to upload to OpenGL.

    Returns:
        int: OpenGL texture identifier containing the supplied RGBA image.
    """
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
    """
    Draw a lightweight 2D ImGui Poincare projection when the qutip or mpl texture renderer is unavailable.

    Args:
        vectors (list[tuple[float, float, float]]): Normalized Stokes vectors on the Poincare sphere.
        size (tuple[float, float]): Available ImGui display width and height in pixels.
        reason (str): Short rendering error description shown by the fallback view.
    """
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
    imgui.text_disabled("QuTiP Bloch rendering unavailable; using fallback Poincare projection.")
    imgui.text_disabled(reason[:120])
    latest = vectors[-1]
    imgui.text(f"S1={latest[0]: .3f}   S2={latest[1]: .3f}   S3={latest[2]: .3f}")


def save_timing_plot(path: str, xs: Sequence[float], ys: Sequence[float], *, title: str, title_font_size: float, x_axis_label: str, x_axis_font_size: float, y_axis_label: str, y_axis_font_size: float, tick_frequency: float, tick_font_size: float, target_xs: Sequence[float] | None = None, target_ys: Sequence[float] | None = None, target_label: str = "Target", target_y_axis_label: str = "Target", target_y_axis_font_size: float = 13.0) -> None:
    """
    Write a mpl timing plot, including an optional right-axis target series, to an image file.

    Args:
        path (str): Input file path to read or output file path to write.
        xs (Sequence[float]): X-axis coordinates for generated predictions or plot data.
        ys (Sequence[float]): Primary y-axis prediction values.
        title (str): Title displayed above the plot or native dialog.
        title_font_size (float): Font size used for the plot title in points.
        x_axis_label (str): Text displayed beside the primary x-axis.
        x_axis_font_size (float): Font size used for the x-axis label in points.
        y_axis_label (str): Text displayed beside the primary y-axis.
        y_axis_font_size (float): Font size used for the y-axis label in points.
        tick_frequency (float): Major x-axis tick interval; zero leaves locator selection to mpl.
        tick_font_size (float): Font size used for x- and y-axis tick labels in points.
        target_xs (Sequence[float] | None): Optional x-axis coordinates aligned with observed target values.
        target_ys (Sequence[float] | None): Optional observed target values drawn on the right y-axis.
        target_label (str): Legend label used for the optional target series.
        target_y_axis_label (str): Label displayed beside the optional right y-axis.
        target_y_axis_font_size (float): Font size used for the right y-axis label in points.

    Raises:
        ValueError: Raised when fewer than two timing samples are available to save.
    """
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("No timing plot data is available to save.")

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    ax.plot(list(xs), list(ys), linewidth=1.8, label="Prediction")
    ax.set_title(str(title), fontsize=float(title_font_size))
    ax.set_xlabel(str(x_axis_label), fontsize=float(x_axis_font_size))
    ax.set_ylabel(str(y_axis_label), fontsize=float(y_axis_font_size))
    ax.tick_params(axis="both", labelsize=float(tick_font_size))

    handles, labels = ax.get_legend_handles_labels()
    has_target = target_xs is not None and target_ys is not None and len(target_xs) >= 2 and len(target_ys) >= 2
    if has_target:
        ax_right = ax.twinx()
        target_color = _matplotlib_series_color(1) or "C1"
        ax_right.plot(list(target_xs), list(target_ys), linewidth=1.5, color=target_color, label=str(target_label))
        ax_right.set_ylabel(str(target_y_axis_label), fontsize=float(target_y_axis_font_size), color=target_color)
        ax_right.tick_params(axis="y", labelsize=float(tick_font_size), colors=target_color)
        right_handles, right_labels = ax_right.get_legend_handles_labels()
        handles.extend(right_handles)
        labels.extend(right_labels)

    if tick_frequency and float(tick_frequency) > 0:
        ax.xaxis.set_major_locator(MultipleLocator(float(tick_frequency)))
    if handles:
        ax.legend(handles, labels, fontsize=max(6.0, float(tick_font_size)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_poincare_plot(path: str, states: Sequence[Any], *, title: str, title_font_size: float, start_index: int = 0, window_size: int = 200) -> None:
    """
    Render the requested Poincare distribution window and save the resulting image with a user-specified title.

    Args:
        path (str): Input file path to read or output file path to write.
        states (Sequence[Any]): Polarization observations expressed as Stokes vectors, qubit amplitudes, or supported mappings.
        title (str): Title displayed above the plot or native dialog.
        title_font_size (float): Font size used for the plot title in points.
        start_index (int): Zero-based first observation included in the polarization distribution window.
        window_size (int): Number of consecutive observations included in the polarization distribution window.

    Raises:
        ValueError: Raised when the requested observation window contains no usable polarization states.
    """
    vectors = _window_stokes_vectors(states, start_index=start_index, window_size=window_size)
    if not vectors:
        raise ValueError("No polarization plot data is available to save.")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rgba = _render_poincare_distribution_rgba(vectors, width=900, height=720)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    ax.imshow(np.asarray(rgba))
    ax.axis("off")
    ax.set_title(str(title), fontsize=float(title_font_size), pad=8.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
