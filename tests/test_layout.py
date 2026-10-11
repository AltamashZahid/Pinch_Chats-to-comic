import math

import numpy as np
from PIL import Image

from pinch.layout import COLUMNS, GUTTER, PANEL, bubble_outline, compose_page, draw_bubble
from pinch.render import RenderedPanel
from pinch.scenes import Line, Panel


def _radii(style):
    pts = bubble_outline(style, 0, 0, 100, 50)
    return [math.hypot(x, y) for x, y in pts]


def test_burst_reaches_further_than_round():
    assert max(_radii("burst")) > max(_radii("jagged")) > max(_radii("round"))


def test_droopy_sags_below_round():
    low = lambda style: max(y for _, y in bubble_outline(style, 0, 0, 100, 50))
    assert low("droopy") > low("round")
    top = lambda style: min(y for _, y in bubble_outline(style, 0, 0, 100, 50))
    assert top("droopy") == top("round")  # only the bottom changes


def test_bubbles_stack_downward_and_stay_inside():
    img = Image.new("RGB", (512, 512), "gray")
    y1 = draw_bubble(img, "first message here", "round", "left", 10)
    y2 = draw_bubble(img, "a much longer second message that has to wrap onto lines", "jagged", "right", y1)
    assert 10 < y1 < y2 < 512
    arr = np.array(img)
    assert (arr[:, :8] == 128).all() and (arr[:, -8:] == 128).all()  # margins untouched


def _panel(tmp_path, i):
    path = tmp_path / f"p{i}.png"
    Image.new("RGB", (512, 512), (40 * i, 90, 140)).save(path)
    return RenderedPanel(Panel([Line("me", "hi", "surprise", 1.0)], "scene"), path, "generated", 0.0)


def test_page_size_and_centred_last_row(tmp_path):
    panels = [_panel(tmp_path, i) for i in range(3)]
    out = compose_page(panels, tmp_path / "page.png")
    page = np.array(Image.open(out))
    assert page.shape[1] == COLUMNS * PANEL + (COLUMNS + 1) * GUTTER
    assert page.shape[0] == 2 * PANEL + 3 * GUTTER
    # Third panel sits in the middle of the second row, so the row's left edge is blank.
    row2_y = 2 * GUTTER + PANEL + PANEL // 2
    assert (page[row2_y, :GUTTER + PANEL // 4] == 255).all()
