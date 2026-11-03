"""贯穿各层的数据结构。"""

from dataclasses import dataclass


@dataclass
class Rect:
    """矩形。坐标是**标定坐标**（截图已归一化到 dpi.BASELINE_WIDTH），
    要变成屏幕物理坐标得走 dpi.box_center_to_screen。"""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
