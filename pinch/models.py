from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass
class Bubble:
    box: Box
    side: str             # "left" or "right"
    fill: tuple[int, int, int]  # BGR fill colour of the bubble
    dark: bool            # True if the fill is dark (light text on dark bubble)
    fill_map: object = field(default=None, compare=False, repr=False)  # (h, w, 3) fill of a gradient bubble


@dataclass
class Message:
    speaker: str          # "them" (left side) or "me" (right side, usually the phone's owner)
    text: str
    box: Box
    emoji: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
