from pathlib import Path


class Shapes:
    sin = (3,) + (32,) * 3  # shape input
    sinc = 3  # shape input channels
    sout = (3,) + (128,) * 3  # shape output
    soutc = 3  # shape output channels
    slatent = (3, 4, 4, 4)


def ensured_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path