from dataclasses import dataclass

@dataclass
class Options:
    znear: float = 0.5
    zfar: float = 2.5
    fovy: float = 1028.9873046875
    fovx: float = 1031.4962158203125
    output_size_x: int = 1280
    output_size_y: int = 720