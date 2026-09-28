"""

Keeping constants here makes core models independent of dataset/decoder imports.
Input clips are RGB floating-point tensors in [0, 1]; backbones normalize them.
"""

NORMALIZATION = {
    "kinetics": ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "none": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
