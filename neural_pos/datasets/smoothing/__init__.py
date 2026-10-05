
from neural_pos.datasets.smoothing.kernel_moving_avg import kernel_moving_average
from neural_pos.datasets.smoothing.temporal_moving_avg import temporal_moving_average


Smoothing = {
    "temporal_moving_average": temporal_moving_average,
    "kernel_moving_average": kernel_moving_average,
}