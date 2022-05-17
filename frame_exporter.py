import random

import cv2
import numpy as np

# change this for output grid size
target_grid_size = (3, 4)

vidcap = cv2.VideoCapture('your_video_file.mov')

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_frames_num = np.prod(target_grid_size)
available_frames_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# change this for output resolution (it is multiplied with target_grid_size to maintain the aspect ratio)
target_resolution = (height * target_grid_size[0] // target_grid_size[1], width)

capture_length = np.math.ceil(available_frames_num / target_frames_num)

merged_grid = np.tile(255, (*target_resolution, 3))

each_grid_size = (target_resolution[0] // target_grid_size[0], target_resolution[1] // target_grid_size[1])

padding = 5

for frame_index in range(available_frames_num):
    success, image = vidcap.read()

    if frame_index % capture_length == 0:
        count = frame_index // capture_length
        i, j = count // target_grid_size[1], count % target_grid_size[1]

        i_start, i_stop = i * each_grid_size[0], (i + 1) * each_grid_size[0]
        j_start, j_stop = j * each_grid_size[1], (j + 1) * each_grid_size[1]

        i_start, i_stop = i_start + padding, i_stop - padding
        j_start, j_stop = j_start + padding, j_stop - padding
        merged_grid[i_start:i_stop, j_start:j_stop] = cv2.resize(image, (
            each_grid_size[1] - padding * 2, each_grid_size[0] - padding * 2))

cv2.imwrite("saved_graphs/merged.jpg", merged_grid)
