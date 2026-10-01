import argparse 
#Python 自带的命令行参数解析：在终端/命令行运行程序时额外传参数，而不需要修改代码里的变量
from functools import partial 
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import color_tracker
from color_tracker.utils.camera import CameraReadError, RetryingCamera

# You can determine these values with the HSVColorRangeDetector() 
# HSV（色调、饱和度、亮度）
HSV_LOWER_VALUE = [55, 103, 82]
HSV_UPPER_VALUE = [178, 255, 255]
COLOR_RANGES = {
    'Red':       [([0, 100, 50], [10, 255, 255]), ([170, 100, 50], [179, 255, 255])],
    'LightBlue': [([85, 50, 100], [105, 255, 255])],
    'Blue':      [([106, 120, 50], [130, 255, 255])],
    'Green':     [([35, 80, 50], [85, 255, 255])],
    'Yellow':    [([15, 100, 100], [34, 255, 255])],
    'Black':     [([0, 0, 0], [180, 255, 45])],
}
COLOR_DISPLAY = {
    "Red": (0, 0, 255),
    "LightBlue": (230, 216, 173),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Yellow": (0, 255, 255),
    "Black": (0, 0, 0),
}
def get_args():
    ''' Get the command line arguments. '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-low", "--low", nargs=3, type=int, default=HSV_LOWER_VALUE,
                        help="Lower value for the HSV range. Default = 155, 103, 82")
    parser.add_argument("-high", "--high", nargs=3, type=int, default=HSV_UPPER_VALUE,
                        help="Higher value for the HSV range. Default = 178, 255, 255")
    parser.add_argument("-c", "--contour-area", type=float, default=2500,
                        help="Minimum object contour area. This controls how small objects should be detected. Default = 2500")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    return args



def count_objects_in_ranges(frame, color_ranges, kernel, min_contour_area):
    ''' Count the number of objects in a frame that are within the given color ranges.'''
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in color_ranges:
        lower = np.asarray(lower, dtype=np.uint8)
        upper = np.asarray(upper, dtype=np.uint8)
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return sum(cv2.contourArea(contour) > min_contour_area for contour in contours)


def show_color_counts(counts):
    ''' Show all detected color counts in a single window. '''
    panel = np.full((52 + 42 * len(counts), 360, 3), 245, dtype=np.uint8)
    for index, (color_name, count) in enumerate(counts.items()):
        y = 36 + index * 42
        cv2.rectangle(panel, (20, y - 16), (44, y + 8), COLOR_DISPLAY[color_name], -1)
        cv2.putText(panel, "{0}: {1}".format(color_name, count), (60, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
    cv2.imshow("Color counts", panel)


def tracking_callback(tracker: color_tracker.ColorTracker, verbose: bool = True,
                      min_contour_area: float = 2500, kernel=None):
    ''' Callback function that is called at every iteration of the tracking loop. '''

    cv2.imshow("Camera", tracker.frame)
    counts = {}
    for color_name, hsv_bounds in COLOR_RANGES.items():
        counts[color_name] = count_objects_in_ranges(
            tracker.frame, hsv_bounds, kernel, min_contour_area)
    show_color_counts(counts)
    # Stop the script when we press ESC
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()
    # ASCII 码中 27 代表键盘左上角的 ESC 键
    if verbose:
        for obj in tracker.tracked_objects:
            print("Object {0} center {1}".format(obj.id, obj.last_point))
    # 如果开启了详细输出模式：遍历当前画面里识别到的每一个目标物体。
    # 在控制台终端打印物体的编号 ID (obj.id) 以及当前的中心点坐标 XY 值 (obj.last_point)。

def main():
    args = get_args()

    # Creating a kernel for the morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
   
    # Init the ColorTracker object
    tracker = color_tracker.ColorTracker(max_nb_of_objects=5, max_nb_of_points=20, debug=False)

    # Setting a callback which is called at every iteration
    callback = partial(tracking_callback, verbose=args.verbose,
                       min_contour_area=args.contour_area, kernel=kernel)
    tracker.set_tracking_callback(tracking_callback=callback)
    
    
    # Start tracking with a camera
    with color_tracker.WebCamera(video_src=0) as webcam:
    # 打开电脑摄像头。0 代表系统默认的第一台摄像头。
    # with ... as webcam：使用上下文管理器。
    # 它的好处是即使程序中途报错退出，也会自动关闭并释放摄像头，防止摄像头被后台占用。

        # Start the actual tracking of the object
        try:
            tracker.track(RetryingCamera(webcam),
                          hsv_lower_value=args.low,
                          hsv_upper_value=args.high,
                          min_contour_area=args.contour_area,
                          kernel=kernel)
        except CameraReadError as error:
            print("Camera read failed; stopping tracking: {0}".format(error))
        #启动死循环正式开始追踪。将摄像头、HSV 下限、HSV 上限、最小过滤面积和形态学核
        #全部传入，程序会一直运行直到按下 ESC 或关掉窗口。


if __name__ == "__main__":
    main()
#如果这个文件是被直接运行的（而不是被别的 Python 文件 import 导入的），就立刻执行 main() 主函数。