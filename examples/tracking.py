import argparse 
#Python 自带的命令行参数解析：在终端/命令行运行程序时额外传参数，而不需要修改代码里的变量
from functools import partial 
import cv2
import numpy as np

import color_tracker

# You can determine these values with the HSVColorRangeDetector() 
# HSV（色调、饱和度、亮度）
HSV_LOWER_VALUE = [55, 103, 82]
HSV_UPPER_VALUE = [178, 255, 255]
COLOR_RANGES = {
    "RED": [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))],
    "GREEN": [((35, 80, 50), (85, 255, 255))],
    "BLUE": [((90, 80, 50), (135, 255, 255))],
}
COLOR_DISPLAY = {
    "RED": (0, 0, 255),
    "GREEN": (0, 180, 0),
    "BLUE": (255, 0, 0),
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
# parser（解析器）：创建一个参数解析对象
# add_argument：向解析器添加一条规则
# nargs=3：代表这个参数后面必须跟着 3 个值（对应 H, S, V）。
# type=int：限制输入的值必须是整数。
# default=HSV_LOWER_VALUE：如果用户没传这个参数，就默认使用前面定义的 [155, 103, 82]。
# help：提示说明信息。
# -c / --contour-area：最小轮廓面积（单位：像素点）
#-v / --verbose：详细输出开关。
# parse_args()：开始真正解析命令行命令。
# return args：把解析出来的所有参数结果打包返回出来。


def count_objects_in_ranges(frame, color_ranges, kernel, min_contour_area):
    ''' Count the number of objects in a frame that are within the given color ranges.'''
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in color_ranges:
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return sum(cv2.contourArea(contour) > min_contour_area for contour in contours)


def count_red_objects(frame, kernel, min_contour_area):
    return count_objects_in_ranges(frame, COLOR_RANGES["RED"], kernel, min_contour_area)


def count_green_objects(frame, kernel, min_contour_area):
    return count_objects_in_ranges(frame, COLOR_RANGES["GREEN"], kernel, min_contour_area)


def count_blue_objects(frame, kernel, min_contour_area):
    return count_objects_in_ranges(frame, COLOR_RANGES["BLUE"], kernel, min_contour_area)


def show_color_count(color_name, count):
    ''' Show one color's count in its own window. '''
    panel = np.full((120, 260, 3), 245, dtype=np.uint8)
    color = COLOR_DISPLAY[color_name]
    cv2.rectangle(panel, (20, 24), (52, 56), color, -1)
    cv2.putText(panel, color_name, (68, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
    cv2.putText(panel, "Count: {0}".format(count), (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (35, 35, 35), 2)
    cv2.imshow("{0} count".format(color_name.title()), panel)


def tracking_callback(tracker: color_tracker.ColorTracker, verbose: bool = True,
                      min_contour_area: float = 2500, kernel=None):
    #定义函数，接收两个参数：
    # tracker：当前的追踪器对象（包含画面和目标数据）。
    # verbose：是否开启详细输出（默认 True）

    # Visualizing the original frame and the debugger frame
    cv2.imshow("original frame", tracker.frame)
    cv2.imshow("debug frame", tracker.debug_frame)
    color_counter_functions = {
        "RED": count_red_objects,
        "GREEN": count_green_objects,
        "BLUE": count_blue_objects,
    }
    for color_name, count_function in color_counter_functions.items():
        count = count_function(tracker.frame, kernel, min_contour_area)
        show_color_count(color_name, count)

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
    #创建一个 11x11 像素的椭圆形结构元素

    # Init the ColorTracker object
    tracker = color_tracker.ColorTracker(max_nb_of_objects=5, max_nb_of_points=20, debug=True)

    # Setting a callback which is called at every iteration
    callback = partial(tracking_callback, verbose=args.verbose,
                       min_contour_area=args.contour_area, kernel=kernel)
    tracker.set_tracking_callback(tracking_callback=callback)
    # partial(...)：把 为tracking_callback 函数里的 verbose 参数绑定命令行传入的 args.verbose。
    # set_tracking_callback(...)：把这个函数注册给追踪器。这样追踪器每处理完一帧图像，就会自动调用一次这个函数。
    
    # Start tracking with a camera
    with color_tracker.WebCamera(video_src=0) as webcam:
    # 打开电脑摄像头。0 代表系统默认的第一台摄像头。
    # with ... as webcam：使用上下文管理器。
    # 它的好处是即使程序中途报错退出，也会自动关闭并释放摄像头，防止摄像头被后台占用。

        # Start the actual tracking of the object
        tracker.track(webcam,
                      hsv_lower_value=args.low,
                      hsv_upper_value=args.high,
                      min_contour_area=args.contour_area,
                      kernel=kernel)
        #启动死循环正式开始追踪。将摄像头、HSV 下限、HSV 上限、最小过滤面积和形态学核
        #全部传入，程序会一直运行直到按下 ESC 或关掉窗口。


if __name__ == "__main__":
    main()
#如果这个文件是被直接运行的（而不是被别的 Python 文件 import 导入的），就立刻执行 main() 主函数。