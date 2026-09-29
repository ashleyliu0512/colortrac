import cv2
import numpy as np


RED_RANGES = (
    ((0, 100, 80), (10, 255, 255)),
    ((170, 100, 80), (179, 255, 255)),
)
GREEN_RANGES = (((35, 80, 50), (85, 255, 255)),)
BLUE_RANGES = (((90, 80, 50), (135, 255, 255)),)


def find_target_center(frame, color_ranges=RED_RANGES):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in color_ranges:
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centers = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 150 or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        centers.append((center_x, center_y))

    if not centers:
        return None

    center_x = int(np.median([center[0] for center in centers]))
    center_y = int(np.median([center[1] for center in centers]))
    return center_x, center_y


def find_red_center(frame):
    return find_target_center(frame, RED_RANGES)


def find_green_center(frame):
    return find_target_center(frame, GREEN_RANGES)


def find_blue_center(frame):
    return find_target_center(frame, BLUE_RANGES)


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to open camera 0")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            color_detectors = (
                ("RED", find_red_center, (0, 0, 255)),
                ("GREEN", find_green_center, (0, 255, 0)),
                ("BLUE", find_blue_center, (255, 0, 0)),
            )
            for row, (color_name, detector, display_color) in enumerate(color_detectors):
                center = detector(frame)
                if center is None:
                    label = "{0}: not found".format(color_name)
                else:
                    center_x, center_y = center
                    cv2.drawMarker(frame, center, display_color, cv2.MARKER_CROSS, 24, 2)
                    label = "{0}: ({1}, {2})".format(color_name, center_x, center_y)

                cv2.putText(frame, label, (20, 35 + row * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            display_color, 2)

            cv2.imshow("Target center", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()