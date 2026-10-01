import time

import cv2


class CameraReadError(RuntimeError):
    pass


def read_with_retry(camera, max_attempts=10, retry_delay=0.1):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_delay < 0:
        raise ValueError("retry_delay cannot be negative")

    last_error = None
    for attempt in range(max_attempts):
        try:
            ret, frame = camera.read()
        except cv2.error as error:
            last_error = error
            ret, frame = False, None

        if ret and frame is not None:
            return True, frame
        if attempt + 1 < max_attempts:
            time.sleep(retry_delay)

    message = "No camera frame after {0} consecutive attempts".format(max_attempts)
    raise CameraReadError(message) from last_error


class RetryingCamera:
    def __init__(self, camera, max_attempts=10, retry_delay=0.1):
        self._camera = camera
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay

    def read(self):
        return read_with_retry(self._camera, self._max_attempts, self._retry_delay)

    def release(self):
        self._camera.release()

    def __getattr__(self, name):
        return getattr(self._camera, name)