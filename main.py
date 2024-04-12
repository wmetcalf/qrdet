import cv2 as cv
import time
#
from qrdet import QRDetector, _plot_result


if __name__ == '__main__':
    image = cv.cvtColor(cv.imread(filename='source_files/qrs2.jpg'), code=cv.COLOR_BGR2RGB)
    # image = cv.cvtColor(cv.imread(filename='source_files/qrs7.png'), code=cv.COLOR_BGR2RGB)

    print("\nВызов QRDetector - onnx")
    detector = QRDetector(model_size='s')
    #
    start_time = time.time()
    detections = detector.detect(image=image, is_bgr=False)
    print("ALL --- %s seconds ---\n" % (time.time() - start_time))
    #
    print("len(detections)", len(detections))
    _plot_result(image=image, detections=detections)






