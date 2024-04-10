import cv2 as cv
import time
#
from qrdet import QRDetectorPT, _plot_result
from qrdet import QRDetector


if __name__ == '__main__':

    print("\nВызов QRDetectorPT (старый)")
    detector = QRDetectorPT(model_size='s')
    # detector = QRDetectorYOLO8()
    image = cv.cvtColor(cv.imread(filename='source_files/qrs7.png'), code=cv.COLOR_BGR2RGB)
    # image = cv.cvtColor(cv.imread(filename='resources/qreader_test_image.jpeg'), code=cv.COLOR_BGR2RGB)
    start_time = time.time()
    detections = detector.detect(image=image, is_bgr=False, legacy=False)
    print("ALL--- %s seconds ---" % (time.time() - start_time))
    #
    print("len(detections)", len(detections))
    # _plot_result(image=image, detections=detections)
    # exit(77)


    print("\nВызов QRDetector (новый)")
    detector = QRDetector(model_size='s')
    # detector = QRDetector()
    image = cv.cvtColor(cv.imread(filename='source_files/qrs7.png'), code=cv.COLOR_BGR2RGB)
    # image = cv.cvtColor(cv.imread(filename='resources/qreader_test_image.jpeg'), code=cv.COLOR_BGR2RGB)
    #
    start_time = time.time()
    detections = detector.detect(image=image, is_bgr=False)
    print("ALL--- %s seconds ---\n" % (time.time() - start_time))
    #
    start_time = time.time()
    detections = detector.detect(image=image, is_bgr=False)
    print("ALL--- %s seconds ---\n" % (time.time() - start_time))
    #
    print("len(detections)", len(detections))
    # _plot_result(image=image, detections=detections)






