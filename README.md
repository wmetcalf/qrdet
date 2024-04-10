# QRDet - onnx

Forked from Eric-Canas/qrdet

Models have been converted to onnx format.

Original detector available by name QRDetectorPT


## Usage

There is only one function you'll need to call to use **QRDet**, ``detect``:

```python

from qrdet import QRDetector
import cv2

detector = QRDetector(model_size='s')
image = cv2.imread(filename='source_files/qreader_test_image.jpeg')
detections = detector.detect(image=image, is_bgr=True)

# Draw the detections
for detection in detections:
    x1, y1, x2, y2 = detection['bbox_xyxy']
    confidence = detection['confidence']
    segmenation_xy = detection['quadrilateral_xy']
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)
# Save the results
cv2.imwrite(filename='source_files/qreader_test_image_detections.jpeg', img=image)
```

