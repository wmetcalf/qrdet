"""
This class the YOLOv7 QR Detector. It uses a YOLOv7-tiny model trained to detect QR codes in the wild.

Author: Eric Canas.
Github: https://github.com/Eric-Canas/qrdet
Email: eric@ericcanas.com
Date: 11-12-2022
"""

from __future__ import annotations

import math
import numpy as np
import cv2 as cv
# from PIL import Image, ImageDraw

import os
import requests
import time
import tqdm

from ultralytics import YOLO
import onnxruntime as ort

# from yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid
from utils import xywh2xyxy, nms, draw_detections, sigmoid

#
import qrdet
from qrdet import _yolo_v8_results_to_dict, _prepare_input
from qrdet import BBOX_XYXY, BBOX_XYXYN, POLYGON_XY, POLYGON_XYN, \
    CXCY, CXCYN, WH, WHN, IMAGE_SHAPE, CONFIDENCE, PADDED_QUAD_XY, PADDED_QUAD_XYN, \
    QUAD_XY, QUAD_XYN

from quadrilateral_fitter import QuadrilateralFitter

_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), '.model')
_CURRENT_RELEASE_TXT_FILE = os.path.join(_WEIGHTS_FOLDER, 'current_release.txt')
_WEIGHTS_URL_FOLDER = 'https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release'
_MODEL_FILE_NAME = 'qrdet-{size}.pt'


# #############################################################
class QRDetector:
    def __init__(self, model_size: str = 's', conf_th: float = 0.5, nms_iou: float = 0.3):
        """
        Инициализация QRDetector'а

        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param conf_th: float. The confidence threshold to use for the detections. Detection with a confidence lower
                                than this value will be discarded. Default: 0.5.
        :param nms_iou: float. The IoU threshold to use for the Non-Maximum Suppression. Detections with an IoU higher
                                than this value will be discarded. Default: 0.3.
        """
        assert model_size in ('n', 's', 'm', 'l'), f'Invalid model size: {model_size}. ' \
                                                   f'Valid values are: \'n\', \'s\', \'m\' or \'l\'.'

        self._model_size = model_size
        path = f'models/qrdet-{self._model_size}.onnx'
        assert os.path.exists(path), f'Could not find model weights at {path}.'

        # EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        EP_list = ['CPUExecutionProvider']
        self.model = ort.InferenceSession(path, providers=EP_list)

        self._conf_th = conf_th
        self._nms_iou = nms_iou

        # ==========================================
        self.conf_threshold = conf_th
        self.iou_threshold = nms_iou
        self.num_masks = 32
        self.img_height, self.img_width = 0, 0

        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]


    def detect(self, image: np.ndarray|'PIL.Image'|'torch.Tensor'|str, is_bgr: bool = False,
               **kwargs) -> tuple[dict[str, np.ndarray|float|tuple[float, float]]]:
        """
        Detect QR codes in the given image.

        :param image: str|np.ndarray|PIL.Image|torch.Tensor. Numpy array (H, W, 3), Tensor (1, 3, H, W), or
                                            path/url to the image to predict. 'screen' for grabbing a screenshot.
        :param is_bgr: input image in BGR format
        :return: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
            - 'expanded_quadrilateral_xy': np.ndarray. An expanded version of quadrilateral_xy, with shape (4, 2),
                that always include all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        start_time_pred = time.time()
        # Любое изображение приводится к numpy
        img = _prepare_input(source=image, is_bgr=is_bgr)
        img_height, img_width = img.shape[:2]


        # ============================
        self.img_height, self.img_width = img.shape[:2]


        # Blob
        input = qrdet.get_blob(img)
        # input = cv.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=False)
        print("  PredObr--- %s seconds ---" % (time.time() - start_time_pred))

        # Predict
        start_time = time.time()
        outputs = self.model.run(None, {"images": input})
        print("  Run--- %s seconds ---" % (time.time() - start_time))

        start_time_new = time.time()
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        print("  NEW boxes & masks --- %s seconds ---" % (time.time() - start_time_new))

        start_time_old = time.time()
        output0 = outputs[0].astype("float")
        output0 = output0[0].transpose()

        output1 = outputs[1].astype("float")
        output1 = output1[0]

        boxes = qrdet.get_boxes(output0, output1)
        print("  OLD boxes & masks --- %s seconds ---" % (time.time() - start_time_old))



        # parse and filter detected objects
        objects = []
        for row in boxes:
            prob = row[4:5].max()  # 84
            if prob < self._conf_th:
                continue
            class_id = row[4:5].argmax()  # 84
            # label = yolo_classes[class_id]
            label = qrdet.custom_classes[class_id]
            #
            xc, yc, w, h = row[:4]
            x1 = (xc - w / 2) / 640 * img_width
            y1 = (yc - h / 2) / 640 * img_height
            x2 = (xc + w / 2) / 640 * img_width
            y2 = (yc + h / 2) / 640 * img_height

            mask = qrdet.get_mask(row[5:25684], (x1, y1, x2, y2), img_width, img_height)  # 84
            polygon = qrdet.get_polygon(mask)
            objects.append([x1, y1, x2, y2, label, prob, polygon])

        objects.sort(key=lambda x: x[5], reverse=True)
        # print(len(objects))
        exit(77)

        results = []
        while len(objects) > 0:
            results.append(objects[0])
            objects = [object for object in objects if qrdet.iou(object, objects[0]) < self._nms_iou]
        # print(len(results))

        #
        if len(results) == 0:
            return []

        im_h, im_w = img_height, img_width
        detections = []
        #
        for result in results:
            # print(result)
            x1, y1, x2, y2 = result[:4]
            confidence = result[5]
            bbox_xyxy = np.array([x1, y1, x2, y2])
            bbox_xyxyn = np.array([x1 / im_w, y1 / im_h, x2 / im_w, y2 / im_h])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cxn, cyn = cx / im_w, cy / im_h
            bbox_w, bbox_h = x2 - x1, y2 - y1
            bbox_wn, bbox_hn = bbox_w / im_w, bbox_h / im_h

            #
            polygon = result[6]
            accurate_polygon_xyn = []
            for point in polygon:
                # print(point)
                point[0] = point[0] + x1
                point[1] = point[1] + y1
                #
                accurate_polygon_xyn.append([point[0] / im_w, point[1] / im_h])
            #
            accurate_polygon_xy = np.array(polygon)
            accurate_polygon_xyn = np.array(accurate_polygon_xyn)
            # print("polygon", accurate_polygon_xy.shape, accurate_polygon_xy)
            # print("polygon", accurate_polygon_xyn.shape, accurate_polygon_xyn)

            # Fit a quadrilateral to the polygon (Don't clip accurate_polygon_xy yet, to fit the quadrilateral before)
            _quadrilateral_fit = QuadrilateralFitter(polygon=accurate_polygon_xy)
            quadrilateral_xy = _quadrilateral_fit.fit(simplify_polygons_larger_than=8,
                                                      start_simplification_epsilon=0.1,
                                                      max_simplification_epsilon=2.,
                                                      simplification_epsilon_increment=0.2)

            # Clip the data to make sure it's inside the image
            np.clip(bbox_xyxy[::2], a_min=0., a_max=im_w, out=bbox_xyxy[::2])
            np.clip(bbox_xyxy[1::2], a_min=0., a_max=im_h, out=bbox_xyxy[1::2])
            np.clip(bbox_xyxyn, a_min=0., a_max=1., out=bbox_xyxyn)

            np.clip(accurate_polygon_xy[:, 0], a_min=0., a_max=im_w, out=accurate_polygon_xy[:, 0])
            np.clip(accurate_polygon_xy[:, 1], a_min=0., a_max=im_h, out=accurate_polygon_xy[:, 1])
            np.clip(accurate_polygon_xyn, a_min=0., a_max=1., out=accurate_polygon_xyn)

            # NOTE: We are not clipping the quadrilateral to the image size, because we actively want it to be larger
            # than the polygon. It allows cropped QRs to be fully covered by the quadrilateral with only 4 points.

            expanded_quadrilateral_xy = np.array(_quadrilateral_fit.expanded_quadrilateral, dtype=np.float32)
            quadrilateral_xy = np.array(quadrilateral_xy, dtype=np.float32)

            expanded_quadrilateral_xyn = expanded_quadrilateral_xy / (im_w, im_h)
            quadrilateral_xyn = quadrilateral_xy / (im_w, im_h)

            #
            detections.append({
                CONFIDENCE: confidence,

                BBOX_XYXY: bbox_xyxy,
                BBOX_XYXYN: bbox_xyxyn,
                CXCY: (cx, cy), CXCYN: (cxn, cyn),
                WH: (bbox_w, bbox_h), WHN: (bbox_wn, bbox_hn),

                POLYGON_XY: accurate_polygon_xy,
                POLYGON_XYN: accurate_polygon_xyn,
                QUAD_XY: quadrilateral_xy,
                QUAD_XYN: quadrilateral_xyn,
                PADDED_QUAD_XY: expanded_quadrilateral_xy,
                PADDED_QUAD_XYN: expanded_quadrilateral_xyn,

                IMAGE_SHAPE: (im_h, im_w),
            })
            # print(detections[-1]['polygon_xy'])
        # qrdet.crop_qr(image=image, detection=detections[0], crop_key=PADDED_QUAD_XYN)
        print("Post--- %s seconds ---" % (time.time() - start_time_post))
        return detections

# ==========================================================================

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv.INTER_CUBIC)

            crop_mask = cv.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes


# #############################################################
class QRDetectorPT:
    def __init__(self, model_size: str = 's', conf_th: float = 0.5, nms_iou: float = 0.3):
        """
        Initialize the QRDetector.
        It loads the weights of the YOLOv8 model and prepares it for inference.
        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param conf_th: float. The confidence threshold to use for the detections. Detection with a confidence lower
                                than this value will be discarded. Default: 0.5.
        :param nms_iou: float. The IoU threshold to use for the Non-Maximum Suppression. Detections with an IoU higher
                                than this value will be discarded. Default: 0.3.
        """
        assert model_size in ('n', 's', 'm', 'l'), f'Invalid model size: {model_size}. ' \
                                                   f'Valid values are: \'n\', \'s\', \'m\' or \'l\'.'
        self._model_size = model_size
        path = self.__download_weights_or_return_path(model_size=model_size)
        assert os.path.exists(path), f'Could not find model weights at {path}.'

        self.model = YOLO(model=path, task='segment')

        self._conf_th = conf_th
        self._nms_iou = nms_iou

    def detect(self, image: np.ndarray|'PIL.Image'|'torch.Tensor'|str, is_bgr: bool = False,
               **kwargs) -> tuple[dict[str, np.ndarray|float|tuple[float, float]]]:
        """
        Detect QR codes in the given image.

        :param image: str|np.ndarray|PIL.Image|torch.Tensor. Numpy array (H, W, 3), Tensor (1, 3, H, W), or
                                            path/url to the image to predict. 'screen' for grabbing a screenshot.
        :param legacy: bool. If sent as **kwarg**, will parse the output to make it identical to 1.x versions.
                            Not Recommended. Default: False.
        :return: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
            - 'expanded_quadrilateral_xy': np.ndarray. An expanded version of quadrilateral_xy, with shape (4, 2),
                that always include all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        image = _prepare_input(source=image, is_bgr=is_bgr)
        # Predict
        results = self.model.predict(source=image, conf=self._conf_th, iou=self._nms_iou, half=False,
                                device=None, max_det=100, augment=False, agnostic_nms=True,
                                classes=None, verbose=False)
        assert len(results) == 1, f'Expected 1 result if no batch sent, got {len(results)}'
        results = _yolo_v8_results_to_dict(results=results[0], image=image)

        if 'legacy' in kwargs and kwargs['legacy']:
            return self._parse_legacy_results(results=results, **kwargs)
        return results

    def _parse_legacy_results(self, results, return_confidences: bool = True, **kwargs) \
            -> tuple[tuple[list[float, float, float, float], float], ...] | tuple[list[float, float, float, float], ...]:
        """
        Parse the results to make it compatible with the legacy version of the library.
        :param results: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. The results to parse.
        """
        if return_confidences:
            return tuple((result[BBOX_XYXY], result[CONFIDENCE]) for result in results)
        else:
            return tuple(result[BBOX_XYXY] for result in results)

    def __download_weights_or_return_path(self, model_size: str = 's', desc: str = 'Downloading weights...') -> None:
        """
        Download the weights of the YoloV8 QR Segmentation model.
        :param model_size: str. The size of the model to download. Can be 's', 'm' or 'l'. Default: 's'.
        :param desc: str. The description of the download. Default: 'Downloading weights...'.
        """
        self.downloading_model = True
        path = os.path.join(_WEIGHTS_FOLDER, _MODEL_FILE_NAME.format(size=model_size))
        if os.path.isfile(path):
            if os.path.isfile(_CURRENT_RELEASE_TXT_FILE):
                # Compare the current release with the actual release URL
                with open(_CURRENT_RELEASE_TXT_FILE, 'r') as file:
                    current_release = file.read()
                # If the current release is the same as the URL, the weights are already downloaded.
                if current_release == _WEIGHTS_URL_FOLDER:
                    self.downloading_model = False
                    return path
        # Create the directory to save the weights.
        elif not os.path.exists(_WEIGHTS_FOLDER):
            os.makedirs(_WEIGHTS_FOLDER)

        url = f"{_WEIGHTS_URL_FOLDER}/{_MODEL_FILE_NAME.format(size=model_size)}"

        # Download the weights.
        from warnings import warn
        warn("QRDetector has been updated to use the new YoloV8 model. Use legacy=True when calling detect "
             "for backwards compatibility with 1.x versions. Or update to new output (new output is a tuple of dicts, "
             "containing several new information (1.x output is accessible through 'bbox_xyxy' and 'confidence')."
             "Forget this message if you are reading it from QReader. "
             "[This is a first download warning and will be removed at 2.1]")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=desc) as progress_bar:
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    file.write(data)
        # Save the current release URL
        with open(_CURRENT_RELEASE_TXT_FILE, 'w') as file:
            file.write(_WEIGHTS_URL_FOLDER)
        # Check the weights were downloaded correctly.
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            # Delete the weights if the download failed.
            os.remove(path)
            raise EOFError('Error, something went wrong while downloading the weights.')

        self.downloading_model = False
        return path

    def __del__(self):
        path = os.path.join(_WEIGHTS_FOLDER, _MODEL_FILE_NAME.format(size=self._model_size))
        # If the weights didn't finish downloading, delete them.
        if hasattr(self, 'downloading_model') and self.downloading_model and os.path.isfile(path):
            os.remove(path)