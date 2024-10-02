"""
 QRDetector with running onnx model by onnxruntime
"""
from __future__ import annotations
#
import numpy as np
import math
import cv2 as cv
from PIL import Image
#
import os
import time
import pkg_resources
#import onnx
#
import onnxruntime as ort
#from onnxruntime.quantization import quantize_dynamic, QuantType
#
from qrdet.utils import xywh2xyxy, nms, sigmoid
#
import qrdet
from qrdet import _prepare_input
from qrdet import BBOX_XYXY, BBOX_XYXYN, POLYGON_XY, POLYGON_XYN, \
    CXCY, CXCYN, WH, WHN, IMAGE_SHAPE, CONFIDENCE, PADDED_QUAD_XY, PADDED_QUAD_XYN, \
    QUAD_XY, QUAD_XYN
from quadrilateral_fitter import QuadrilateralFitter
#
TIMINGS = False


# ##############################################################
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
        #path = f'models/qrdet-{self._model_size}.onnx'
        #path = f'models/qrdet-n-sim.onnx'
        path = pkg_resources.resource_filename('qrdet','models/qrdet-n-sim-q.onnx')
        assert os.path.exists(path), f'Could not find model weights at {path}.'

        # EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        EP_list = ['CPUExecutionProvider']
        #quantize_dynamic(
        #     model_input=path,
        #     model_output=dynpath,
        #     weight_type=QuantType.QUInt8  # You can also use QuantType.QUInt8
        #)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model = ort.InferenceSession(path, sess_options=session_options, providers=EP_list)
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # print("input_names", self.input_names)
        self.input_shape = model_inputs[0].shape
        # print("input_shape", self.input_shape)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # ==============================================================
        self.conf_threshold = conf_th
        self.iou_threshold = nms_iou
        self.num_masks = 32

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
        start_time_prepare = time.time()

        # Любое изображение приводится к numpy
        img = _prepare_input(source=image, is_bgr=is_bgr)
        self.img_height, self.img_width = img.shape[:2]

        # Blob
        input = qrdet.get_blob(img)
        if TIMINGS:
            print("  Prepare --- %s seconds ---" % (time.time() - start_time_prepare))

        # Predict
        start_time_run = time.time()
        outputs = self.model.run(None, {"images": input})
        if TIMINGS:
            print("  Run --- %s seconds ---" % (time.time() - start_time_run))

        # Boxes & masks
        start_time_unwrap = time.time()
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        if TIMINGS:
            print("  Boxes & masks --- %s seconds ---" % (time.time() - start_time_unwrap))

        start_time_post = time.time()
        results = []
        for idx, row in enumerate(self.boxes):
            #
            x1, y1, x2, y2 = row
            label = qrdet.custom_classes[self.class_ids[idx]]
            prob = self.scores[idx]
            #
            mask_x1 = round(x1)
            mask_y1 = round(y1)
            mask_x2 = round(x2)
            mask_y2 = round(y2)
            curr_mask = self.mask_maps[idx][mask_y1:mask_y2,mask_x1:mask_x2]
            curr_mask = (curr_mask > 0.5).astype('uint8') * 255
            img_mask = Image.fromarray(curr_mask, "L")
            # img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
            mask = np.array(img_mask)
            polygon = qrdet.get_polygon(mask)
            #
            results.append([x1, y1, x2, y2, label, prob, polygon])
        #
        if len(results) == 0:
            return []
        #
        im_h, im_w = self.img_height, self.img_width
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
            if len(accurate_polygon_xy) < 4:
                print(f"A valid polygon requires at least 4 coordinates, but got {len(accurate_polygon_xy)}")
                continue
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
        if TIMINGS:
            print("  Results --- %s seconds ---" % (time.time() - start_time_post))
        return detections

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
        #blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        # Ensure that mask_width and mask_height are greater than zero
        if mask_width > 0 and mask_height > 0:
            blur_size = (max(1, int(self.img_width / mask_width)), max(1, int(self.img_height / mask_height)))
        else:
        # Handle case where mask dimensions are invalid
            blur_size = (1, 1)  # Set a fallback blur size or handle it appropriately
        
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


