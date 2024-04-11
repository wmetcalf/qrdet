"""
Модуль с функциями для работы с моделью в формате onnx
"""
import numba
import numpy as np
import cv2 as cv
from PIL import Image
#
from numba import njit, jit, vectorize, float64, stencil
#
custom_classes = ['qrcode']


def get_blob(source_img, input_shape=(640, 640)):
    """
    Подготовка изображения для передачи в модель YOLOv8.

    :param input_shape:
    :param source_img:
    :return: Numpy array in a shape (1, 3, width, height) where 3 is number of color channels
    """
    resized = cv.resize(source_img, input_shape, interpolation=cv.INTER_LINEAR)
    blob = resized.transpose(2, 0, 1).reshape(1, 3, input_shape[0], input_shape[1]).astype('float32')
    return blob / 255.0


def get_mask(row,box, img_width, img_height):
    """
    Function extracts segmentation mask for object in a row
    :param row: Row with object
    :param box: Bounding box of the object [x1,y1,x2,y2]
    :param img_width: Width of original image
    :param img_height: Height of original image
    :return: Segmentation mask as NumPy array
    """
    mask = row.reshape(160,160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype('uint8')*255
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*160)
    mask_y1 = round(y1/img_height*160)
    mask_x2 = round(x2/img_width*160)
    mask_y2 = round(y2/img_height*160)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask


def get_polygon(mask):
    """
    Function calculates bounding polygon based on segmentation mask
    :param mask: Segmentation mask as Numpy Array
    :return:
    """
    contours = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    polygon = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    return polygon


@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@njit(cache=True)
def get_boxes(o0, o1):
    b = o0[:, 0:5]
    m = o0[:, 5:]

    o1 = o1.reshape(32, 160 * 160)

    m = np.ascontiguousarray(m)
    # o1 = np.ascontiguousarray(o1)
    m = np.dot(m, o1)

    # b = np.hstack((b, m))
    b = np.concatenate((b, m), axis=1)
    return b


# @njit
def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


# @njit
def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


# @njit
def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)


# Array of class labels
# yolo_classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
#     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
#     "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
#     "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
#     "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
#     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]

