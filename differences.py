import cv2
import numpy as np


def transform_image(frame: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param frame: image at current timestep
    :return: transformed image at current timestep
    """
    frame = cv2.GaussianBlur(src=frame, ksize=(7,7), sigmaX=1.5)
    frame_rgb = frame.copy()
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame, frame_rgb

# def standartization_diff(frame: np.ndarray, prev_frame: np.ndarray, max_possible_sum: int) -> (float, np.ndarray):
#     """
#     :param frame: image at current timestep
#     :param prev_frame: image at previous timestep
#     :param max_possible_sum: max_possible_sum of image with this shape
#     :return: standart absolute difference
#     """
#     return abs(frame - prev_frame).sum() / max_possible_sum, frame

def calculate_histogram(img: np.ndarray) -> np.ndarray:
    hist =  cv2.calcHist(images=[img[np.newaxis, :, :]], channels=[1], mask=None, histSize=[256],
                              ranges=[0, 256])
    cv2.normalize(hist, hist)

    return hist

def histogram_diff(img: np.ndarray, prev_img_hist: np.ndarray, separator: float, alpha: float) -> (float, np.ndarray):
    """
    :param img: img at current timestep
    :param prev_img_hist: img histogram at previous timestep
    :param alpha: coefficient for weighted sum
    :param separator: ...
    :return: correlation between frame and prev_frame, separator, histogram of current frame
    """
    hist_frame = calculate_histogram(img)
    corr_coef = cv2.compareHist(hist_frame, prev_img_hist, cv2.HISTCMP_CORREL)
    separator = alpha * separator + (1-alpha) * (1-corr_coef)
    return corr_coef, separator, hist_frame