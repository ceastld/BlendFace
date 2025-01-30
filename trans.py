import numpy as np
import cv2
from PIL import Image


def transform_image_to_quad(src_img, quad):
    src_img = np.array(src_img)
    h, w = src_img.shape[:2]
    src_pts = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)
    dst_pts = np.array(quad, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output_img = cv2.warpPerspective(src_img, H, (w, h), flags=cv2.INTER_CUBIC)
    return output_img


def get_mask(src_img: np.ndarray, quad):
    h, w = src_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    quad_int = np.int32(quad)
    cv2.fillPoly(mask, [quad_int], 255, lineType=cv2.LINE_AA)  # 使用抗锯齿
    return mask


def save_image_with_alpha(img, output_path):
    img.save(output_path, "PNG")


def main():
    src_img = Image.open("examples/1.jpg")
    quad = [(150, 70), (400, 100), (400, 400), (100, 400)]
    transformed_img = transform_image_to_quad(src_img, quad)
    save_image_with_alpha(transformed_img, "test.png")


if __name__ == "__main__":
    main()
