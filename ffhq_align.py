import os

import PIL.ImageTransform
import numpy as np
import argparse
import scipy.ndimage
import PIL.Image
import face_alignment
from torch.autograd.grad_mode import enable_grad
from PIL import Image
import cv2


def ffhq_pad(img, quad, qsize, border):
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]
    return img, quad


def image_align_ori(src_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
    assert os.path.isfile(src_file), f"Cannot find source image: {src_file}"
    quad, qsize = clac_quad(face_landmarks)
    img = PIL.Image.open(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    if enable_padding:
        img, quad = ffhq_pad(img, quad, qsize, border)

    # Transform.
    # img.save('test.png', 'PNG')
    img = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    return img


def clac_quad(face_landmarks):
    lm = np.array(face_landmarks)
    lm_chin = lm[0:17, :2]  # left-right
    lm_eyebrow_left = lm[17:22, :2]  # left-right
    lm_eyebrow_right = lm[22:27, :2]  # left-right
    lm_nose = lm[27:31, :2]  # top-down
    lm_nostrils = lm[31:36, :2]  # top-down
    lm_eye_left = lm[36:42, :2]  # left-clockwise
    lm_eye_right = lm[42:48, :2]  # left-clockwise
    lm_mouth_outer = lm[48:60, :2]  # left-clockwise
    lm_mouth_inner = lm[60:68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return quad, qsize


def get_box(quad):
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    return crop


def image_align(self, src_file, face_landmarks, output_size=256):
    quad, qsize = clac_quad(face_landmarks)
    img = PIL.Image.open(src_file)
    img = img.transform((output_size, output_size), Image.Transform.QUAD, (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    return img


def transform_image_to_quad(src_img: np.ndarray, quad, output_size):
    if not isinstance(src_img, np.ndarray):
        src_img = np.array(src_img)
    w, h = src_img.shape[:2]
    src_pts = np.array([(0, 0), (0, h), (w, h), (w, 0)], dtype=np.float32)
    dst_pts = np.array(quad, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    white_image = np.ones((h, w), dtype=np.uint8) * 255
    output_img = cv2.warpPerspective(src_img, H, output_size, flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(white_image, H, output_size, flags=cv2.INTER_CUBIC)
    kernel_size = 5  # 卷积核大小，控制收缩程度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀操作，收缩掩膜
    return output_img, mask


def image_unalign(src_file, face_landmarks, aligned_file: str):
    img = Image.open(src_file)
    quad, qsize = clac_quad(face_landmarks)
    warped, mask = transform_image_to_quad(Image.open(aligned_file), quad, img.size)
    img.paste(Image.fromarray(warped), (0, 0), Image.fromarray(mask))
    return img


def get_landmarks(img_path="examples/1.jpg", save_path=None):
    if save_path is None:
        save_path = os.path.splitext(img_path)[0] + ".npy"
    if os.path.exists(save_path):
        ldmks = np.load(save_path)
    else:
        landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
        ldmkss = landmarks_detector.get_landmarks("examples/1.jpg")
        ldmks = ldmkss[0]
        np.save(save_path, ldmks)
    return ldmks


def test_align():
    img_path = "examples/1.jpg"
    img = image_align_ori(img_path, get_landmarks(img_path))
    save_path = os.path.splitext(img_path)[0] + "_aligned.jpg"
    img.save(save_path)
    os.system("code " + save_path)


def test_unalign():
    img_path = "examples/1.jpg"
    img = image_unalign(img_path, get_landmarks(img_path), "examples/1_aligned.jpg")
    save_path = os.path.splitext(img_path)[0] + "_unaligned.jpg"
    img.save(save_path)
    os.system("code " + save_path)


if __name__ == "__main__":
    test_unalign()
