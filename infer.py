import os
from PIL import Image
import cv2
import face_alignment
import torch
from torchvision import transforms
from tqdm import tqdm
from swapping.blendswap import BlendSwap
import numpy as np
from expdataloader import *
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ffhq_align import image_align, transform_image_to_quad


class Inference:
    def __init__(self, device="cuda"):
        self.device = device
        self._swap_model = None
        self._insight_model = None
        self._landmarks_detector = None

    @property
    def swap_model(self):
        if self._swap_model is None:
            blend_swap = BlendSwap()
            blend_swap.load_state_dict(torch.load("./checkpoints/blendswap.pth", map_location="cpu"))
            blend_swap.eval()
            blend_swap.to(self.device)
            self._swap_model = blend_swap
        return self._swap_model

    @property
    def insight_model(self):
        if self._insight_model is None:
            app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"], allowed_modules=["detection"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._insight_model = app
        return self._insight_model

    @property
    def landmark_detector(self):
        if self._landmarks_detector is None:
            self._landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
        return self._landmarks_detector

    def crop_face(self, img_path):
        img = np.array(Image.open(img_path))
        faces = self.insight_model.get(img)
        assert len(faces) == 1, f"Found {len(faces)} faces"
        warped = face_align.norm_crop(img, faces[0].kps)
        return warped
        # Image.fromarray(warped).save("examples/warped.jpg")
        # print("Saved to examples/warped.jpg")

    def ffhq_align(self, img_path):
        ldmkss = self.landmark_detector.get_landmarks(img_path)
        ldmks = ldmkss[0]
        return image_align(img_path, ldmks)

    def ffhq_unalign(self, src_img: Image.Image, quad: np.ndarray, aligned_img: Image.Image):
        warped, mask = transform_image_to_quad(aligned_img, quad, src_img.size)
        src_img.paste(Image.fromarray(warped), (0, 0), Image.fromarray(mask))
        return src_img

    def swap(self, source_img_path, target_img_path, save_path):
        toTensor = transforms.ToTensor()
        toImage = transforms.ToPILImage()

        source_img = toTensor(self.crop_face(source_img_path)).unsqueeze(0).to(self.device)
        ffhq_target, quad = self.ffhq_align(target_img_path)
        target_img = toTensor(ffhq_target).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.swap_model.forward(source_img=source_img, target_img=target_img)
        result: Image.Image = toImage(output[0])
        result = self.ffhq_unalign(Image.open(target_img_path), quad, result)
        result.save(save_path)
        # print(f"Saved to {save_path}")
        # Image.fromarray((output.permute(0, 2, 3, 1)[0].cpu().data.numpy() * 255).astype(np.uint8)).save(save_path)


class BlendFaceDataLoader(ExpDataLoader):
    model = Inference()

    def __init__(self):
        super().__init__("blendface")

    def run_video(self, source_img_path, target_video_path, out_video_path):
        image_loader = self.get_images_loader(target_video_path)
        pid = get_pid(source_img_path)
        unaligned_dir = get_sub_dir(image_loader.base_dir, f"blendface_{pid}/unalign")
        for target_img_path in tqdm(image_loader.get_image_paths()):
            target_name = os.path.basename(target_img_path)
            save_path = os.path.join(unaligned_dir, target_name)
            self.model.swap(source_img_path, target_img_path, save_path)
        return self.merge_video(unaligned_dir, out_video_path)


class Test:
    infer = Inference()

    def swap(self):
        self.infer.swap("examples/1.jpg", "examples/2.jpg", "examples/1-2.png")

    def crop(self):
        self.infer.crop_face("examples/0.jpg")

    def align(self):
        img, quad = self.infer.ffhq_align("examples/0.jpg")
        img.save("examples/ffhq_aligned.png")
        print("Saved to examples/ffhq_aligned.png")


def main():
    loader = BlendFaceDataLoader()
    loader.run_all()


if __name__ == "__main__":
    main()
