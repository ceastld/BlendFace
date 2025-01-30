import face_alignment
import numpy as np


def main():
    landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
    ldmkss = landmarks_detector.get_landmarks('examples/1.jpg')
    np.save('examples/1.npy', ldmkss[0])    

if __name__ == '__main__':
    # main()
    ldmks = np.load('examples/1.npy')
    print(ldmks)
    
    