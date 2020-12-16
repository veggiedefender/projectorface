import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def cull_landmarks(landmarks):
    """
    Remove facial landmarks around eyes and mouth, except for corners, so that they are placed in
    the right locations but can still make facial expressions
    """
    return np.delete(landmarks, np.r_[
        49:54, 55:68, # Mouth
        37:39, 40:42, # Left eye
        43:45, 46:48, # Right eye
    ], axis=0)

with open('coordinates.txt') as f:
    target_landmarks = np.array([list(map(int, map(float, line.split(' ')))) for line in f.read().splitlines()])
    target_landmarks = cull_landmarks(target_landmarks)
    face_tri_indices = Delaunay(target_landmarks).simplices
    target_tris = target_landmarks[face_tri_indices]

cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    morphed_frame = np.ones(frame.shape, dtype=frame.dtype) * 255

    faces = detector(gray, 0)
    if len(faces) > 0:
        # Find and triangulate facial landmarks, and warp those triangles so they match the
        # landmarks on the target (foam head).
        # Heavily inspired by https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
        landmarks = np.array([[p.x, p.y] for p in predictor(gray, faces[0]).parts()])
        landmarks = cull_landmarks(landmarks)
        source_tris = landmarks[face_tri_indices]
        for src_tri, tgt_tri in zip(source_tris, target_tris):
            src_x, src_y, src_w, src_h = cv2.boundingRect(np.float32(src_tri))
            tgt_x, tgt_y, tgt_w, tgt_h = cv2.boundingRect(np.float32(tgt_tri))

            src_tri_relative = src_tri - [src_x, src_y]
            tgt_tri_relative = tgt_tri - [tgt_x, tgt_y]

            warp_matrix = cv2.getAffineTransform(np.float32(src_tri_relative), np.float32(tgt_tri_relative))
            warped = cv2.warpAffine(frame[src_y:src_y+src_h, src_x:src_x+src_w], warp_matrix, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask = np.zeros((tgt_h, tgt_w, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, tgt_tri_relative, (1.0, 1.0, 1.0))

            morphed_frame[tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w] = (
                morphed_frame[tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w] * (1 - mask) +
                warped * mask
            )

    cv2.imshow('frame', morphed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
