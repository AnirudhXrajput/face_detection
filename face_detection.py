import cv2
import mediapipe as mp
import numpy as np

def detect_faces():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)  # Allow multiple faces
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        ih, iw, _ = frame.shape
        face_count = 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose_tip = face_landmarks.landmark[1]
                
                left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
                right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
                nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
                
                angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180.0 / np.pi
                
                if abs(angle) > 15:
                    print("Head rotated")
        

        if face_count == 0:
            print("No face detected!")
        elif face_count > 1:
            print(f"Multiple faces detected! ({face_count} faces)")
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
