import cv2
import mediapipe as mp
import numpy as np
import time

def detect_faces():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)

    last_face_time = time.time()  
    face_present = False 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        ih, iw, _ = frame.shape
        face_count = 0  

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)  
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                
                
                left_eye = face_landmarks.landmark[33]    
                right_eye = face_landmarks.landmark[263]  
                nose_tip = face_landmarks.landmark[1]     
                chin = face_landmarks.landmark[152]       
                
                left_eye_x, left_eye_y = int(left_eye.x * iw), int(left_eye.y * ih)
                right_eye_x, right_eye_y = int(right_eye.x * iw), int(right_eye.y * ih)
                nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
                chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)

                
                center_face_x = (left_eye_x + right_eye_x) / 2
                face_width = abs(right_eye_x - left_eye_x)

                
                nose_offset = (nose_x - center_face_x) / face_width

                
                if nose_offset < -0.6:  
                    print(f"Face {i+1}: Head Rotated RIGHT")
                elif nose_offset > 0.6:  
                    print(f"Face {i+1}: Head Rotated LEFT")


                angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180.0 / np.pi

                last_face_time = time.time()
        
        
        if face_count == 0:
            if time.time() - last_face_time > 1.0:
                face_present = False
                print("No face detected!")
        else:
            face_present = True
            if face_count > 1:
                print(f"Multiple Faces Detected! ({face_count} faces)")

        cv2.imshow('Face Detection & Head Rotation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
