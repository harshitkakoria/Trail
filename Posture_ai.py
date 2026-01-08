import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

GOOD_POSTURE_Y = 250  
SLOUCH_THRESHOLD = 70
POSTURE_STATE = "CALIBRATING"
TIMER_STARTED = False
SLOUCH_TIME_SECONDS = 0.0

cap = cv2.VideoCapture(0)

print("Starting Posture AI... Press 'c' to calibrate. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    
    image_h, image_w, _ = image.shape
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_landmark = face_landmarks.landmark[1]
            nose_y = int(nose_landmark.y * image_h)
            
            if POSTURE_STATE == "CALIBRATING":
                cv2.putText(image, "SIT UP STRAIGHT AND PRESS 'C'", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif POSTURE_STATE == "MONITORING":
                cv2.line(image, (0, GOOD_POSTURE_Y), (image_w, GOOD_POSTURE_Y), (0, 255, 0), 2)
                cv2.line(image, (0, GOOD_POSTURE_Y + SLOUCH_THRESHOLD), (image_w, GOOD_POSTURE_Y + SLOUCH_THRESHOLD), (0, 0, 255), 2)

                if nose_y > (GOOD_POSTURE_Y + SLOUCH_THRESHOLD):
                    POSTURE_TEXT = "SLOUCHING"
                    TEXT_COLOR = (0, 0, 255)
                    if not TIMER_STARTED:
                        TIMER_STARTED = True
                        SLOUCH_START_TIME = time.time()
                    
                    SLOUCH_TIME_SECONDS = int(time.time() - SLOUCH_START_TIME)
                    cv2.putText(image, f"Time: {SLOUCH_TIME_SECONDS}s", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    POSTURE_TEXT = "GOOD"
                    TEXT_COLOR = (0, 255, 0)
                    TIMER_STARTED = False
                    SLOUCH_TIME_SECONDS = 0.0

                cv2.putText(image, POSTURE_TEXT, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

    cv2.imshow('AI Posture Monitor', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'): 
        if results.multi_face_landmarks:
            nose_y = int(results.multi_face_landmarks[0].landmark[1].y * image_h)
            GOOD_POSTURE_Y = nose_y
            POSTURE_STATE = "MONITORING"
            print(f"Calibration complete! Good posture Y-level set to: {GOOD_POSTURE_Y}")

cap.release()
cv2.destroyAllWindows()
face_mesh.close()