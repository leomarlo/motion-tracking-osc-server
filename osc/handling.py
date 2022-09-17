import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class MediaPipe:
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 
    
    cap: cv2.VideoCapture

    def startCapture(*args):
        MediaPipe.cap = cv2.VideoCapture(0)
    
    def stopCapture():
        MediaPipe.cap.release()

    def handleCapture(with_drawing_landmarks=True):
        while MediaPipe.cap.isOpened():
            success, image = MediaPipe.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = MediaPipe.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if with_drawing_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            try:
                newData = {coord + str(j):lm.__getattribute__(coord)  for j, lm in enumerate(results.pose_landmarks.landmark) for coord in ["x", "y"]}
                # PoseData.append(newData)
                print('x0', newData['x0'])
            except Exception as e:
                print(str(e))

    
    
