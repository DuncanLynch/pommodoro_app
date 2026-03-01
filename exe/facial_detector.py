from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import numpy as np

# Create estimator and calibrate
estimator = GazeEstimator()
run_9_point_calibration(estimator)

# Save model
estimator.save_model("gaze_model.pkl")

# Load model
estimator = GazeEstimator()
estimator.load_model("gaze_model.pkl")

# Create a full-screen black window
screen_width = 1920  # Adjust to match your screen resolution
screen_height = 1080  # Adjust to match your screen resolution
frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Extract features from the frame
    ret, webcam_frame = cap.read()

    # Extract gaze features and blink status
    features, blink = estimator.extract_features(webcam_frame)

    # Predict screen coordinates
    if features is not None and not blink:
        x, y = estimator.predict([features])[0]
        print(f"Gaze: ({x:.0f}, {y:.0f})")
        
        x = np.clip(x, 0, screen_width - 1)
        y = np.clip(y, 0, screen_height - 1)
        
        cv2.circle(frame, (int(x), int(y)), 20, (0, 0, 255), -1)  # Red circle with radius 20
    
    cv2.imshow("Gaze Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame.fill(0)  # Reset the frame to black before drawing the next gaze circle

cap.release()
cv2.destroyAllWindows()