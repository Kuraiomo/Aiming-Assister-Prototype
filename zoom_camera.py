import cv2
import numpy as np

# Set the fixed focus point (e.g., focus on a point at 30% from the left and 40% from the top)
focus_x, focus_y = 0.59, 0.5  # Relative coordinates (0.0 to 1.0)

def detect_movement(frame1, frame2, threshold=500):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference to get the areas with significant changes
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Sum up the number of white pixels
    movement = np.sum(diff) / 255
    return movement

def zoom(frame, zoom_factor, focus_x, focus_y):
    height, width = frame.shape[:2]
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    center_x = int(focus_x * width)
    center_y = int(focus_y * height)
    
    x1 = max(0, center_x - new_width // 2)
    y1 = max(0, center_y - new_height // 2)
    x2 = min(width, x1 + new_width)
    y2 = min(height, y1 + new_height)
    
    # Ensure the cropped area is within the frame
    x1 = max(0, x2 - new_width)
    y1 = max(0, y2 - new_height)
    
    # Crop the image to the zoomed region
    frame_zoomed = frame[y1:y2, x1:x2]
    # Resize back to the original size
    frame_zoomed = cv2.resize(frame_zoomed, (width, height))
    
    return frame_zoomed

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
zoom_factor = 1.0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    movement = detect_movement(frame1, frame2)
    
    if movement < 1000:
        zoom_factor = min(zoom_factor + 0.01, 5.0)  # Zoom in slower
    elif movement > 500:
        zoom_factor = max(zoom_factor - 0.02, 1.4)  # Zoom out slower
    
    frame_zoomed = zoom(frame2, zoom_factor, focus_x, focus_y)
    
    # Draw a red dot in the middle of the display screen
    center_x = frame_zoomed.shape[1] // 2
    center_y = frame_zoomed.shape[0] // 2
    cv2.circle(frame_zoomed, (center_x, center_y), 5, (0, 0, 255), -1)
    
    cv2.imshow('Zoomed Frame', frame_zoomed)
    
    frame1 = frame2
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

