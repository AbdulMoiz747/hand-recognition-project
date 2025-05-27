import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None  # Variables to store previous coordinates
threshold = 20  # Movement threshold in pixels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    height, width, _ = frame.shape

    # Convert to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            current_x = int(index_finger_tip.x * width)
            current_y = int(index_finger_tip.y * height)
            
            # If previous position exists, calculate delta
            if prev_x is not None and prev_y is not None:
                delta_x = current_x - prev_x
                delta_y = current_y - prev_y
                
                # Determine dominant movement direction
                if abs(delta_x) > abs(delta_y):
                    if delta_x > threshold:
                        pyautogui.press('right')
                        cv2.putText(frame, "Right", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif delta_x < -threshold:
                        pyautogui.press('left')
                        cv2.putText(frame, "Left", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if delta_y > threshold:
                        pyautogui.press('down')
                        cv2.putText(frame, "Down", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif delta_y < -threshold:
                        pyautogui.press('up')
                        cv2.putText(frame, "Up", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Update previous coordinates
            prev_x, prev_y = current_x, current_y

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
threshold = 30          # Increase the threshold to avoid false triggers
cooldown = 0.5          # 500 ms cooldown between key presses
last_press_time = 0

# Buffer to average movement deltas over the last few frames
delta_buffer = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame for natural interaction
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get current index finger tip position (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            current_x = int(index_tip.x * width)
            current_y = int(index_tip.y * height)
            
            if prev_x is not None and prev_y is not None:
                # Calculate deltas
                delta_x = current_x - prev_x
                delta_y = current_y - prev_y
                
                # Add current delta to buffer and compute average
                delta_buffer.append((delta_x, delta_y))
                avg_delta_x = sum([dx for dx, _ in delta_buffer]) / len(delta_buffer)
                avg_delta_y = sum([dy for _, dy in delta_buffer]) / len(delta_buffer)
                
                current_time = time.time()
                if current_time - last_press_time > cooldown:
                    # Determine dominant movement direction using average delta
                    if abs(avg_delta_x) > abs(avg_delta_y):
                        if avg_delta_x > threshold:
                            pyautogui.press('right')
                            last_press_time = current_time
                            cv2.putText(frame, "Right", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        elif avg_delta_x < -threshold:
                            pyautogui.press('left')
                            last_press_time = current_time
                            cv2.putText(frame, "Left", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        if avg_delta_y > threshold:
                            pyautogui.press('down')
                            last_press_time = current_time
                            cv2.putText(frame, "Down", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        elif avg_delta_y < -threshold:
                            pyautogui.press('up')
                            last_press_time = current_time
                            cv2.putText(frame, "Up", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Update previous coordinates for the next iteration
            prev_x, prev_y = current_x, current_y

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()