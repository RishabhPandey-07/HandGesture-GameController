import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)

last_gesture = None
cooldown = 0.3
last_time = 0

def detect_gesture(lm_list):
    ix, iy = lm_list[8]
    ibx, iby = lm_list[6]

    mx, my = lm_list[12]
    mbx, mby = lm_list[10]

    side_thresh = 35
    up_thresh = 35

    # TWO fingers up → DOWN
    if (iby - iy) > up_thresh and (mby - my) > up_thresh:
        return "DOWN"

    # ONE finger up → UP (jump)
    if (iby - iy) > up_thresh:
        return "UP"

    # ---- FIXED LEFT/RIGHT ----
    if (ix - ibx) > side_thresh:
        return "LEFT"

    if (ibx - ix) > side_thresh:
        return "RIGHT"

    return None

while True:
    ret, frame = cap.read()
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]

        lm_list = []
        for lm in hand.landmark:
            lm_list.append([int(lm.x * w), int(lm.y * h)])

        gesture = detect_gesture(lm_list)

        # Show detected gesture
        cv2.putText(frame, str(gesture), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- ONE-TIME TRIGGER: Only when NEW gesture appears ---
        if gesture != last_gesture and gesture is not None:
            if time.time() - last_time > cooldown:

                if gesture == "LEFT":
                    pyautogui.press("left")
                elif gesture == "RIGHT":
                    pyautogui.press("right")
                elif gesture == "UP":
                    pyautogui.press("up")
                elif gesture == "DOWN":
                    pyautogui.press("down")

                last_time = time.time()

        last_gesture = gesture

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
