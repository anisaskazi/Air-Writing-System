import cv2
import numpy as np
from hand_tracker import HandTracker

# ======================
# SETTINGS
# ======================
WIDTH, HEIGHT = 900, 500
CAM_W, CAM_H = 450, 400

BRUSH = 6
ERASER = 50

ALPHA = 0.3  # smoothing factor

# COLORS
colors = [
    (255, 255, 255),  # white
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255)     # yellow
]

color_index = 0
draw_color = colors[color_index]

tracker = HandTracker()
cap = cv2.VideoCapture(0)

canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0

move_start = None
canvas_copy = None

# ======================
# HELPERS
# ======================
def fingers_up(lm):
    if not lm:
        return [0]*5

    tips = [4, 8, 12, 16, 20]
    fingers = []

    # thumb
    fingers.append(1 if lm[4][1] > lm[3][1] else 0)

    for tip in tips[1:]:
        fingers.append(1 if lm[tip][2] < lm[tip-2][2] else 0)

    return fingers


def get_point(lm, id):
    for i, x, y in lm:
        if i == id:
            return x, y
    return None


def draw_palette(img):
    for i, c in enumerate(colors):
        cv2.circle(img, (50 + i*60, 30), 15, c, -1)


# ======================
# MAIN LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H))

    frame, lm = tracker.get_landmarks(frame)
    fingers = fingers_up(lm)

    index = get_point(lm, 8)
    middle = get_point(lm, 12)
    ring = get_point(lm, 16)

    # ======================
    # COLOR SELECT (1 finger top)
    # ======================
    if index:
        x, y = index

        if y < 60:
            for i in range(len(colors)):
                if 50 + i*60 - 20 < x < 50 + i*60 + 20:
                    color_index = i
                    draw_color = colors[i]

    # ======================
    # DRAW (1 finger)
    # ======================
    if fingers[1] == 1 and fingers[2] == 0:
        if index:
            x, y = index

            smooth_x = int(ALPHA * x + (1-ALPHA)*smooth_x)
            smooth_y = int(ALPHA * y + (1-ALPHA)*smooth_y)

            if prev_x == 0:
                prev_x, prev_y = smooth_x, smooth_y

            cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, BRUSH)

            prev_x, prev_y = smooth_x, smooth_y

    # ======================
    # ERASE (2 fingers)
    # ======================
    elif fingers[1] == 1 and fingers[2] == 1:
        if index:
            x, y = index
            cv2.circle(canvas, (x, y), ERASER, (0, 0, 0), -1)

        prev_x, prev_y = 0, 0

    # ======================
    # MOVE (3 fingers)
    # ======================
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
        if index:
            x, y = index

            if move_start is None:
                move_start = (x, y)
                canvas_copy = canvas.copy()
            else:
                dx = x - move_start[0]
                dy = y - move_start[1]

                M = np.float32([[1, 0, dx], [0, 1, dy]])
                canvas = cv2.warpAffine(canvas_copy, M, (CAM_W, CAM_H))

    else:
        prev_x, prev_y = 0, 0
        move_start = None

    # ======================
    # UI
    # ======================
    draw_palette(frame)

    combined = np.hstack((frame, canvas))
    cv2.imshow("Advanced Air Writing System", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()