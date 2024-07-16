import mediapipe as mp
import numpy as np
import cv2

class HandTracker():
    def __init__(self, mode=False, model_complexity=1, maxHands=2, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Initialize HandTracker with given parameters
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Define MediaPipe hands module and initialize hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon)
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process hands in the image
        self.results = self.hands.process(imgRGB)

        # If hand landmarks are found, draw them on the image
        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLm, self.mp_hands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, handNo=0, draw=True):
        # Retrieve landmarks for a specific hand from the processed results
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for lm in myHand.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def getUpFingers(self, img):
        # Determine fingers that are up based on landmark positions
        pos = self.getPostion(img, draw=False)
        self.upfingers = []
        if pos:
            # Check each finger's position and append True/False for each
            # thumb
            self.upfingers.append((pos[4][1] < pos[3][1] and (pos[5][0] - pos[4][0] > 10)))
            # index
            self.upfingers.append((pos[8][1] < pos[7][1] and pos[7][1] < pos[6][1]))
            # middle
            self.upfingers.append((pos[12][1] < pos[11][1] and pos[11][1] < pos[10][1]))
            # ring
            self.upfingers.append((pos[16][1] < pos[15][1] and pos[15][1] < pos[14][1]))
            # pinky
            self.upfingers.append((pos[20][1] < pos[19][1] and pos[19][1] < pos[18][1]))
        return self.upfingers
