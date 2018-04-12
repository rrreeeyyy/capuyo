import cv2
import numpy as np

board = cv2.imread('./puyo2.png')

templates = {
    'blue': cv2.imread('templates/b.png'),
    'purple': cv2.imread('templates/p.png'),
    'ojama': cv2.imread('templates/o.png'),
    'red': cv2.imread('templates/r.png'),
    'green': cv2.imread('templates/g.png'),
}

detect_color = 'ojama'

match = cv2.matchTemplate(board, templates[detect_color], cv2.TM_CCOEFF_NORMED)
threshold = 0.35

h, w, c = templates[detect_color].shape
loc = np.where(match >= threshold)

for pt in zip(*loc[::-1]):
    cropped = board[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
    blue, green, red = cropped.mean(0).mean(0)
    # if blue > red and blue > green:
    #     cv2.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # if green > blue and green > red:
    #     cv2.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    if blue < 120 and green < 120 and red > 170:
        cv2.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # cv2.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('result.png', board)
cv2.imwrite('cropped.png', cropped)
