import cv2
import numpy as np

TEMPLATE_THRESHOLD = 0.6
BOARD_FILE = './puyo2.png'

board = cv2.imread(BOARD_FILE)
result = cv2.imread(BOARD_FILE)

colors = {
    'red': {
        'template': cv2.imread('templates/r.png'),
        'bgr': [0, 0, 255]
    },
    'green': {
        'template': cv2.imread('templates/g.png'),
        'bgr': [0, 128, 0]
    },
    'blue': {
        'template': cv2.imread('templates/b.png'),
        'bgr': [255, 0, 0]
    },
    'purple': {
        'template': cv2.imread('templates/p.png'),
        'bgr': [128, 0, 128]
    },
    'ojama': {
        'template': cv2.imread('templates/o.png'),
        'bgr': [0, 0, 0]
    },
}

for color, contents in colors.items():
    template = contents['template']
    match = cv2.matchTemplate(board, template, cv2.TM_CCOEFF_NORMED)
    histogram = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    h, w, c = template.shape
    loc = np.where(match >= TEMPLATE_THRESHOLD)

    for pt in zip(*loc[::-1]):
        cropped = board[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
        cropped_hist = cv2.calcHist([cropped], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        d = cv2.compareHist(histogram, cropped_hist, cv2.HISTCMP_CORREL)
        if d > 0.1:
            cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), contents['bgr'], 2)

cv2.imwrite('result.png', result)
