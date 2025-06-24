import argparse
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

def grade(frame, answer_key, choices=4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = max(cnts, key=cv2.contourArea)

    paper = four_point_transform(frame, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    qCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if 20 < w < 50 and 20 < h < 50 and 0.9 < ar < 1.1:
            qCnts.append(c)

    qCnts = contours.sort_contours(qCnts, method="top-to-bottom")[0]
    correct = 0

    for (q, i) in enumerate(range(0, len(qCnts), choices)):
        cnts_row = contours.sort_contours(qCnts[i:i + choices], method="left-to-right")[0]
        bubbled = None
        max_val = 0
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            if total > max_val:
                max_val = total
                bubbled = j

        if bubbled is not None and answer_key.get(q) == bubbled:
            correct += 1

    score = (correct / float(len(answer_key))) * 100
    print(f"Acertos: {correct}/{len(answer_key)} – Nota: {score:.1f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", required=True,
                    help="Endereço IP do vídeo da câmera do celular (ex: 192.168.1.2:8080)")
    args = vars(ap.parse_args())

    url = f'http://{args["ip"]}/video'
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Erro ao acessar a câmera. Verifique o IP e conexão.")
        return

    # ✏️ Defina aqui o gabarito mestre: {questão_index: alternativa_correta_index}
    ANSWER_KEY = {
        0: 1, 1: 4, 2: 0, 3: 3, 4: 1,
        5: 4, 6: 3, 7: 2, 8: 1, 9: 0  # exemplo com 10 questões
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar quadro.")
            break

        grade(frame, ANSWER_KEY)
        cv2.imshow("Câmera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
