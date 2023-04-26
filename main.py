from ultralytics import YOLO
import cv2
import math

def putTextRect(img, text, pos, scale=3, thickness=3, offset=10, border=None):
        color = (0, 0, 255)
        ox, oy = pos
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
        x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        if border is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, border)
        cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/Video 5.mp4")  # For Video

    model = YOLO("pothole.pt")

    classNames = ['pothole']

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence calculation
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.5:
                    putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
