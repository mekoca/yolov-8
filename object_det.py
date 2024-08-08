from ultralytics import YOLO
import cv2
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import os
import tempfile


def video_detection(camera_index):
    cap = cv2.VideoCapture(camera_index)
    model = YOLO("../YOLO-Weights/yolov8n.pt")
    translator = Translator()

    # List of class names supported by the model
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    total_detections = 0
    high_conf_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, stream=True)
        for result in results:
            for box in result.boxes:
                try:
                    if box.xyxy.shape[0] == 1:  # Check if there's exactly one detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        class_id = int(box.cls[0])
                        class_name = classNames[class_id]
                        translation = translator.translate(class_name, dest='en').text

                        tts = gTTS(text=translation, lang='en')
                        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as tmpfile:
                            tts.save(tmpfile.name)
                            playsound(tmpfile.name)

                        confidence = box.conf[0]
                        label = f'{class_name} {confidence:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        total_detections += 1
                        if confidence > 0.5:  # Adjust this threshold as needed
                            high_conf_detections += 1
                    else:
                        print("More than one box or no boxes detected.")
                except Exception as e:
                    print(f"Error processing detection: {e}")

        accuracy = (high_conf_detections / total_detections) * 100 if total_detections > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")

        cv2.imshow("YOLOv8 Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_detection(0)  # 0 for default webcam
