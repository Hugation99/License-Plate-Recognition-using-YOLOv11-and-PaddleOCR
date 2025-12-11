from logging import config
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO


def main():

    VIDEO_PATH = './assets/pexels-george-morina-5222550-2160p.mp4'

    # Load the YOLOv11 model for license plate detection
    model = YOLO('yolov11-license-plate.pt')
    CONFIDENCE_THRESHOLD = 0.5

    # Initialize PaddleOCR for license plate recognition
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


    cap = cv2.VideoCapture(VIDEO_PATH)
    out = cv2.VideoWriter('results_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read() #returns image array 
        if not ret:
            break

        results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #make sure bbox is within image bounds
                h, w, _ = frame.shape
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                cropped_image = frame[y1:y2, x1:x2] #crop license plate
                
                ocr_results = ocr.ocr(cropped_image, cls=True) #recognise text with ocr model
               
                if not result or result[0] is None:  
                    return None, 0.0
                
                best_detection = max(result[0], key=lambda x: x[1][1]) #get ocr result with highest confidence
                detected_text = best_detection[1][0]
                ocr_conf = best_detection[1][1]
                print(f"Detected: {detected_text} (Conf: {ocr_conf:.2f})")

                #display results on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        out.write(frame)
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

