# import numpy as np
# import cv2
# import imutils
# import sys
# import pytesseract
# import pandas as pd
# import time

# # ✅ Specify the path to tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load image
# image = cv2.imread('car.jpeg')
# if image is None:
#     print("Error: Image not found or path is incorrect.")
#     sys.exit()

# # Resize image for better processing
# image = imutils.resize(image, width=500)
# cv2.imshow("Original Image", image)

# # Convert to grayscale and filter noise
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)

# # Edge detection
# edged = cv2.Canny(gray, 170, 200)

# # Find contours
# cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

# NumberPlateCnt = None
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     if len(approx) == 4:
#         NumberPlateCnt = approx
#         break

# # Check if plate contour was found
# if NumberPlateCnt is None:
#     print("Number plate contour not found.")
#     sys.exit()

# # Mask the rest of the image except the plate
# mask = np.zeros(gray.shape, np.uint8)
# new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
# new_image = cv2.bitwise_and(image, image, mask=mask)

# # Show the detected plate region
# cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
# cv2.imshow("Final_image", new_image)

# # OCR config and reading
# config = ('-l eng --oem 1 --psm 3')
# text = pytesseract.image_to_string(new_image, config=config).strip()

# # Save result to CSV
# raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'v_number': [text]}
# df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
# df.to_csv('data.csv', index=False)

# # Print recognized text
# print("Detected Number Plate Text:", text)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open video file or webcam (0 for webcam)
video_path = 'car.mp4'  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    NumberPlateCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    if NumberPlateCnt is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Optional: convert to grayscale and threshold to improve OCR accuracy
        plate_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(plate_thresh, config=config).strip()

        if text:
            timestamp = time.asctime(time.localtime(time.time()))
            print(f"Detected Plate: {text} at {timestamp}")
            results.append({'date': timestamp, 'v_number': text})

        # Show the frame with detected plate contour
        cv2.drawContours(frame, [NumberPlateCnt], -1, (0, 255, 0), 3)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save results to CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv('video_plate_data.csv', index=False)
    print("Results saved to video_plate_data.csv")
else:
    print("No plates detected.")
    #C:\Users\asai4\Downloads\Vehicle-Number-Plate-Reading-master\Vehicle-Number-Plate-Reading-master>venv\Scripts\Activate

