import datetime
import io
from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import PIL.Image as Image
import os
from skimage.metrics import structural_similarity
import imutils
import cv2

app = Flask(__name__)

# Secret key for sessions encryption
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def home():
    return render_template("index.html", title="Image Reader")

@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    if request.method == 'POST':
        start_time = datetime.datetime.now()
        image_data = request.files['file'].read()
        uploaded_image = Image.open(io.BytesIO(image_data)).resize((250,160))
        uploaded_image = np.array(uploaded_image)
        uploaded_image = uploaded_image[:, :, ::-1].copy() # Convert RGB to BGR
        original_image = Image.open('./static/original.jpg').resize((250,160))
        original_image = np.array(original_image)
        original_image = original_image[:, :, ::-1].copy()
        print(type(uploaded_image))

        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Calculate threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw contours on image
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save all output images (if required)
        cv2.imwrite('./Output/image_original.jpg', original_image)
        cv2.imwrite('./Output/image_uploaded.jpg', uploaded_image)
        cv2.imwrite('./Output/image_diff.jpg', diff)
        cv2.imwrite('./Output/image_thresh.jpg', thresh)

        if round(score*100,2) > 90:
            text = "Sucessfully Match...Thank You..."
            print("Found data:", str(round(score*100,2)) + '%' + ' correct')
        else:
            text = "Tempered Pand Card Detected....Froud Image..."

        session['data'] = {
            "text": str(round(score*100,2)) + '%' + ' Match' +'           '+text,
            "time": str((datetime.datetime.now() - start_time).total_seconds())
        }

        return redirect(url_for('result'))

@app.route('/result')
def result():
    if "data" in session:
        data = session['data']
        return render_template(
            "result.html",
            title="Result",
            time=data["time"],
            text=data["text"],
            words=len(data["text"].split(" "))
        )
    else:
        return "Wrong request method."


if __name__ == '__main__':
    # Setup Tesseract executable path
    #pytesseract.pytesseract.tesseract_cmd = r'D:\TesseractOCR\tesseract'
    app.run(debug=True)
