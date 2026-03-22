import base64
import json
import sys
import traceback
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pytesseract
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from lxml import html
from PIL import Image
from portalocker import lock, LOCK_EX, LOCK_NB, unlock
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from skpartial.pipeline import make_partial_pipeline

app = Flask(__name__)
CORS(app, origins=["*"])

model_path_str = str(
    Path("TextModel.pkl"))
lock_path_str = str(
    Path("TextModel.lock"))

KNOWN_ERROR_KEYWORDS = ["404", "500", "502", "Page not found", "Service unavailable", "Error"]

# Update the lock functions to use the file object directly
def acquire_lock():
    with open(lock_path_str, 'wb') as lock_file:
        lock(lock_file, LOCK_EX | LOCK_NB)

def release_lock():
    with open(lock_path_str, 'wb') as lock_file:
        unlock(lock_file)

# Update the model functions to use the file object directly
def load_model():
    try:
        acquire_lock()
        with open(model_path_str, 'rb') as model_file:
            loaded_model = joblib.load(model_file)
        return loaded_model
    except Exception as e:
        return str(e)
    finally:
        release_lock()

def save_model(loaded_model):
    try:
        acquire_lock()
        with open(model_path_str, 'wb') as model_file:
            joblib.dump(loaded_model, model_file)
    except Exception as e:
        return str(e)
    finally:
        release_lock()

def train_model():
    try:
        # Acquire a lock to ensure exclusive access to the model file during training
        acquire_lock()

        r = requests.get(
            '<BASE_API_URL>'
            '/GetTrainingData')
        json_object = json.loads(r.text)
        df = pd.DataFrame(json_object, columns=['failureClassification', 'screenshot', 'message'])

        X = df['message'].values
        Y = df['failureClassification'].values

        df['extracted_text'] = df['screenshot'].apply(
            lambda img_path: pytesseract.image_to_string(Image.open(BytesIO(base64.b64decode(img_path)))))

        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
        model = load_model()
        # model = make_partial_pipeline(
        #     HashingVectorizer(alternate_sign=False),
        #     MultinomialNB()
        # )

        model.partial_fit(X_train, y_train,
                          classes=['Automation Issue', 'Application Issue', 'Environment Issue', 'UNKNOWN'])
        save_model(model)
        pass
    except Exception as e:
        return str(e)
    finally:
        # Release the lock when done
        release_lock()

def predict(MESSAGE, IMAGE):
    try:
        # Define class priority (higher in list = higher priority)
        priority = ['Environment Issue','Application Issue','Automation Issue','UNKNOWN']

        # Decode and load image
        image_data = base64.b64decode(IMAGE)
        image = Image.open(BytesIO(image_data))

        # Extract text from image
        IMAGE_MESSAGE = pytesseract.image_to_string(image)

        # Load model
        loaded_model = load_model()

        # Set probability threshold
        threshold = 0.60

        # Store valid predictions from message and image
        valid_predictions = []

        # Predict from MESSAGE
        y_pred_prob_message = loaded_model.predict_proba([MESSAGE])[0]
        for c, p in enumerate(y_pred_prob_message):
            if p >= threshold:
                label = list(loaded_model.classes_)[c]
                valid_predictions.append((label, p))

        # Predict from IMAGE_MESSAGE (OCR text)
        y_pred_prob_image = loaded_model.predict_proba([IMAGE_MESSAGE])[0]
        for c, p in enumerate(y_pred_prob_image):
            if p >= threshold:
                label = list(loaded_model.classes_)[c]
                valid_predictions.append((label, p))

        # Prioritize predictions based on defined class order
        if valid_predictions:
            # Sort first by priority index, then by probability (desc)
            valid_predictions.sort(
                key=lambda x: (priority.index(x[0]) if x[0] in priority else len(priority), -x[1])
            )
            return valid_predictions[0][0]

        # Fallback message if no confident prediction
        return "Unable to predict with the provided information. Please provide more detailed information."

    except Exception as e:
        return str(e)

def predictWithMessage(MESSAGE):
    try:
        data = {'MESSAGE': MESSAGE}

        df = pd.DataFrame([data])
        X = df['MESSAGE'].values

        loaded_model = load_model()
        y_pred__prob_text = loaded_model.predict_proba(X)
        threshold = 0.60

        y_pred_text = []

        for probs in y_pred__prob_text:
            max_prob_index = None
            max_prob = 0

            for c, p in enumerate(probs):
                if p >= threshold and p > max_prob:  # Using threshold and ensuring we're picking the max probability
                    max_prob = p
                    max_prob_index = c

            y_pred_text.append(max_prob_index)

        # Ensure we return some meaningful value
        if None in y_pred_text:
            result_text = "Unable to predict with the provided error message. Please provide more detailed information."
        else:
            result_text = list(loaded_model.predict(X))[0]

        return result_text

    except Exception as e:
        return str(e)

def predictWithImage(IMAGE):
    try:
        # Decode and load image
        image_data = base64.b64decode(IMAGE)
        image = Image.open(BytesIO(image_data))

        # Extract text from the image using OCR
        IMAGE_MESSAGE = pytesseract.image_to_string(image)

        data = {'MESSAGE': IMAGE_MESSAGE}

        df = pd.DataFrame([data])
        X = df['MESSAGE'].values

        loaded_model = load_model()
        y_pred__prob_text = loaded_model.predict_proba(X)
        threshold = 0.60

        y_pred_text = []

        for probs in y_pred__prob_text:
            max_prob_index = None
            max_prob = 0

            for c, p in enumerate(probs):
                if p >= threshold and p > max_prob:  # Using threshold and ensuring we're picking the max probability
                    max_prob = p
                    max_prob_index = c

            y_pred_text.append(max_prob_index)

        # Ensure we return some meaningful value
        if None in y_pred_text:
            result_text = "Unable to predict with the provided error message. Please provide more detailed information."
        else:
            result_text = list(loaded_model.predict(X))[0]

        return result_text

    except Exception as e:
        return str(e)

def learnWithMessage(MESSAGE=None, CLASSIFICATION=None):
    try:
        if not CLASSIFICATION:
            raise ValueError("CLASSIFICATION is mandatory.")

        if not MESSAGE:
            raise ValueError("MESSAGE should be provided.")

        df = pd.DataFrame()

        if MESSAGE:
            df = pd.concat([df, pd.DataFrame({"MESSAGE": [MESSAGE], "CLASSIFICATION": [CLASSIFICATION]})],
                           ignore_index=True)
        loaded_model = load_model()

        X = df['MESSAGE'].values
        Y = df['CLASSIFICATION'].values

        loaded_model.partial_fit(X, Y,
                                 classes=['Automation Issue', 'Application Issue', 'Environment Issue', 'UNKNOWN'])
        save_model(loaded_model)

        status = "Model Trained successfully with error message : " + MESSAGE
        return status

    except Exception as e:
        return str(e)

def learnWithImage(IMAGE=None, CLASSIFICATION=None):
    try:
        if not CLASSIFICATION:
            raise ValueError("CLASSIFICATION is mandatory.")

        if not IMAGE:
            raise ValueError("IMAGE encoded string should be provided.")

        df = pd.DataFrame()

        if IMAGE:
            # Decode and load image
            image_data = base64.b64decode(IMAGE)
            image = Image.open(BytesIO(image_data))

            # Extract text from the image using OCR
            image_message = pytesseract.image_to_string(image)
            df = pd.concat([df, pd.DataFrame({"MESSAGE": [image_message], "CLASSIFICATION": [CLASSIFICATION]})],
                           ignore_index=True)

        loaded_model = load_model()

        X = df['MESSAGE'].values
        Y = df['CLASSIFICATION'].values

        loaded_model.partial_fit(X, Y,
                                 classes=['Automation Issue', 'Application Issue', 'Environment Issue', 'UNKNOWN'])
        save_model(loaded_model)

        status = "Model Trained successfully with the image encoded string"
        return status

    except Exception as e:
        return str(e)
    
def is_element_in_page(element_img_bytes, full_img_bytes, threshold=0.9):
    try:
        full_img = np.array(Image.open(BytesIO(full_img_bytes)).convert("RGB"))
        element_img = np.array(Image.open(BytesIO(element_img_bytes)).convert("RGB"))
        result = cv2.matchTemplate(full_img, element_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_val >= threshold, max_val, max_loc
    except Exception as e:
        return None, 0.0, None

def detect_known_error_text(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        for keyword in KNOWN_ERROR_KEYWORDS:
            if keyword.lower() in text.lower():
                return True, keyword
        return False, ""
    except:
        return False, ""

def is_xpath_in_dom(dom_string, xpath):
    try:
        tree = html.fromstring(dom_string)
        elements = tree.xpath(xpath)
        return len(elements) > 0
    except:
        return False

def triage_classification(element_img, full_img, dom_html, xpath):
    match, score, loc = is_element_in_page(element_img, full_img)
    xpath_found = is_xpath_in_dom(dom_html, xpath)
    error_detected, error_keyword = detect_known_error_text(full_img)

    # Reasoning logic
    if error_detected:
        return  f"UI displays known error text ('{error_keyword}'), indicating a backend or application issue."

    if match and not xpath_found:
        return "Element is visible on screen but not found in the DOM using provided XPath. This suggests the XPath is outdated or incorrect."

    if not match and xpath_found:
        return  "Element is found in DOM but not visible in screenshot. Likely the element is off-screen, covered, or delayed — often due to automation timing issues."

    if not match and not xpath_found:
        return  "Element is neither in the DOM nor visible in screenshot. Suggests UI or frontend removed it due to a change or failure."

    if match and xpath_found:
        return "Element is present visually and in the DOM. No failure symptoms observed."

    return  "Unable to confidently classify. More context or logs may be required."

@app.route('/train', methods=['POST'])
def train_api():
    try:
        train_model()
        return jsonify({"message": "Model trained successfully"}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    try:
        result = predict(data['MESSAGE'], data['IMAGE'])
        return jsonify({"prediction": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route('/predictWithMessage', methods=['POST'])
def predictWithMessage_api():
    data = request.get_json()
    try:
        result = predictWithMessage(data['MESSAGE'])
        return jsonify({"prediction": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route('/predictWithImage', methods=['POST'])
def predictWithImage_api():
    data = request.get_json()
    try:
        result = predictWithImage(data['IMAGE'])
        return jsonify({"prediction": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route('/learnWithMessage', methods=['POST'])
def learnWithMessage_api():
    data = request.get_json()
    try:
        result = learnWithMessage(data['MESSAGE'], data['CLASSIFICATION'])
        return jsonify({"Training Status": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route('/learnWithImage', methods=['POST'])
def learnWithImage_api():
    data = request.get_json()
    try:
        result = learnWithImage(data['IMAGE'], data['CLASSIFICATION'])
        return jsonify({"Training Status": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500
    
@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.json

        # Basic validation
        required_keys = ["elementImage", "fullPageImage", "domHTML", "xpath"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400

        # Extract and decode base64 data
        element_img_bytes = base64.b64decode(data["elementImage"])
        full_img_base64 = data["fullPageImage"]
        full_img_bytes = base64.b64decode(full_img_base64)
        dom_html = base64.b64decode(data["domHTML"]).decode("utf-8")
        xpath = data["xpath"]
        message = data.get("message", "")

        print(message)
        print(full_img_base64)
        # Step 1: ML prediction (based on message + image)
        label_ml = predict(message, full_img_base64)
        print(f"ML Prediction: {label_ml}")
        # Step 2: Triage rule-based logic for explanation
        reason = triage_classification(element_img_bytes, full_img_bytes, dom_html, xpath)

        # Combine both into the response
        return jsonify({
            "ml_prediction": label_ml,
            "description": reason
        }), 200

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
