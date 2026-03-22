# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:43:22 2023

@author: mayjoshi
"""
import json

import pandas as pd
import requests
from portalocker import lock, LOCK_EX, LOCK_NB, unlock
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from skpartial.pipeline import make_partial_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
import traceback
import sys
import joblib
from pathlib import Path
from flask import Flask, jsonify, request
import base64
from PIL import Image
from io import BytesIO
import pytesseract
import re

from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

model_path_str = str(
    Path("/model/predict//TextModel.pkl"))
lock_path_str = str(
    Path("/model/predict//TextModel.lock"))

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
            '<BASE_API_URI>'
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

def sanitize_message_string(message: str) -> str:
    """
    Fixes unescaped inner JSON objects inside a string and returns a clean message string.
    """
    try:
        # Try putting the message into JSON to check validity
        json.dumps({"MESSAGE": message})
        return message  # Already clean
    except Exception:
        pass  # Try to sanitize

    # Escape inner JSON-like blocks by escaping double quotes
    def escape_inner_json(m):
        inner = m.group(0)
        return inner.replace('"', '\\"')

    # Replace all `{...}` blocks that contain double quotes with escaped versions
    fixed_message = re.sub(r'\{[^{}"]*"[^{}"]*"[^{}"]*\}', escape_inner_json, message)

    return fixed_message



def predict(MESSAGE, IMAGE):
    try:
        MESSAGE = sanitize_message_string(MESSAGE)
        # Decode and load image
        image_data = base64.b64decode(IMAGE)
        image = Image.open(BytesIO(image_data))

        # Extract text from the image using OCR
        IMAGE_MESSAGE = pytesseract.image_to_string(image)

        # Combine message and image text
        combined_message = MESSAGE + ' ' + IMAGE_MESSAGE

        # Load the model
        loaded_model = load_model()

        # Predict for the combined message
        y_pred_prob_combined = loaded_model.predict_proba([combined_message])[0]

        # Set threshold
        threshold = 0.60

        # Check if any class probability for combined message passes the threshold
        max_prob_index_combined = None
        max_prob_combined = 0
        for c, p in enumerate(y_pred_prob_combined):
            if p >= threshold and p > max_prob_combined:
                max_prob_combined = p
                max_prob_index_combined = c

        if max_prob_index_combined is not None:
            result_combined = list(loaded_model.classes_)[max_prob_index_combined]

            # Check if the result from the combined message is an environment issue
            if result_combined == "Environment Issue":
                return result_combined

        # If the combined message doesn't pass the threshold or is not an environment issue,
        # fallback to individual predictions

        # Predict for the original message
        y_pred_prob_message = loaded_model.predict_proba([MESSAGE])[0]

        # Predict for the image text
        y_pred_prob_image = loaded_model.predict_proba([IMAGE_MESSAGE])[0]

        # Check if any class probability for message passes the threshold
        max_prob_index_message = None
        max_prob_message = 0
        for c, p in enumerate(y_pred_prob_message):
            if p >= threshold and p > max_prob_message:
                max_prob_message = p
                max_prob_index_message = c

        if max_prob_index_message is not None:
            result_message = list(loaded_model.classes_)[max_prob_index_message]
            return result_message

        # Check if any class probability for image text passes the threshold
        max_prob_index_image = None
        max_prob_image = 0
        for c, p in enumerate(y_pred_prob_image):
            if p >= threshold and p > max_prob_image:
                max_prob_image = p
                max_prob_index_image = c

        if max_prob_index_image is not None:
            result_image = list(loaded_model.classes_)[max_prob_index_image]
            return result_image

        # If none of the predictions pass the threshold, return an error message
        return "Unable to predict with the provided information. Please provide more detailed information."

    except Exception as e:
        return str(e)

def predictWithMessage(MESSAGE):
    try:
        MESSAGE = sanitize_message_string(MESSAGE)
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
        MESSAGE = sanitize_message_string(MESSAGE)
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


@app.route('/train', methods=['POST'])
def train_api():
    try:
        train_model()
        return jsonify({"message": "Model trained successfully"}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route('/predict', methods=['GET'])
def predict_api():
    data = request.get_json()
    try:
        result = predict(data['MESSAGE'], data['IMAGE'])
        return jsonify({"prediction": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route('/predictWithMessage', methods=['GET'])
def predictWithMessage_api():
    data = request.get_json()
    try:
        result = predictWithMessage(data['MESSAGE'])
        return jsonify({"prediction": result}), 200
    except:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route('/predictWithImage', methods=['GET'])
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
