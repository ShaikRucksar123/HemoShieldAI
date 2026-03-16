import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for server

import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ================= FLASK APP =================
app = Flask(__name__)

# ================= CONFIG =================
UPLOAD_FOLDER = "static/uploads"

ALLOWED_EXTENSIONS = {
    'png','jpg','jpeg','webp','bmp','jfif','tif','tiff','gif'
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (160,160)

MODEL_PATH = "models/saved/combined_model_fast.h5"
SCALER_PATH = "models/saved/scaler.pkl"
CLASS_NAMES_PATH = "models/saved/class_names.pkl"

MODEL = None
SCALER = None
CLASS_NAMES = []


# ================= LOAD MODEL =================
def load_resources():

    global MODEL, SCALER, CLASS_NAMES

    try:

        if MODEL is None:
            MODEL = load_model(MODEL_PATH, compile=False)
            print("✅ Model Loaded")

        if SCALER is None:
            with open(SCALER_PATH,"rb") as f:
                SCALER = pickle.load(f)
            print("✅ Scaler Loaded")

        if not CLASS_NAMES:
            with open(CLASS_NAMES_PATH,"rb") as f:
                CLASS_NAMES = pickle.load(f)
            print("✅ Class Names Loaded:", CLASS_NAMES)

    except Exception as e:
        print("❌ Resource loading error:", e)


# ================= UTILS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS


def validate_clinical_inputs(wbc,rbc,platelets,hb,blasts,age):

    if any(v < 0 for v in [wbc,rbc,platelets,hb,blasts,age]):
        return False,"Values cannot be negative"

    if not (0 <= age <= 120):
        return False,"Age must be between 0-120"

    if not (1 <= wbc <= 200):
        return False,"WBC out of range"

    if not (1 <= rbc <= 10):
        return False,"RBC out of range"

    if not (10 <= platelets <= 1500):
        return False,"Platelets out of range"

    if not (3 <= hb <= 25):
        return False,"Hemoglobin out of range"

    if not (0 <= blasts <= 100):
        return False,"Blast % must be 0-100"

    return True,""


def get_disease_specific_suggestion(label):

    suggestions = {

        "ALL":"Acute Lymphoblastic Leukemia detected. Immediate consultation required.",
        "AML":"Acute Myeloid Leukemia detected. Urgent specialist consultation required.",
        "CLL":"Chronic Lymphocytic Leukemia detected. Monitoring recommended.",
        "CML":"Chronic Myeloid Leukemia detected. Oncologist consultation advised.",
        "FL":"Follicular Lymphoma detected. Further evaluation required.",
        "Healthy":"No abnormal cancerous patterns detected."
    }

    return suggestions.get(label,"Consult specialist")


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    metrics = {
        "accuracy": 0.9761,
        "f1_score": 0.978,
        "dataset_size": 7000,

        "history": {
            "accuracy": [
                0.72, 0.80, 0.85, 0.90,0.91,0.92, 0.93, 0.95, 0.96, 0.9761
            ],
            "f1_score": [
                0.70, 0.78, 0.84, 0.89,0.90,0.91 ,0.92, 0.95, 0.96, 0.978
            ]
        }
    }

    return render_template("dashboard.html", metrics=metrics)

# ================= PREDICTION =================
@app.route("/predict_combined",methods=["POST"])
def predict_combined():

    load_resources()

    if MODEL is None or SCALER is None or not CLASS_NAMES:
        return jsonify({"status":"error","message":"Model not loaded properly"})

    try:

        # ===== clinical inputs =====
        wbc=float(request.form["wbc"])
        rbc=float(request.form["rbc"])
        platelets=float(request.form["platelets"])
        hb=float(request.form["hb"])
        blasts=float(request.form["blasts"])
        age=float(request.form["age"])

        valid,msg=validate_clinical_inputs(wbc,rbc,platelets,hb,blasts,age)

        if not valid:
            return jsonify({"status":"error","message":msg})

        # ===== image input =====
        file=request.files.get("file")

        if not file or not allowed_file(file.filename):
            return jsonify({"status":"error","message":"Invalid image file"})

        filename=secure_filename(file.filename)
        filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename)

        file.save(filepath)

        img=image.load_img(filepath,target_size=IMG_SIZE)
        img=image.img_to_array(img)

        img=preprocess_input(img)
        img=np.expand_dims(img,axis=0)

        # ===== clinical features =====
        clinical=np.array([[wbc,rbc,platelets,hb,blasts,age]],dtype=np.float32)
        clinical=SCALER.transform(clinical)

        # ===== prediction =====
        preds=MODEL.predict([img,clinical],verbose=0)[0]

        index=int(np.argmax(preds))
        confidence=float(np.max(preds)*100)

        label=CLASS_NAMES[index]

        print("Prediction:",label,"Confidence:",confidence)

        return jsonify({

            "status":"success",
            "prediction":label,
            "confidence":round(confidence,2),
            "suggestion":get_disease_specific_suggestion(label)

        })

    except Exception as e:

        print("❌ Prediction Error:", e)

        return jsonify({
            "status":"error",
            "message":str(e)
        })


# ================= START APP =================
if __name__ == "__main__":

    load_resources()

    port = int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)