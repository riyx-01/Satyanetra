import os
import secrets
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO
import dlib

from modules.blur import blur_faces
from modules.deepfake_video_transfer import predict_video
from modules.metadata import extract_metadata
from modules.same_image import compare_images

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)

users = {
    "admin": {"password": generate_password_hash("admin123"), "role": "admin"},
    "user": {"password": generate_password_hash("user123"), "role": "user"}
}

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return ("Fake", float(prediction[0][0])) if prediction[0][0] > 0.5 else ("Real", float(prediction[0][0]))

@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('deepfake_image'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username]['password'], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if username in users:
            flash('Username taken')
        elif password != confirm_password:
            flash('Passwords do not match')
        else:
            users[username] = {'password': generate_password_hash(password), 'role': 'user'}
            flash('Registration successful')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/deepfake_image', methods=['GET', 'POST'])
def deepfake_image():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        label, confidence = predict_image(path)
        result = {"label": label, "confidence": round(confidence * 100, 2), "file_path": path}
        if label == "Fake":
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmap_path = os.path.splitext(path)[0] + "_heatmap.jpg"
            cv2.imwrite(heatmap_path, heatmap)
            result["heatmap"] = '/' + heatmap_path.replace('\\', '/')

    return render_template("index.html", functionality="deepfake_image", result=result, username=session.get("username"))

@app.route('/deepfake_video', methods=['GET', 'POST'])
def deepfake_video():
    result = None
    if request.method == 'POST':
        video = request.files['video']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video.filename))
        video.save(path)
        label, confidence = predict_video(path)

        # Ensure confidence is between 0–1
        if confidence > 1:  # likely already in 0–100
            confidence = confidence / 100

        result = {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "file_path": path.replace('\\', '/')
        }
    return render_template("index.html", functionality="deepfake_video", result=result, username=session.get("username"))


@app.route('/blur', methods=['GET', 'POST'])
def blur():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        output = blur_faces(path)
        result = {"output": '/' + output.replace('\\', '/')}
    return render_template("index.html", functionality="blur", result=result, username=session.get("username"))

@app.route('/metadata', methods=['GET', 'POST'])
def metadata():
    metadata = None
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        metadata = extract_metadata(path)
    return render_template("index.html", functionality="metadata", metadata=metadata, username=session.get("username"))

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    result = None
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename))
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image2.filename))
        image1.save(path1)
        image2.save(path2)
        mse = compare_images(path1, path2)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        diff = cv2.absdiff(img1, img2)
        diff_path = os.path.join(app.config['UPLOAD_FOLDER'], f"diff_{secrets.token_hex(5)}.jpg")
        cv2.imwrite(diff_path, diff)
        match_percent = max(0.0, round(100 - (mse / 100), 2))
        similarity = "Images are similar" if mse < 1000 else "Images are different"
        timestamp = extract_metadata(path1).get("DateTime", "Not available")
        result = {"similarity": similarity, "match_percent": match_percent, "timestamp": timestamp,
                  "image1": '/' + path1.replace('\\', '/'), "image2": '/' + path2.replace('\\', '/'), "diff": '/' + diff_path.replace('\\', '/')}
    return render_template("index.html", functionality="compare", result=result, username=session.get("username"))

@app.route('/transpose', methods=['GET', 'POST'])
def transpose():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        img = Image.open(path)
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        filename = f"transposed_{secrets.token_hex(5)}.jpg"
        flipped_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        flipped.save(flipped_path)
        result = {"transposed": flipped_path.replace('\\', '/')}
    return render_template("index.html", functionality="transpose", result=result, username=session.get("username"))

@app.route('/remove')
def remove():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('remove.html')
@app.route('/remove_object', methods=['POST'])
def remove_object():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        mask_data = base64.b64decode(data['mask'].split(',')[1])

        img = np.array(Image.open(BytesIO(image_data)).convert('RGB'))[:, :, ::-1]
        mask = np.array(Image.open(BytesIO(mask_data)).convert('L'))

        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        filename = f"inpainted_{secrets.token_hex(5)}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(output_path, inpainted)

        return jsonify({"success": True, "output_url": f"/static/uploads/{filename}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/live_detect')
def live_detect():
    return render_template("index.html", functionality="live_detect", username=session.get("username"))

def generate():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for i, face in enumerate(faces):
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
