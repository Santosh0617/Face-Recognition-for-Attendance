import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import os
import requests
from datetime import datetime
from deepface import DeepFace

app = Flask(__name__)

class SimpleFaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_embeddings = {}  # Store face embeddings
        self.face_labels = {}      # Map embeddings to names
        self.attendance_log = []
        self.is_trained = False
        self.setup_known_faces()
        
    def download_image(self, url, filename):
        """Download image from URL and save locally"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
        return False
    
    def preprocess_image(self, img):
        """Preprocess image for better face detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for better contrast
        gray = cv2.equalizeHist(gray)
        return gray
    
    def setup_known_faces(self):
        """Setup known faces with their embeddings using DeepFace"""
        if not os.path.exists('faces'):
            os.makedirs('faces')
        
        expected_faces = {
            'user.jpg': 'Santosh Padhy',
            'elon_musk.jpg': 'Elon Musk',
            'jeff_bezos.jpg': 'Jeff Bezos',
            'donald_trump.jpg': 'Donald Trump'
        }
        
        current_label = 0
        valid_embeddings = 0
        
        for filename, name in expected_faces.items():
            filepath = os.path.join('faces', filename)
            
            if os.path.exists(filepath):
                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"Failed to load image: {filepath}")
                        continue
                    
                    gray = self.preprocess_image(img)
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,  # Tighter scale factor for better detection
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                    
                    if len(faces) == 0:
                        print(f"No face detected in {filepath}")
                        continue
                    
                    # Use the first detected face
                    (x, y, w, h) = faces[0]
                    face_region = img[y:y+h, x:x+w]
                    
                    # Generate embedding using DeepFace
                    embedding = DeepFace.represent(
                        face_region, 
                        model_name='VGG-Face', 
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if not embedding or 'embedding' not in embedding[0]:
                        print(f"Failed to generate embedding for {filepath}")
                        continue
                    
                    self.face_embeddings[current_label] = embedding[0]['embedding']
                    self.face_labels[current_label] = name
                    valid_embeddings += 1
                    print(f"Loaded face: {name} (Label: {current_label})")
                    current_label += 1
                    
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
            else:
                print(f"Face image not found: {filepath}")
        
        if valid_embeddings > 0:
            self.is_trained = True
            print(f"Training completed with {valid_embeddings} face embeddings")
        else:
            print("No valid face data found for training")
        
        with open('faces/README.txt', 'w') as f:
            f.write("""Face Recognition Setup Instructions:

1. Add the following images to this 'faces' folder:
   - user.jpg (your face photo)
   - elon_musk.jpg (Elon Musk's photo)
   - jeff_bezos.jpg (Jeff Bezos' photo)  
   - donald_trump.jpg (Donald Trump's photo)

2. Ensure images are clear, front-facing, well-lit photos
3. Restart the application after adding images
4. Install DeepFace: `pip install deepface`
5. Ensure TensorFlow is installed: `pip install tensorflow`

The system will automatically detect and train on these faces.""")
    
    def recognize_faces(self, frame):
        """Recognize faces in the given frame using DeepFace"""
        gray = self.preprocess_image(frame)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            name = "Unknown"
            confidence = 0
            
            if self.is_trained:
                try:
                    # Generate embedding for the detected face
                    embedding = DeepFace.represent(
                        face_region, 
                        model_name='VGG-Face', 
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if not embedding or 'embedding' not in embedding[0]:
                        print("Failed to generate embedding for detected face")
                        recognized_faces.append({
                            'name': name,
                            'box': (x, y, w, h),
                            'confidence': confidence
                        })
                        continue
                    
                    current_embedding = embedding[0]['embedding']
                    
                    # Compare with stored embeddings
                    min_distance = float('inf')
                    recognized_label = -1
                    
                    for label, stored_embedding in self.face_embeddings.items():
                        # Calculate cosine distance using numpy for accuracy
                        stored_embedding = np.array(stored_embedding)
                        current_embedding = np.array(current_embedding)
                        distance = np.dot(current_embedding, stored_embedding) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                        )
                        distance = np.arccos(np.clip(distance, -1.0, 1.0)) / np.pi  # Normalize to [0,1]
                        
                        if distance < min_distance:
                            min_distance = distance
                            recognized_label = label
                    
                    # Threshold for recognition (0.3 is a reasonable starting point for VGG-Face)
                    if min_distance < 0.3:
                        name = self.face_labels.get(recognized_label, "Unknown")
                        confidence = (1 - min_distance) * 100
                        self.log_attendance(name)
                        print(f"Recognized: {name} (Distance: {min_distance:.3f}, Confidence: {confidence:.1f}%)")
                    else:
                        print(f"Face not recognized (Distance: {min_distance:.3f})")
                    
                except Exception as e:
                    print(f"Error recognizing face: {e}")
            
            recognized_faces.append({
                'name': name,
                'box': (x, y, w, h),
                'confidence': confidence
            })
        
        return recognized_faces
    
    def log_attendance(self, name):
        """Log attendance for recognized person"""
        if name == "Unknown":
            return
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        recent_log = any(
            log['name'] == name and 
            (datetime.now() - datetime.strptime(log['time'], "%Y-%m-%d %H:%M:%S")).seconds < 10
            for log in self.attendance_log
        )
        
        if not recent_log:
            self.attendance_log.append({
                'name': name,
                'time': current_time
            })
            print(f"Attendance logged: {name} at {current_time}")

# Initialize the face recognition system
face_system = SimpleFaceRecognitionSystem()

def generate_frames():
    """Generate video frames with face recognition"""
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        recognized_faces = face_system.recognize_faces(frame)
        
        for face_info in recognized_faces:
            name = face_info['name']
            x, y, w, h = face_info['box']
            confidence = face_info['confidence']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.1f}%)"
                
            cv2.rectangle(frame, (x, y-30), (x+len(label)*10, y), color, -1)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not face_system.is_trained:
            cv2.putText(frame, "Add face images to 'faces' folder and restart", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Face Recognition Attendance System</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                background-color: #f0f0f0;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .video-container {
                margin: 20px 0;
            }
            .attendance-log {
                margin-top: 20px;
                text-align: left;
            }
            .log-item {
                padding: 8px;
                margin: 5px 0;
                background: #e7f3ff;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }
            button {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                font-size: 14px;
            }
            button:hover {
                background: #0056b3;
            }
            .status {
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                font-weight: bold;
            }
            .status.trained {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.not-trained {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .instructions {
                text-align: left;
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Simple Face Recognition Attendance System</h1>
            
            <div id="status-display" class="status"></div>
            
            <div class="instructions">
                <h4>Setup Instructions:</h4>
                <ol>
                    <li>Add face images to the <strong>'faces'</strong> folder in your project directory</li>
                    <li>Required images: user.jpg, elon_musk.jpg, jeff_bezos.jpg, donald_trump.jpg</li>
                    <li>Ensure images are clear, front-facing, well-lit photos</li>
                    <li>Restart the application after adding images</li>
                    <li>Install DeepFace: <code>pip install deepface</code></li>
                    <li>Install TensorFlow: <code>pip install tensorflow</code></li>
                </ol>
            </div>
            
            <div class="video-container">
                <img src="/video_feed" width="640" height="480" style="border: 2px solid #ddd; border-radius: 8px;" />
            </div>
            
            <button onclick="refreshAttendance()">Refresh Attendance</button>
            <button onclick="clearAttendance()">Clear Log</button>
            <button onclick="checkStatus()">Check Training Status</button>
            
            <div class="attendance-log">
                <h3>Attendance Log:</h3>
                <div id="attendance-list"></div>
            </div>
        </div>
        
        <script>
            function refreshAttendance() {
                fetch('/attendance')
                    .then(response => response.json())
                    .then(data => {
                        const list = document.getElementById('attendance-list');
                        list.innerHTML = '';
                        if (data.length === 0) {
                            list.innerHTML = '<p style="color: #666;">No attendance records yet.</p>';
                        } else {
                            data.reverse().forEach(item => {
                                const div = document.createElement('div');
                                div.className = 'log-item';
                                div.innerHTML = `<strong>${item.name}</strong> - ${item.time}`;
                                list.appendChild(div);
                            });
                        }
                    });
            }
            
            function clearAttendance() {
                if (confirm('Are you sure you want to clear all attendance records?')) {
                    fetch('/clear_attendance', {method: 'POST'})
                        .then(() => refreshAttendance());
                }
            }
            
            function checkStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status-display');
                        if (data.is_trained) {
                            statusDiv.className = 'status trained';
                            statusDiv.innerHTML = `✅ System trained with ${data.face_count} faces: ${data.faces.join(', ')}`;
                        } else {
                            statusDiv.className = 'status not-trained';
                            statusDiv.innerHTML = '❌ System not trained. Please add face images and restart.';
                        }
                    });
            }
            
            setInterval(refreshAttendance, 5000);
            refreshAttendance();
            checkStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    """Get attendance log"""
    return jsonify(face_system.attendance_log)

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    """Clear attendance log"""
    face_system.attendance_log = []
    return jsonify({'status': 'cleared'})

@app.route('/status')
def get_status():
    """Get system training status"""
    return jsonify({
        'is_trained': face_system.is_trained,
        'face_count': len(face_system.face_labels),
        'faces': list(face_system.face_labels.values())
    })

if __name__ == '__main__':
    print("Starting Simple Face Recognition Attendance System...")
    print("Add face images to 'faces' folder:")
    print("- user.jpg (your face)")
    print("- elon_musk.jpg")
    print("- jeff_bezos.jpg") 
    print("- donald_trump.jpg")
    print("Install DeepFace: pip install deepface")
    print("Install TensorFlow: pip install tensorflow")
    print("Server will run on http://localhost:3000")
    
    if not os.path.exists('faces'):
        os.makedirs('faces')
    
    app.run(host='0.0.0.0', port=3000, debug=True)