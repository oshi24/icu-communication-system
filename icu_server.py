from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import json
import sqlite3
import datetime
from collections import defaultdict
import threading
import time

# Import your ML model components (assuming they're in separate files)
# from medical_ml_model import generate_medical_dataset, train_model
# For now, we'll include the essential ML components here

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class ICUCommunicationSystem:
    def __init__(self):
        self.ml_model = None
        self.input_encoder = None
        self.label_encoder = None
        self.condition_symptom_map = {}
        self.button_mappings = {}
        self.init_database()
        self.load_ml_model()
        
    def init_database(self):
        """Initialize SQLite database for storing requests and patient data"""
        conn = sqlite3.connect('icu_system.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bed_number TEXT UNIQUE,
                patient_name TEXT,
                condition TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bed_number TEXT,
                patient_id INTEGER,
                button_pressed INTEGER,
                message TEXT,
                risk_level TEXT,
                predicted_symptoms TEXT,
                status TEXT DEFAULT 'new',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Add sample patients for testing
        self.add_sample_patients()
    
    def add_sample_patients(self):
        """Add sample patients to the database"""
        sample_patients = [
            ("1A", "John D.", "Cardiovascular Disease"),
            ("1B", "Mary T.", "Respiratory Infection"),
            ("2A", "Robert K.", "Hypertension"),
            ("2B", "Lisa S.", "Pneumonia"),
            ("3A", "David W.", "Asthma"),
            ("3B", "Sarah M.", "Diabetes Complications")
        ]
        
        conn = sqlite3.connect('icu_system.db')
        cursor = conn.cursor()
        
        for bed, name, condition in sample_patients:
            cursor.execute('''
                INSERT OR IGNORE INTO patients (bed_number, patient_name, condition)
                VALUES (?, ?, ?)
            ''', (bed, name, condition))
        
        conn.commit()
        conn.close()
    
    def load_ml_model(self):
        """Load or train the ML model"""
        try:
            # Try to load pre-trained model
            with open('ml_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.ml_model = model_data['model']
                self.input_encoder = model_data['input_encoder']
                self.label_encoder = model_data['label_encoder']
                self.condition_symptom_map = model_data['condition_symptom_map']
            print("ML model loaded successfully!")
        except FileNotFoundError:
            print("Training new ML model...")
            self.train_new_model()
    
    def train_new_model(self):
        """Train a new ML model using your existing code"""
        # Simplified version of your ML training code
        self.condition_symptom_map = {
            'Cardiovascular Disease': {
                'primary_symptoms': ['Chest Pain', 'High Blood Pressure', 'Irregular Heartbeat'],
                'secondary_symptoms': ['Shortness of Breath', 'Fatigue', 'Dizziness', 'Swelling in Legs'],
                'rare_symptoms': ['Nausea', 'Cold Sweats', 'Back Pain']
            },
            'Respiratory Infection': {
                'primary_symptoms': ['Coughing', 'Shortness of Breath', 'Fever'],
                'secondary_symptoms': ['Chest Congestion', 'Fatigue', 'Headache', 'Sore Throat'],
                'rare_symptoms': ['Chills', 'Body Aches', 'Loss of Appetite']
            },
            'Hypertension': {
                'primary_symptoms': ['High Blood Pressure', 'Headache'],
                'secondary_symptoms': ['Dizziness', 'Fatigue', 'Blurred Vision'],
                'rare_symptoms': ['Chest Pain', 'Irregular Heartbeat', 'Shortness of Breath']
            },
            'Pneumonia': {
                'primary_symptoms': ['Coughing', 'Fever', 'Chest Pain'],
                'secondary_symptoms': ['Shortness of Breath', 'Fatigue', 'Chills', 'Chest Congestion'],
                'rare_symptoms': ['Confusion', 'Nausea', 'Vomiting', 'Diarrhea']
            },
            'Asthma': {
                'primary_symptoms': ['Shortness of Breath', 'Wheezing', 'Coughing'],
                'secondary_symptoms': ['Chest Tightness', 'Fatigue'],
                'rare_symptoms': ['Chest Pain', 'Sleep Disturbance', 'Anxiety']
            },
            'Diabetes Complications': {
                'primary_symptoms': ['Fatigue', 'Frequent Urination', 'Excessive Thirst'],
                'secondary_symptoms': ['Blurred Vision', 'Slow Healing Wounds', 'Weight Loss'],
                'rare_symptoms': ['Numbness in Extremities', 'Dizziness', 'Confusion']
            }
        }
        
        # Generate training data
        X_raw, Y_raw, all_possible_labels = self.generate_training_data()
        
        # Train encoders and model
        X_reshaped = np.array(X_raw).reshape(-1, 1)
        self.input_encoder = OneHotEncoder(sparse_output=False)
        X_encoded = self.input_encoder.fit_transform(X_reshaped)
        
        self.label_encoder = MultiLabelBinarizer(classes=all_possible_labels)
        Y_encoded = self.label_encoder.fit_transform(Y_raw)
        
        self.ml_model = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
        self.ml_model.fit(X_encoded, Y_encoded)
        
        # Save the model
        model_data = {
            'model': self.ml_model,
            'input_encoder': self.input_encoder,
            'label_encoder': self.label_encoder,
            'condition_symptom_map': self.condition_symptom_map
        }
        
        with open('ml_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("ML model trained and saved successfully!")
    
    def generate_training_data(self, num_samples=500):
        """Generate training data for the ML model"""
        import random
        
        all_symptoms = set()
        for condition_data in self.condition_symptom_map.values():
            for symptom_category in condition_data.values():
                all_symptoms.update(symptom_category)
        
        all_possible_labels = sorted(list(all_symptoms))
        
        X_raw = []
        Y_raw = []
        
        conditions = list(self.condition_symptom_map.keys())
        
        for _ in range(num_samples):
            condition = random.choice(conditions)
            symptoms = self.generate_symptoms_for_condition(self.condition_symptom_map[condition])
            X_raw.append(condition)
            Y_raw.append(symptoms)
        
        return X_raw, Y_raw, all_possible_labels
    
    def generate_symptoms_for_condition(self, condition_data):
        """Generate realistic symptoms for a medical condition"""
        import random
        
        symptoms = []
        
        # Primary symptoms (1-2, high probability)
        primary_count = random.choices([1, 2], weights=[0.4, 0.6])[0]
        symptoms.extend(random.sample(condition_data['primary_symptoms'],
                                    min(primary_count, len(condition_data['primary_symptoms']))))
        
        # Secondary symptoms (0-2, medium probability)
        secondary_count = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        if secondary_count > 0:
            symptoms.extend(random.sample(condition_data['secondary_symptoms'],
                                        min(secondary_count, len(condition_data['secondary_symptoms']))))
        
        # Rare symptoms (0-1, low probability)
        rare_count = random.choices([0, 1], weights=[0.7, 0.3])[0]
        if rare_count > 0:
            symptoms.extend(random.sample(condition_data['rare_symptoms'],
                                        min(rare_count, len(condition_data['rare_symptoms']))))
        
        return list(set(symptoms))
    
    def get_patient_by_bed(self, bed_number):
        """Get patient information by bed number"""
        conn = sqlite3.connect('icu_system.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, bed_number, patient_name, condition
            FROM patients WHERE bed_number = ?
        ''', (bed_number,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'bed_number': result[1],
                'patient_name': result[2],
                'condition': result[3]
            }
        return None
    
    def predict_symptoms(self, condition):
        """Predict symptoms for a given condition"""
        if not self.ml_model:
            return []
        
        try:
            # Preprocess input
            condition_reshaped = np.array([condition]).reshape(-1, 1)
            condition_encoded = self.input_encoder.transform(condition_reshaped)
            
            # Get probability scores
            prob_scores = self.ml_model.predict_proba(condition_encoded)[0]
            
            # Create label-probability pairs and sort by probability
            label_prob_pairs = list(zip(self.label_encoder.classes_, prob_scores))
            sorted_predictions = sorted(label_prob_pairs, key=lambda x: x[1], reverse=True)
            
            # Return top 6 symptoms with probabilities > 0.3
            top_symptoms = [(label, prob) for label, prob in sorted_predictions[:6] if prob > 0.3]
            
            return top_symptoms
        except Exception as e:
            print(f"Error in prediction: {e}")
            return []
    
    def get_button_options(self, condition):
        """Generate 4 button options based on predicted symptoms"""
        predicted_symptoms = self.predict_symptoms(condition)
        
        if not predicted_symptoms:
            # Fallback options
            return {
                1: {"message": "EMERGENCY HELP", "risk": "HIGH"},
                2: {"message": "PAIN / DISCOMFORT", "risk": "MEDIUM"},
                3: {"message": "BREATHING DIFFICULTY", "risk": "MEDIUM"},
                4: {"message": "BASIC NEED", "risk": "LOW"}
            }
        
        # Create button mappings based on predicted symptoms
        button_options = {}
        
        # Button 1: Always emergency
        button_options[1] = {"message": "EMERGENCY HELP", "risk": "HIGH"}
        
        # Buttons 2-4: Based on top predicted symptoms
        symptom_to_message = {
            'Chest Pain': {"message": "CHEST PAIN", "risk": "HIGH"},
            'Shortness of Breath': {"message": "BREATHING DIFFICULTY", "risk": "MEDIUM"},
            'High Blood Pressure': {"message": "BLOOD PRESSURE ISSUE", "risk": "MEDIUM"},
            'Fever': {"message": "FEVER / TEMPERATURE", "risk": "MEDIUM"},
            'Coughing': {"message": "COUGH / THROAT", "risk": "LOW"},
            'Headache': {"message": "HEAD / NECK PAIN", "risk": "LOW"},
            'Fatigue': {"message": "WEAKNESS / TIRED", "risk": "LOW"},
            'Nausea': {"message": "NAUSEA / STOMACH", "risk": "MEDIUM"},
            'Dizziness': {"message": "DIZZY / BALANCE", "risk": "MEDIUM"}
        }
        
        button_num = 2
        used_messages = set()
        
        for symptom, prob in predicted_symptoms[:3]:  # Top 3 symptoms
            if symptom in symptom_to_message and button_num <= 4:
                message_data = symptom_to_message[symptom]
                if message_data["message"] not in used_messages:
                    button_options[button_num] = message_data
                    used_messages.add(message_data["message"])
                    button_num += 1
        
        # Fill remaining buttons with generic options
        generic_options = [
            {"message": "PAIN / DISCOMFORT", "risk": "MEDIUM"},
            {"message": "BASIC NEED", "risk": "LOW"},
            {"message": "MEDICATION REQUEST", "risk": "LOW"}
        ]
        
        for option in generic_options:
            if button_num <= 4 and option["message"] not in used_messages:
                button_options[button_num] = option
                button_num += 1
        
        return button_options
    
    def save_request(self, bed_number, button_pressed, patient_data, button_options):
        """Save button press request to database"""
        conn = sqlite3.connect('icu_system.db')
        cursor = conn.cursor()
        
        button_info = button_options.get(button_pressed, {"message": "UNKNOWN", "risk": "LOW"})
        predicted_symptoms = self.predict_symptoms(patient_data['condition'])
        symptoms_json = json.dumps([{"symptom": s, "probability": p} for s, p in predicted_symptoms])
        
        cursor.execute('''
            INSERT INTO requests (bed_number, patient_id, button_pressed, message, risk_level, predicted_symptoms, status)
            VALUES (?, ?, ?, ?, ?, ?, 'new')
        ''', (bed_number, patient_data['id'], button_pressed, button_info['message'], 
              button_info['risk'], symptoms_json))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return request_id

# Initialize the system
icu_system = ICUCommunicationSystem()

# API Routes

@app.route('/api/get_ml_data', methods=['GET'])
def get_ml_data():
    """
    API endpoint for devices to get ML-based button options
    Usage: GET /api/get_ml_data?bed_number=1A
    """
    bed_number = request.args.get('bed_number')
    
    if not bed_number:
        return jsonify({"error": "bed_number parameter is required"}), 400
    
    # Get patient data
    patient = icu_system.get_patient_by_bed(bed_number)
    if not patient:
        return jsonify({"error": f"Patient not found for bed {bed_number}"}), 404
    
    # Get ML predictions and button options
    predicted_symptoms = icu_system.predict_symptoms(patient['condition'])
    button_options = icu_system.get_button_options(patient['condition'])
    
    response_data = {
        "bed_number": bed_number,
        "patient_name": patient['patient_name'],
        "condition": patient['condition'],
        "predicted_symptoms": [
            {"symptom": symptom, "probability": round(prob, 3)} 
            for symptom, prob in predicted_symptoms
        ],
        "button_options": button_options,
        "timestamp": datetime.datetime.now().isoformat(),
        "device_instructions": {
            "lcd_display": f"Patient: {patient['patient_name']} | Condition: {patient['condition']}",
            "button_mappings": {
                f"Button {k}": v["message"] for k, v in button_options.items()
            }
        }
    }
    
    return jsonify(response_data)

@app.route('/api/button_pressed', methods=['POST'])
def button_pressed():
    """
    API endpoint for when a patient presses a button
    Usage: POST /api/button_pressed
    Body: {"bed_number": "1A", "button": 2}
    """
    data = request.get_json()
    
    if not data or 'bed_number' not in data or 'button' not in data:
        return jsonify({"error": "bed_number and button parameters are required"}), 400
    
    bed_number = data['bed_number']
    button_num = int(data['button'])
    
    if button_num not in [1, 2, 3, 4]:
        return jsonify({"error": "button must be 1, 2, 3, or 4"}), 400
    
    # Get patient data
    patient = icu_system.get_patient_by_bed(bed_number)
    if not patient:
        return jsonify({"error": f"Patient not found for bed {bed_number}"}), 404
    
    # Get current button options for this patient
    button_options = icu_system.get_button_options(patient['condition'])
    
    # Save the request
    request_id = icu_system.save_request(bed_number, button_num, patient, button_options)
    
    button_info = button_options.get(button_num, {"message": "UNKNOWN", "risk": "LOW"})
    
    response_data = {
        "request_id": request_id,
        "bed_number": bed_number,
        "patient_name": patient['patient_name'],
        "button_pressed": button_num,
        "message": button_info['message'],
        "risk_level": button_info['risk'],
        "status": "received",
        "timestamp": datetime.datetime.now().isoformat(),
        "acknowledgment": f"Request received from {patient['patient_name']} in bed {bed_number}: {button_info['message']}"
    }
    
    print(f"📱 Button Press Alert: Bed {bed_number} - {button_info['message']} ({button_info['risk']} priority)")
    
    return jsonify(response_data)

@app.route('/api/nurse_dashboard', methods=['GET'])
def get_nurse_dashboard_data():
    """
    API endpoint for nurse dashboard to get current requests
    """
    conn = sqlite3.connect('icu_system.db')
    cursor = conn.cursor()
    
    # Get active requests with patient information
    cursor.execute('''
        SELECT r.id, r.bed_number, p.patient_name, r.button_pressed, r.message, 
               r.risk_level, r.status, r.created_at, r.predicted_symptoms
        FROM requests r
        JOIN patients p ON r.patient_id = p.id
        WHERE r.status != 'resolved'
        ORDER BY 
            CASE r.risk_level 
                WHEN 'HIGH' THEN 1 
                WHEN 'MEDIUM' THEN 2 
                WHEN 'LOW' THEN 3 
            END,
            r.created_at ASC
    ''')
    
    requests = []
    for row in cursor.fetchall():
        predicted_symptoms = json.loads(row[8]) if row[8] else []
        
        requests.append({
            "id": f"req_{row[0]}",
            "bedNumber": row[1],
            "patientId": f"Patient {row[2]}",
            "message": row[4],
            "riskLevel": row[5],
            "status": row[6],
            "timestamp": row[7],
            "predicted_symptoms": predicted_symptoms
        })
    
    conn.close()
    
    return jsonify({
        "requests": requests,
        "total_active": len(requests),
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/acknowledge_request', methods=['POST'])
def acknowledge_request():
    """Acknowledge a request"""
    data = request.get_json()
    request_id = data.get('request_id', '').replace('req_', '')
    
    conn = sqlite3.connect('icu_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE requests SET status = 'acknowledged' WHERE id = ?
    ''', (request_id,))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "acknowledged", "request_id": request_id})

@app.route('/api/resolve_request', methods=['POST'])
def resolve_request():
    """Resolve a request"""
    data = request.get_json()
    request_id = data.get('request_id', '').replace('req_', '')
    
    conn = sqlite3.connect('icu_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE requests SET status = 'resolved' WHERE id = ?
    ''', (request_id,))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "resolved", "request_id": request_id})

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """Get system status information"""
    conn = sqlite3.connect('icu_system.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM patients')
    total_patients = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM requests WHERE status != "resolved"')
    active_requests = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "ml_model_loaded": icu_system.ml_model is not None,
        "total_patients": total_patients,
        "active_requests": active_requests,
        "system_time": datetime.datetime.now().isoformat(),
        "database_status": "connected"
    })

# Test endpoints for development
@app.route('/api/test_ml', methods=['GET'])
def test_ml():
    """Test ML model prediction"""
    condition = request.args.get('condition', 'Cardiovascular Disease')
    symptoms = icu_system.predict_symptoms(condition)
    button_options = icu_system.get_button_options(condition)
    
    return jsonify({
        "condition": condition,
        "predicted_symptoms": symptoms,
        "button_options": button_options
    })
# Add this route before the if __name__ == '__main__': line in your icu_server.py

@app.route('/')
def home():
    """Root endpoint showing system information"""
    return """
    <html>
    <head>
        <title>ICU Communication System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; max-width: 800px; margin: 0 auto; }
            .endpoint { background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .status { color: #28a745; font-weight: bold; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 ICU Communication System API</h1>
            <p class="status">✅ System Online</p>
            
            <h2>📊 Available API Endpoints:</h2>
            
            <div class="endpoint">
                <strong>System Status:</strong><br>
                <a href="/api/system_status" target="_blank">GET /api/system_status</a>
            </div>
            
            <div class="endpoint">
                <strong>Test ML Model:</strong><br>
                <a href="/api/test_ml?condition=Cardiovascular Disease" target="_blank">GET /api/test_ml?condition=Cardiovascular Disease</a>
            </div>
            
            <div class="endpoint">
                <strong>Get Patient Data:</strong><br>
                <a href="/api/get_ml_data?bed_number=1A" target="_blank">GET /api/get_ml_data?bed_number=1A</a>
            </div>
            
            <div class="endpoint">
                <strong>Dashboard Data:</strong><br>
                <a href="/api/nurse_dashboard" target="_blank">GET /api/nurse_dashboard</a>
            </div>
            
            <h2>🖥️ Frontend Dashboard:</h2>
            <p>Open <strong>dashboard.html</strong> in your browser to access the full dashboard interface.</p>
            
            <h2>📝 Sample API Calls:</h2>
            <div class="endpoint">
                <strong>Simulate Button Press (POST):</strong><br>
                <code>curl -X POST http://localhost:5000/api/button_pressed -H "Content-Type: application/json" -d '{"bed_number": "1A", "button": 1}'</code>
            </div>
            
            <h2>📚 Sample Patients:</h2>
            <ul>
                <li>Bed 1A: John D. (Cardiovascular Disease)</li>
                <li>Bed 1B: Mary T. (Respiratory Infection)</li>
                <li>Bed 2A: Robert K. (Hypertension)</li>
                <li>Bed 2B: Lisa S. (Pneumonia)</li>
                <li>Bed 3A: David W. (Asthma)</li>
                <li>Bed 3B: Sarah M. (Diabetes Complications)</li>
            </ul>
        </div>
    </body>
    </html>
    """
if __name__ == '__main__':
    print("🏥 ICU Communication System Starting...")
    print("📊 ML Model Status:", "Loaded" if icu_system.ml_model else "Not Loaded")
    print("🔗 API Endpoints Available:")
    print("   GET  /api/get_ml_data?bed_number=1A")
    print("   POST /api/button_pressed")
    print("   GET  /api/nurse_dashboard")
    print("   GET  /api/system_status")
    print("\n🚀 Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)