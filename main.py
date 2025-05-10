"""
Fixed Doodle Classifier Web App - Mac Compatible
This script checks for the model file properly and handles errors
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sys
import io
import base64
from PIL import Image

# Create Flask app
app = Flask(__name__)

# Categories matching those used during training
CATEGORIES = ['apple', 'banana', 'car', 'cat', 'chair', 
              'door', 'face', 'house', 'star', 'umbrella']

# Global variable for the model
model = None

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', categories=CATEGORIES, model_loaded=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    """Process the drawn image and return predictions."""
    global model
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 503  # Service Unavailable
    
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Decode the base64 image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale and resize to match the model's input
        image = image.convert('L').resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.reshape(1, 28, 28, 1)
        img_array = 1 - (img_array / 255.0)  # Invert colors for white drawing on black background
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        
        # Format and return the results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'category': CATEGORIES[i],
                'probability': float(pred)
            })
        
        # Sort by probability (highest first)
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'topPrediction': results[0]['category'],
            'confidence': results[0]['probability'] * 100,
            'allPredictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_template():
    """Create the HTML template file."""
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Doodle Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #f5f5f5;
        }
        #canvas-container {
            border: 2px solid #333;
            margin: 20px auto;
            display: inline-block;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        canvas {
            display: block;
            background-color: black;
        }
        #result {
            font-size: 24px;
            margin: 20px 0;
            min-height: 30px;
            color: #333;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #predictions-container {
            width: 100%;
            margin-top: 20px;
        }
        .instructions {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            text-align: left;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .instructions ul {
            text-align: left;
        }
        .category-tag {
            display: inline-block;
            background-color: #eee;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 10px;
            font-size: 14px;
        }
        h1 {
            color: #2C3E50;
            margin-bottom: 10px;
        }
        .header {
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .prediction-bar {
            height: 30px;
            margin: 5px 0;
            background-color: #f1f1f1;
            border-radius: 5px;
            overflow: hidden;
            text-align: left;
        }
        .prediction-fill {
            height: 100%;
            background-color: #4CAF50;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            transition: width 0.5s;
        }
        .error-banner {
            background-color: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: left;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .setup-instructions {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: left;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Doodle Classifier</h1>
        <p>Draw a simple doodle and see if the AI can recognize it!</p>
    </div>
    
    {% if not model_loaded %}
    <div class="error-banner">
        <h3>⚠️ Model Not Loaded</h3>
        <p>The doodle classifier model hasn't been trained yet. Please train the model first before using this app.</p>
    </div>
    
    <div class="setup-instructions">
        <h3>How to Train the Model:</h3>
        <ol>
            <li>Make sure you have the <code>trainer.py</code> file</li>
            <li>Run it with: <code>python trainer.py</code></li>
            <li>Wait for the training to complete (this will download data and train the model)</li>
            <li>Restart this web app after training is complete</li>
        </ol>
    </div>
    {% endif %}
    
    <div class="instructions">
        <h3>Instructions:</h3>
        <ul>
            <li>Draw a simple doodle in the black canvas below</li>
            <li>Try to draw one of these categories:
                <div>
                    {% for category in categories %}
                    <span class="category-tag">{{ category }}</span>
                    {% endfor %}
                </div>
            </li>
            <li>Click "Classify" to see if the AI can recognize your drawing</li>
            <li>Use "Clear" to start over</li>
        </ul>
    </div>
    
    <div id="canvas-container">
        <canvas id="drawingCanvas" width="280" height="280"></canvas>
    </div>
    
    <div>
        <button id="clearButton">Clear</button>
        <button id="classifyButton" {% if not model_loaded %}disabled{% endif %}>Classify</button>
    </div>
    
    <div id="result">
        {% if not model_loaded %}
        Please train the model first
        {% else %}
        Draw something and click Classify!
        {% endif %}
    </div>
    
    <div id="predictions-container"></div>
    
    <script>
        // Canvas setup
        const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // Model status
        const modelLoaded = {{ 'true' if model_loaded else 'false' }};
        
        // Initialize canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        
        // Event listeners for drawing
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', stopDrawing);
        
        // Button event listeners
        document.getElementById('clearButton').addEventListener('click', clearCanvas);
        document.getElementById('classifyButton').addEventListener('click', classifyDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            
            [lastX, lastY] = [e.offsetX, e.offsetY];
        }
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const offsetX = touch.clientX - rect.left;
            const offsetY = touch.clientY - rect.top;
            
            isDrawing = true;
            [lastX, lastY] = [offsetX, offsetY];
        }
        
        function handleTouchMove(e) {
            e.preventDefault();
            if (!isDrawing) return;
            
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const offsetX = touch.clientX - rect.left;
            const offsetY = touch.clientY - rect.top;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(offsetX, offsetY);
            ctx.stroke();
            
            [lastX, lastY] = [offsetX, offsetY];
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            if (modelLoaded) {
                document.getElementById('result').textContent = 'Draw something and click Classify!';
            } else {
                document.getElementById('result').textContent = 'Please train the model first';
            }
            
            document.getElementById('predictions-container').innerHTML = '';
        }
        
        function classifyDrawing() {
            if (!modelLoaded) {
                document.getElementById('result').textContent = 'Error: Model not loaded';
                return;
            }
            
            // Show loading state
            document.getElementById('result').textContent = 'Analyzing your drawing...';
            
            // Get the canvas data
            const imageData = canvas.toDataURL('image/png');
            
            // Send to the backend
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    document.getElementById('result').textContent = 'Error: ' + data.error;
                    return;
                }
                
                // Display the top prediction
                const resultText = `I think it's a ${data.topPrediction} (${data.confidence.toFixed(1)}%)`;
                document.getElementById('result').textContent = resultText;
                
                // Update the predictions display
                updatePredictions(data.allPredictions);
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').textContent = 'Error classifying doodle. Make sure the model is trained.';
            });
        }
        
        function updatePredictions(predictions) {
            const container = document.getElementById('predictions-container');
            container.innerHTML = '';
            
            // Sort predictions by probability (highest first)
            predictions.sort((a, b) => b.probability - a.probability);
            
            // Take top 5 predictions
            const topPredictions = predictions.slice(0, 5);
            
            // Create prediction bars
            topPredictions.forEach(pred => {
                const percent = (pred.probability * 100).toFixed(1);
                const predDiv = document.createElement('div');
                predDiv.className = 'prediction-bar';
                
                const fillDiv = document.createElement('div');
                fillDiv.className = 'prediction-fill';
                fillDiv.style.width = `${percent}%`;
                fillDiv.textContent = `${pred.category}: ${percent}%`;
                
                predDiv.appendChild(fillDiv);
                container.appendChild(predDiv);
            });
        }
    </script>
</body>
</html>
        ''')
    
    return os.path.join('templates', 'index.html')

def main():
    """Main function to start the application."""
    global model
    
    # Create template
    create_template()
    
    # Check if model exists
    if os.path.exists('doodle_classifier.h5'):
        print("Loading existing model...")
        try:
            # Only import TensorFlow if the model exists
            import tensorflow as tf
            model = tf.keras.models.load_model('doodle_classifier.h5')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("The web app will run but classification won't work.")
    else:
        print("\n" + "="*50)
        print("Model file 'doodle_classifier.h5' not found.")
        print("Please train the model first using the trainer script.")
        print("The web app will run but classification won't work.")
        print("="*50 + "\n")
    
    # Start the Flask application
    print("\nStarting web server. Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)

if __name__ == '__main__':
    main()