import tensorflow as tf
import numpy as np
import cv2
import base64
import json # <-- Required for handling JSON requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- Load the trained model ---
# This part remains the same, but with more robust error handling.
try:
    # IMPORTANT: Update this path to where your model is located on the server.
    MODEL_PATH = 'app/example_app/digit_classifier.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load the model. The application will not work. Error: {e}")
    model = None # Set model to None so we can handle this gracefully in the view

@csrf_exempt
def predict_live_frame(request):
    # Gracefully handle the case where the model failed to load on startup
    if model is None:
        return JsonResponse({'error': 'Model is not loaded on the server.'}, status=500)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)

    try:
        # --- IMPROVEMENT 1: Handle JSON input from the improved frontend ---
        # The frontend now sends a JSON object, not form data.
        # We need to load the request body as JSON.
        body = json.loads(request.body)
        image_data = body.get('image')
        if not image_data:
            return JsonResponse({'error': 'Missing "image" key in JSON payload.'}, status=400)
        
        # Decode the base64 string
        header, encoded_data = image_data.split(';base64,')
        image_bytes = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return JsonResponse({'error': f'Frame decoding or JSON processing failed: {str(e)}'}, status=400)

    # --- OpenCV processing to find contours (remains the same) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- IMPROVEMENT 2: Prepare for Batch Prediction for massive performance gain ---
    digit_rois = [] # This will hold the processed 28x28 digit images
    bounding_boxes = [] # This will hold the corresponding [x, y, w, h] coordinates

    for c in contours:
        # Filter out contours that are too small or too large to be digits
        if cv2.contourArea(c) < 100:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop the digit from the thresholded image
        digit_crop = thresh[y:y+h, x:x+w]
        
        # The same resizing and padding logic as before
        canvas_dim, target_dim = 28, 20
        larger_dim = max(w, h)
        scale_factor = target_dim / float(larger_dim)
        resized_w, resized_h = int(w * scale_factor), int(h * scale_factor)
        
        # Skip invalid contours that would result in a 0-sized image
        if resized_w == 0 or resized_h == 0: continue
        
        resized_digit = cv2.resize(digit_crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        
        padded_digit = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)
        pad_x = (canvas_dim - resized_w) // 2
        pad_y = (canvas_dim - resized_h) // 2
        padded_digit[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w] = resized_digit

        # Reshape for the model and add the processed digit and its box to our lists
        input_data = padded_digit.reshape(28, 28, 1).astype('float32') / 255.0
        digit_rois.append(input_data)
        bounding_boxes.append([x, y, w, h])

    final_predictions = []
    # Only run prediction if we found any potential digits
    if digit_rois:
        # --- IMPROVEMENT 2 (continued): Perform batch prediction ONCE ---
        # This is much faster than calling model.predict() in a loop.
        batch = np.array(digit_rois)
        model_predictions = model.predict(batch)
        predicted_digits = np.argmax(model_predictions, axis=1)

        # Combine the bounding boxes with their predicted digits
        for box, digit in zip(bounding_boxes, predicted_digits):
            final_predictions.append({
                'box': [int(val) for val in box], # Ensure values are standard integers for JSON
                'digit': int(digit)
            })

    # --- BUG FIX: The return statement is now correctly outside the loop ---
    return JsonResponse({'predictions': final_predictions})
