import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Paths
here = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(here, 'trained_model', 'plant_disease_prediction_model.h5')
class_indices_path = os.path.join(os.path.dirname(here), 'class_indices.json')

if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    sys.exit(2)

# Find a sample image under the repository's test_images/ directory
root = os.path.abspath(os.path.join(here, '..', 'test_images'))
sample_image = None
for dirpath, dirs, files in os.walk(root):
    for fn in files:
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_image = os.path.join(dirpath, fn)
            break
    if sample_image:
        break

if not sample_image:
    print(f"No sample images found under: {root}")
    sys.exit(3)

print('Using sample image:', sample_image)

# Preprocess
try:
    img = Image.open(sample_image).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, 0)
except Exception as e:
    print('Failed to load/preprocess image:', e)
    sys.exit(4)

print('Loading model...')
model = tf.keras.models.load_model(model_path)
print('Model loaded.')

preds = model.predict(arr)
pred_index = int(np.argmax(preds, axis=1)[0])

# Load class indices if available
pred_name = str(pred_index)
if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        pred_name = class_indices.get(str(pred_index), str(pred_index))
    except Exception:
        pass

print('Predicted class index:', pred_index)
print('Predicted class name:', pred_name)
print('Top-5 scores:')
sorted_idx = np.argsort(preds[0])[::-1][:5]
for i in sorted_idx:
    name = class_indices.get(str(i), str(i)) if 'class_indices' in locals() else str(i)
    print(f"  {i}: {name} -> {preds[0][i]:.4f}")
