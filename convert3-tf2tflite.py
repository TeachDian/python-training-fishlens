import tensorflow as tf

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')  # Adjust path

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
