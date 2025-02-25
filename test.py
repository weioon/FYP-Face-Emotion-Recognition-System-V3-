# Create verify_gpu.py
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices('GPU'))

if tf.test.is_gpu_available():
    print("GPU is working!")