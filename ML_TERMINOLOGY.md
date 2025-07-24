# ML Training: Key Terms & Best Practices for Pawikan Sentinel

This guide provides a quick reference for the essential machine learning terminology and best practices we are using to train the custom sea turtle detection model for the Pawikan Sentinel project.

---

## 🧠 Key Machine Learning Terminologies

**Epoch**
: One complete pass of the entire training dataset through the model. If the dataset has 1,000 images, one epoch is finished when the model has seen all 1,000 images once.

**Batch Size**
: The number of training examples utilized in one iteration. Instead of processing the entire dataset at once, the model processes it in smaller "batches." A larger batch size requires more memory but can lead to faster training.

**Learning Rate**
: A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. A learning rate that is too high can cause the model to make erratic jumps and not converge, while one that is too low will make the training process very slow.

**Transfer Learning**
: The technique we are using. It involves taking a model that has been pre-trained on a very large dataset (like COCO, which has millions of images of common objects) and then fine-tuning it on our smaller, specific dataset (sea turtle images). This saves enormous amounts of time and results in a more accurate model.

**Overfitting vs. Underfitting**
:
- **Overfitting**: The model learns the training data *too well*, including its noise and specific details. As a result, it performs poorly on new, unseen data (like a new video feed). This happens when you train for too many epochs.
- **Underfitting**: The model has not learned the patterns in the training data well enough. It performs poorly on both the training data and new data. This happens when you train for too few epochs.

**Validation Set**
: A subset of the dataset that is not used for training the model but is used to evaluate the model's performance after each epoch. This helps us monitor for overfitting and decide when to stop training.

**mAP (mean Average Precision)**
: The primary metric for evaluating the accuracy of object detection models like YOLOv5. It represents the model's accuracy in drawing the correct bounding box around an object and assigning it the correct class label. A higher mAP is better.

**Precision & Recall**
:
- **Precision**: Of all the predictions the model made, how many were actually correct? (Measures false positives).
- **Recall**: Of all the actual turtles in the images, how many did the model successfully detect? (Measures false negatives).

**ONNX (Open Neural Network Exchange)**
: An open format built to represent machine learning models. We convert our trained model to ONNX as an intermediate step before converting it to TensorFlow Lite.

**TensorFlow Lite (TFLite)**
: A special format designed to run machine learning models on mobile and embedded devices like the Raspberry Pi. TFLite models are smaller, faster, and more power-efficient.

**INT8 Quantization**
: An optimization technique that reduces the precision of the numbers used in the model (from 32-bit floating-point numbers to 8-bit integers). This makes the model significantly smaller and faster on the Raspberry Pi's CPU, with a minimal drop in accuracy.

---

## ✅ Best Practices for Training Our Model

1.  **Always Start with Pretrained Weights**
    : We will always use the `yolov5n.pt` weights as our starting point. This leverages the power of transfer learning and is the single most important factor for success.

2.  **Use a Validation Set to Monitor Performance**
    : Never train without a validation set. This is our only reliable way to know how the model will perform in the real world and to detect overfitting.

3.  **Monitor Training with TensorBoard**
    : YOLOv5 automatically integrates with TensorBoard. We should use it to visualize metrics like mAP, precision, recall, and loss curves. This gives us insight into how the training is progressing.

4.  **Use Early Stopping**
    : Instead of guessing the number of epochs, we will configure the training to stop automatically when the validation performance (e.g., mAP) stops improving for a set number of "patience" epochs (e.g., 20). This prevents overfitting and saves time.

5.  **Use Data Augmentation**
    : YOLOv5 has built-in data augmentation (e.g., rotating, scaling, flipping images). This artificially increases the diversity of our training data, making the model more robust and less likely to overfit. We should ensure this is enabled during training.

6.  **Tune Hyperparameters Carefully**
    : The most critical hyperparameter is the **learning rate**. If we ever need to re-train or fine-tune, we should start with a low learning rate (e.g., 0.001) to avoid destroying the pre-trained features.

7.  **Keep Track of Experiments**
    : Every training run is an experiment. The YOLOv5 framework helps by saving results in separate `exp` directories. We should make notes of what was changed in each experiment (e.g., dataset version, hyperparameters).

8.  **Benchmark Before Deploying**
    : After training, we must convert the model to TFLite with INT8 quantization and benchmark its speed and accuracy on the Raspberry Pi itself to ensure it meets our performance requirements.
