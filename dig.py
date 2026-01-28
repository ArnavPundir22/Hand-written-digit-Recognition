import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 45
NUM_CLASSES = 10
DATASET_DIR = "dataset_emnist"
classes = list("0123456789")

# =========================
# CREATE OUTPUT DIRECTORIES
os.makedirs("outputs/model/epochs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/evaluation", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# =========================
# DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR + "/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    DATASET_DIR + "/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# DNN MODEL
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.summary()

# =========================
# COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS (VERY IMPORTANT)
checkpoint_all = tf.keras.callbacks.ModelCheckpoint(
    filepath="outputs/model/epochs/epoch_{epoch:02d}.h5",
    save_best_only=False,
    verbose=1
)

checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath="outputs/model/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    "outputs/training_log.csv",
    append=False
)

# =========================
# TRAIN
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint_all, checkpoint_best, csv_logger],
    verbose=1
)

# =========================
# SAVE FINAL MODEL
model.save("outputs/model/final_model.h5")

# =========================
# PLOT ACCURACY
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("outputs/plots/accuracy.png")
plt.close()

# =========================
# PLOT LOSS
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig("outputs/plots/loss.png")
plt.close()

# =========================
# EVALUATE
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# =========================
# CONFUSION MATRIX
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("outputs/evaluation/confusion_matrix.png")
plt.close()

# =========================
# CLASSIFICATION REPORT
report = classification_report(y_true, y_pred, target_names=classes)
with open("outputs/evaluation/classification_report.txt", "w") as f:
    f.write(report)

# =========================
# SAMPLE TEST PREDICTIONS
x_batch, y_batch = next(test_generator)
batch_preds = model.predict(x_batch)

plt.figure(figsize=(10,6))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(x_batch[i].reshape(28,28), cmap="gray")
    true_label = classes[np.argmax(y_batch[i])]
    pred_label = classes[np.argmax(batch_preds[i])]
    plt.title(f"T:{true_label} P:{pred_label}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/predictions/sample_predictions.png")
plt.close()

print("✅ EVERYTHING SAVED SUCCESSFULLY!")

