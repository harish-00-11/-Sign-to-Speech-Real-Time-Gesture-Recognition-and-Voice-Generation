import pandas as pd
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix


# Enhanced Configuration Class
class ModelConfig:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 10  # Increased to allow more training with callbacks
    FINE_TUNE_EPOCHS = 10  # Increased for better fine-tuning
    DATA_DIR = "Data_sign"
    BASE_LEARNING_RATE = 0.0001  # Reduced initial learning rate
    FINE_TUNE_LEARNING_RATE = 1e-6  # Reduced fine-tuning rate
    MIN_LR = 1e-7  # Minimum learning rate
    L2_REG = 0.001  # L2 regularization factor

    @staticmethod
    def get_data_generator():
        return ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=30,  # Increased rotation
            width_shift_range=0.2,  # Increased shift range
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,  # Added vertical flip control
            validation_split=0.2,
            brightness_range=[0.8, 1.2],  # Wider brightness range
            channel_shift_range=20,  # Added channel shifting
            fill_mode='nearest'  # Define how to fill new pixels
        )


def create_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.75  # Smaller model variant
    )
    base_model.trainable = False  # Freeze base layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu',
              kernel_regularizer=l2(ModelConfig.L2_REG),
              activity_regularizer=l2(ModelConfig.L2_REG / 2)),
        Dropout(0.6),  # Increased dropout
        BatchNormalization(),
        Dense(128, activation='relu',
              kernel_regularizer=l2(ModelConfig.L2_REG)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=ModelConfig.BASE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model, base_model


def main():
    # Create directories if they don't exist
    os.makedirs("Model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    datagen = ModelConfig.get_data_generator()

    train_data = datagen.flow_from_directory(
        ModelConfig.DATA_DIR,
        target_size=(ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE),
        batch_size=ModelConfig.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42  # Fixed seed for reproducibility
    )

    val_data = datagen.flow_from_directory(
        ModelConfig.DATA_DIR,
        target_size=(ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE),
        batch_size=ModelConfig.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    class_labels = list(train_data.class_indices.keys())
    with open("Model/Labels_ER.txt", "w") as f:
        for label in class_labels:
            f.write(f"{label}\n")

    model, base_model = create_model(train_data.num_classes)

    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,  # Increased patience
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001  # Minimum change to qualify as improvement
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,  # Increased patience
        verbose=1,
        min_lr=ModelConfig.MIN_LR,
        min_delta=0.001
    )

    model_checkpoint = ModelCheckpoint(
        'Model/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    tensorboard = TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        update_freq='epoch'
    )

    # Add class weight calculation for imbalanced datasets
    class_counts = np.bincount(train_data.classes)
    class_weights = {i: 1. / count for i, count in enumerate(class_counts)}
    total = sum(class_weights.values())
    class_weights = {i: (weight / total) * len(class_counts) for i, weight in class_weights.items()}

    print("\nTraining initial model...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=ModelConfig.INITIAL_EPOCHS,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tuning with more layers unfrozen
    print("\nFine-tuning model...")
    base_model.trainable = True
    for layer in base_model.layers[:80]:  # Reduced number of frozen layers
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=ModelConfig.FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    fine_tune_history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=ModelConfig.FINE_TUNE_EPOCHS,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],
        class_weight=class_weights,
        verbose=1
    )

    # Save model in Keras format (recommended)
    model.save("Model/sign_language_model_ER.keras")

    # Evaluation
    print("\nEvaluating model...")
    y_true = val_data.classes
    y_pred = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Save comprehensive reports
    report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("Model/classification_report_ER.csv", index=True)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("Model/confusion_matrix_ER.png")
    plt.close()

    print("\nTraining Complete! Model and reports saved.")


if __name__ == "__main__":
    main()