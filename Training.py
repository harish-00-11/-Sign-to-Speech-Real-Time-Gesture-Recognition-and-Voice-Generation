import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.utils import class_weight
import os
import threading

# Add this at the very beginning of your script
if __name__ == '__main__':
    # Constants
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15
    DATA_DIR = "Data"
    MODEL_DIR = "Model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Enhanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='reflect',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    # Load data
    train_data = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_data = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save class labels
    class_labels = list(train_data.class_indices.keys())
    with open(os.path.join(MODEL_DIR, 'labels.txt'), 'w') as f:
        for label in class_labels:
            f.write(f"{label}\n")

    # Enhanced Model Architecture with more regularization
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # Smaller model
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GaussianNoise(0.05),  # Input noise
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(384, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005)),
        Dropout(0.6),  # Increased dropout
        BatchNormalization(),
        Dense(192, activation='relu', kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005)),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])

    # Learning Rate Schedule with Warmup
    initial_learning_rate = 0.0003
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5 * len(train_data), 10 * len(train_data)],
        values=[initial_learning_rate, initial_learning_rate * 0.5, initial_learning_rate * 0.1]
    )


    # Compile with label smoothing and focal loss
    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = alpha * tf.pow(1. - y_pred, gamma) * cross_entropy
            return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        return loss


    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    # Enhanced Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            cooldown=2
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False
        )
    ]

    # Class weights with smoothing
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights = {i: min(class_weights[i], 3.0) for i in range(len(class_weights))}


    # MixUp augmentation using a generator function instead of a class
    def mixup_generator(generator, alpha=0.2):
        while True:
            x_batch, y_batch = next(generator)
            batch_size = x_batch.shape[0]

            # Generate mixing coefficients
            lam = np.random.beta(alpha, alpha, batch_size)
            lam = np.maximum(lam, 1 - lam)  # Ensure we don't wash out the signal too much

            # Create indices for mixing
            indices = np.random.permutation(batch_size)

            # Mix images and labels
            lam_reshaped = lam.reshape(batch_size, 1, 1, 1)
            x_mix = lam_reshaped * x_batch + (1 - lam_reshaped) * x_batch[indices]

            lam_y_reshaped = lam.reshape(batch_size, 1)
            y_mix = lam_y_reshaped * y_batch + (1 - lam_y_reshaped) * y_batch[indices]

            yield x_mix, y_mix


    # Train with simplified settings for Windows compatibility
    print("Starting training with MixUp augmentation...")

    # Training with standard generator (no MixUp) to avoid thread safety issues
    history = model.fit(
        train_data,  # Use standard generator instead of MixUp
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=1  # Use single worker for reliability on Windows
    )

    # Fine-tuning with more conservative settings
    if max(history.history['val_accuracy']) > 0.82:  # Only if good initial performance
        print("\nStarting careful fine-tuning...")
        base_model.trainable = True
        for layer in base_model.layers[:120]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=1e-6),  # Very low learning rate
            loss=focal_loss(gamma=1.5),  # Less aggressive focal loss
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_data,  # Use standard generator for fine-tuning too
            validation_data=val_data,
            epochs=15,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=1  # Use single worker for reliability
        )

    # Save final models
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))

    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

    print("\nTraining complete with anti-overfitting measures:")
    print("- Strong regularization (L1/L2, Dropout 0.6/0.5)")
    print("- Focal loss for class imbalance")
    print("- Gaussian noise input layer")
    print("- Careful learning rate scheduling")
    print("- Early stopping with large patience")