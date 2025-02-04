import tensorflow as tf
from config import IMG_SIZE, NUM_CHANNELS


def create_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, NUM_CHANNELS)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'), 
        tf.keras.layers.GlobalAveragePooling2D(),  # Better than Flatten for classification
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Observation: I have structured classes as "fruit_condition"
# Recommendation: Consider multi-task learning
def create_multitask_model():
    base = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg')
    
    fruit_input = tf.keras.Input(shape=(*IMG_SIZE, 3))
    features = base(fruit_input)
    
    # Task 1: Fruit type classification (14 classes)
    fruit_type = tf.keras.layers.Dense(14, activation='softmax', name='fruit_type')(features)
    
    # Task 2: Health status (binary)
    health_status = tf.keras.layers.Dense(1, activation='sigmoid', name='health_status')(features)
    
    return tf.keras.Model(inputs=fruit_input, outputs=[fruit_type, health_status])