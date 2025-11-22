import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from src.config import IMG_SIZE, LEARNING_RATE

# ============================
# FUNCIÓN: CONSTRUIR MODELO (Sin cambios)
# ============================

def build_mobile_net_detector(img_size=IMG_SIZE,learning_rate=LEARNING_RATE):
    ''' Construir modelo usando MobileNetV2 '''
    base_model = MobileNetV2(
        weights='none', # none para entrenamiento desde cero
        include_top=False, 
        input_shape=(img_size[0], img_size[1], 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False) 
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x) 
    
    outputs = layers.Dense(4, activation='sigmoid', name='bounding_box_output')(x) 
    
    model = models.Model(inputs, outputs) 
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
        loss=tf.keras.losses.Huber(), 
        metrics=['mae', 'mse']
    ) 
    return model