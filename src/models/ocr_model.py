from tensorflow.keras import layers, models

def build_ocr_model(input_shape=(32, 32, 1), num_classes=36):
    """
    CNN ligera para clasificación de caracteres (0-9, A-Z).
    Entrenada desde cero (Scratch).
    """
    model = models.Sequential([
        # Capa de entrada
        layers.Input(shape=input_shape),
        
        # Bloque 1: Extracción de rasgos básicos
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(), # Ayuda a estabilizar el entrenamiento
        
        # Bloque 2: Rasgos medios
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Evitar sobreajuste
        
        # Bloque 3: Rasgos complejos
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Clasificador (Fully Connected)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Salida: 36 neuronas (probabilidad para cada caracter)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # 'sparse' porque usaremos etiquetas enteras (0, 1, 2...)
                  metrics=['accuracy'])
    
    return model