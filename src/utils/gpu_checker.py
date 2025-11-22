# gpu_checker.py

import tensorflow as tf # type: ignore
import os

def check_gpu_status():
    """
    Verifica si TensorFlow puede detectar y usar una GPU, 
    y ejecuta una pequeña operación de matriz en ella para confirmación.
    
    Retorna True si la GPU está operativa y se ha configurado para su uso, False en caso contrario.
    """
    print("-" * 50)
    print("🚀 Iniciando verificación de estado de GPU...")
    
    # 1. Listar dispositivos físicos disponibles
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ Éxito: {len(gpus)} GPU(s) detectada(s) por TensorFlow.")
        print(f"Versión de TensorFlow: {tf.__version__}")
        print(f"Dispositivo(s) encontrado(s): {gpus}")
        
        try:
            # 2. Configurar el primer dispositivo GPU para su uso
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # 3. Ejecutar una operación simple en la GPU para confirmación
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            
            print(f"🧠 Operación Matriz (tf.matmul) realizada en /GPU:0.")
            print(f"Resultado de prueba: {c.numpy()}")
            print("--- La GPU está correctamente configurada y operativa para el trabajo. ---")
            return True
            
        except RuntimeError as e:
            # Esto maneja errores de inicialización o configuración después de la detección
            print(f"❌ Error al inicializar o usar la GPU (CUDA/cuDNN): {e}")
            return False
            
    else:
        print("❌ Falla: No se detectaron GPUs. Las operaciones usarán la CPU.")
        return False
    
    print("-" * 50)

if __name__ == "__main__":
    # Esto permite ejecutar el script directamente para una prueba rápida
    gpu_available = check_gpu_status()
    if not gpu_available:
        print("\nRevisa tu instalación de WSL2, Drivers de NVIDIA, CUDA Toolkit y cuDNN.")