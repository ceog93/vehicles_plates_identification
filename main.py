# /main.py
'''
Punto de entrada principal delmproyecto
'''

from src.generate_dataset import generar_dataset
from src.train import train_model
from src.infer_video import run_video_detection
def main():
    print("=== DETECTOR DE PLACAS COLOMBIANAS ===")
    print("1. Generar dataset sintético")
    print("2. Entrenar modelo")
    print("3. Inferir en video")
    option = input("Seleccione opción (1/2/3): ")

    if option == "1":
        generar_dataset()
    elif option == "2":
        train_model()
    elif option == "3":
        video_path = input("Ruta del video: ")
        run_video_detection(video_path)
    else:
        print("Opción inválida.")

