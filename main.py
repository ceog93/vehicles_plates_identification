# /main.py
'''
Punto de entrada principal del proyecto
'''
from src.utils.preprocess_and_augmentation import preprocess_and_save_data_modular
from src.utils.view_web_dataset import view_web_dataset
from src.train import train_model
from src.infer_video import run_video_detection
import os

def clean_screen():
    #borrar pantalla
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clean_screen()
    print('\n')
    print("########################################")
    print("        INTELIGENCIA ARTIFICIAL")
    print("            REDES NEURONALES")
    print("             DEEP LEARNING")
    print("              TENSORFLOW")
    print("          CNN PARA DETECCIÓN")
    print("          PLACAS VEHICULARES")
    print("########################################")
    print('\n')
    input(' Presione enter para continuar...')
    clean_screen()

def menu():
    clean_screen()
    print('\n')
    print("=========================================")
    print("============ MENÚ PRINCIPAL =============")
    print("===== DETECTOR DE PLACAS COLOMBIANAS ====")
    print("=========================================\n")
    
    print("     1. (¡PRECAUCIÓN!) Preprocesar y aumentar dataset (Reemplaza dataset actual)")
    print("     2. Ver dataset procesado en visor web")
    print("     3. Entrenar modelo")
    print("     4. Inferir en video")
    print("     5. Finalizar programa\n")
    option = input("        Seleccione opción (1/2/3/4/5): ")
    clean_screen()
    return option

def main():
    
    option = menu()

    if option == "1":
        preprocess_and_save_data_modular() # Se puede ajustar parámetros dentro de la función
        main()
    elif option == "2":
        view_web_dataset()
        main()
        pass
    elif option == "3":
        train_model() # Se puede ajustar parámetros dentro de la función
        main()
    elif option == "4":
        video_path = "data/videos/video_prueba.mp4"
        run_video_detection(video_path)
        main()
    elif option == "5":
        print("Finalizando programa. ¡Hasta luego!")
        exit(0)
    else:
        print("Opción inválida.")
        main()

if __name__ == "__main__":
    print_header()
    main()

