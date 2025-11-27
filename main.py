# /main.py
'''
Punto de entrada principal del proyecto
'''

from src.utils.gpu_checker import check_gpu_status
from src.utils.preprocess_and_augmentation import preprocess_and_save_data_modular
from src.utils.view_web_dataset import view_web_dataset
from src.utils.rename_raw_dataset import renombrar_y_actualizar_raw_dataset
from src.train import train_model
from src.config import LATEST_MODEL_PATH
from time import sleep

# inferencia
from src.inference.predict_image import infer_image
# La lógica de carga de modelo ahora está en utils
from src.inference.inference_utils import load_model_safe, open_folder, start_saver_thread, start_ocr_thread
# Los scripts de inferencia ahora cargan su propio modelo, pero mantenemos la importación para referencia
from src.inference.predict_video import process_video
from src.inference.predict_webcam import run_webcam, list_cameras
from src.config import INPUT_FEED_DIR, OUTPUT_FEED_DIR
import glob
import os
from datetime import datetime
import queue


def clean_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clean_screen()
    check_gpu_status()
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
    input("Presione Enter para continuar...")
    clean_screen()


def menu():
    clean_screen()
    print('\n')
    print("=========================================")
    print("============ MENÚ PRINCIPAL =============")
    print("===== DETECTOR DE PLACAS COLOMBIANAS ====")
    print("=========================================\n")

    print(" 1. Preprocesar dataset (⚠ reemplaza)")
    print(" 2. Ver dataset web")
    print(" 3. Entrenar modelo nuevo")
    print(" 4. Inferir (imagen / video / webcam)")
    print(" 5. Utilidades | estandarizar nombres de dataset crudo")
    print(" 6. Finalizar\n")

    option = input("Seleccione opción (1/2/3/4/5): ")
    clean_screen()
    return option


def main():

    option = menu()

    # =======================================================
    # 1. Preprocesamiento
    # =======================================================
    if option == "1":
        print("Estás a punto de preprocesar el dataset. Esto REEMPLAZARÁ cualquier dato preprocesado previo.")
        print("¿Deseas continuar? (s/n)")
        confirm = input().lower()
        if confirm != 's':
            print("Preprocesamiento cancelado.")
            sleep(2)
            return main()
        preprocess_and_save_data_modular()
        return main()

    # =======================================================
    # 2. Ver dataset
    # =======================================================
    elif option == "2":
        view_web_dataset()
        return main()

    # =======================================================
    # 3. Entrenar
    # =======================================================
    elif option == "3":
        print("     Iniciando entrenamiento de modelo nuevo...")
        print("     Esto puede tardar un tiempo dependiendo del hardware.\n")
        # Establecer numero de epochs, batch size y learning rate aquí si se desea
        print("     Parámetros por defecto: Epochs=200, Batch Size=8, Learning Rate=0.001")
        personalizar = input("\n     Desea personalizar los parámetros de entrenamiento? (s/n): ").lower()
        if personalizar == 's':
            try:
                epochs = int(input("\n      Ingrese el número de epochs (por defecto 200): ") or 200)
                print("\n      Seleccione el tamaño del batch acorde a la memoria de su GPU/CPU.")
                print("      Tamaños comunes: 4, 8, 16, 32.")
                print("      Si experimenta errores de memoria, intente con un tamaño menor.")
                batch_size = int(input("\n      Ingrese el tamaño de batch (por defecto 8): ") or 8)
                learning_rate = float(input("      Ingrese la tasa de aprendizaje (por defecto 0.001): ") or 0.001)
                train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
            except ValueError:
                print("     Entrada inválida. Usando parámetros por defecto.")
                train_model()
        else:
            print("\n     Usando parámetros por defecto.")
            train_model()
        return main()

    # =======================================================
    # 4. INFERENCIA
    # =======================================================
    elif option == "4":
        # cargar modelo automáticamente UNA sola vez
        try:
            auto_load = input("\n     Desea cargar el último modelo entrenado automáticamente? (s/n): ").lower()
            if auto_load == 's':
                print(f"    Cargando modelo automáticamente desde: {LATEST_MODEL_PATH}")
                sleep(2)
                model = load_model_safe() # Ya no necesita path, lo busca automáticamente
            else:
                #listar modelos disponibles automaticamente y escoger, que el usuario elija no pedir path
                print("    Escoja el modelo a usar de la siguiente lista:")
                root_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"02_models")
                subdirs = [d for d in os.listdir(root_model_dir)
                            if os.path.isdir(os.path.join(root_model_dir, d))]
                if not subdirs:
                    print("    No se encontraron modelos entrenados.")
                    input("Presione Enter para continuar...")
                    return main()
                print("    Modelos disponibles:")
                for i, d in enumerate(sorted(subdirs), 1):
                    print(f"     {i}. {d}")
                choice = input(f"   Seleccione modelo (1-{len(subdirs)}): ")
                try:
                    choice_idx = int(choice) - 1
                    if choice_idx < 0 or choice_idx >= len(subdirs):
                        raise ValueError
                    selected_model_dir = os.path.join(root_model_dir, sorted(subdirs)[choice_idx])
                    model_path = os.path.join(selected_model_dir, "detector_model.keras")
                    print(f"    Cargando modelo desde: {model_path}")
                    sleep(2)
                    model = load_model_safe(model_path=model_path)
                except (ValueError, IndexError):
                    print("    Selección inválida.")
                    input("    Presione Enter para continuar...")
                    return main()
                if not os.path.exists(root_model_dir):
                    print("    No se encontraron modelos entrenados.")
                    input("    Presione Enter para continuar...")
                    return main()
                
                
        except Exception as e:
            print("    No se pudo cargar el modelo:", e)
            input("    Presione Enter para continuar...")
            return main()
        # =======================================================
        print("Seleccione modo:")
        print(" 1. Imagen")
        print(" 2. Video")
        print(" 3. Webcam")
        print(" 4. Volver al menú principal\n")
        mode = input("Elija (1/2/3/4): ")
        clean_screen()

        # ------------------ IMAGEN -------------------
        if mode == "1":
            print("Seleccion de modo: Imagen")
            sleep(2)
            in_dir = INPUT_FEED_DIR

            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            files = sorted([f for f in glob.glob(in_dir + "/*") if f.lower().endswith(exts)])

            if not files:
                print("No hay imágenes en INPUT_FEED_DIR")
            else:
                # Crear una única carpeta de salida para todo el lote de imágenes
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_output_dir = os.path.join(OUTPUT_FEED_DIR, ts)
                os.makedirs(unique_output_dir, exist_ok=True)
                print(f"Las imágenes procesadas se guardarán en: {unique_output_dir}")
                sleep(2)

                # Iniciar hilos de guardado y OCR para el lote de imágenes
                saver_q = queue.Queue()
                ocr_q = queue.Queue()
                ocr_results = {}
                saver_thread, stop_token = start_saver_thread(saver_q, os.path.join(unique_output_dir, 'metadata.csv'))
                ocr_thread, ocr_stop_token = start_ocr_thread(ocr_q, ocr_results)

                for f in files:
                    print(f"\n--- Procesando imagen: {os.path.basename(f)} ---")
                    # Pasar las colas y el diccionario de resultados a la función
                    infer_image(model, f, out_dir=unique_output_dir, saver_q=saver_q, ocr_q=ocr_q, ocr_results=ocr_results)
                
                # Detener hilos y esperar a que terminen
                saver_q.put(stop_token); saver_thread.join(timeout=15); saver_q.join()
                ocr_q.put(ocr_stop_token); ocr_thread.join(timeout=15); ocr_q.join()
                
                # Abrir la galería de resultados al finalizar el lote
                open_folder(unique_output_dir)
            input("Presione Enter para continuar...")
            return main()

        # ------------------ VIDEO -------------------
        elif mode == "2":
            print("Seleccion de modo: Video")
            sleep(2)
            in_dir = INPUT_FEED_DIR
            out_dir = OUTPUT_FEED_DIR
            os.makedirs(out_dir, exist_ok=True)

            # Buscar todos los videos en el directorio de entrada
            vext = ('.mp4', '.avi', '.mov', '.mkv')
            video_files = sorted([f for f in glob.glob(in_dir + "/*") if f.lower().endswith(vext)])

            if not video_files:
                print(f"No se encontraron videos en la carpeta: {in_dir}")
                sleep(2)
            else:
                print(f"Se encontraron {len(video_files)} videos. Comenzando procesamiento...")
                sleep(2)
                last_output_dir = None
                for video_path in video_files:
                    try:
                        print(f"\n--- Procesando: {os.path.basename(video_path)} ---")
                        # La función ahora devuelve la ruta de la carpeta de salida
                        last_output_dir = process_video(
                            model=model,
                            video_path=video_path, # Pasar la ruta del video actual
                            display=True
                        )
                    except Exception as e:
                        print(f"Error al procesar el video {video_path}: {e}")
                        sleep(2)
                
                # Abrir la galería de la última ejecución al finalizar el lote
                if last_output_dir:
                    open_folder(last_output_dir)
                        
            input("Enter para continuar...")
            return main()

        # ------------------ WEBCAM -------------------
        elif mode == "3":
            print("Seleccion de modo: Webcam")
            input("Presione Enter para continuar...")
            cams = list_cameras()

            if not cams:
                print("No se detectaron cámaras.")
                input("Enter para continuar...")
                return main()

            print("Cámaras detectadas:")
            for c in cams:
                print(" -", c)

            sel = input(f"Seleccione cámara (enter={cams[0]}): ")
            cam = int(sel) if sel.strip() else cams[0]

            # El script de webcam ahora carga su propio modelo
            run_webcam(cam_index=cam)

            input("Enter para continuar...")
            return main()
        # ------------------ VOLVER AL MENÚ -------------------
        elif mode == "4":
            return main()
        else:
            print("Opción inválida.")
            return main()
        
    # =======================================================
    # 5. ESTANDARIZAR NOMBRES DE DATASET CRUDO
    # =======================================================
    elif option == "5":
        renombrar_y_actualizar_raw_dataset()
        return main()
    # =======================================================
    # 6. SALIR
    # =======================================================
    elif option == "6":
        print("Hasta luego!")
        exit(0)

    else:
        print("Opción inválida.")
        return main()


if __name__ == "__main__":
    print_header()
    main()
