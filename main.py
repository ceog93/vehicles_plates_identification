# /main.py
'''
Punto de entrada principal del proyecto
'''

from src.utils.gpu_checker import check_gpu_status
from src.utils.preprocess_and_augmentation import preprocess_and_save_data_modular
from src.utils.view_web_dataset import view_web_dataset
from src.train import train_model
from src.config import LATEST_MODEL_PATH
from time import sleep

# inferencia
from src.inference.predict_image import infer_image
from src.inference.predict_video import (
    process_video as run_video_detection,
    load_model_safe,
    find_latest_model_in_models_dir
)
from src.inference.predict_webcam import run_webcam

from src.config import INPUT_FEED_DIR, OUTPUT_FEED_DIR
import glob
import os


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
    print(" 5. Finalizar\n")

    option = input("Seleccione opción (1/2/3/4/5): ")
    clean_screen()
    return option


def main():

    option = menu()

    # =======================================================
    # 1. Preprocesamiento
    # =======================================================
    if option == "1":
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
        train_model()
        return main()

    # =======================================================
    # 4. INFERENCIA
    # =======================================================
    elif option == "4":

        print("Seleccione modo:")
        print(" 1. Imagen")
        print(" 2. Video")
        print(" 3. Webcam")

        mode = input("Elija (1/2/3): ")

        # cargar modelo automáticamente UNA sola vez
        try:
            print(f"Cargando modelo automáticamente desde: {LATEST_MODEL_PATH}")
            sleep(2)
            model = load_model_safe(LATEST_MODEL_PATH)
        except Exception as e:
            print("No se pudo cargar el modelo:", e)
            input("Presione Enter para continuar...")
            return main()

        # ------------------ IMAGEN -------------------
        if mode == "1":
            print("Seleccion de modo: Imagen")
            sleep(2)
            in_dir = INPUT_FEED_DIR
            out_dir = OUTPUT_FEED_DIR
            os.makedirs(out_dir, exist_ok=True)

            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            files = sorted([f for f in glob.glob(in_dir + "/*") if f.lower().endswith(exts)])

            if not files:
                print("No hay imágenes en INPUT_FEED_DIR")
            else:
                for f in files:
                    out_f = os.path.join(out_dir, "det_" + os.path.basename(f))
                    infer_image(model, f, out_path=out_f)
                    print(f"Imagen procesada: {f} -> {out_f}")
            return main()

        # ------------------ VIDEO -------------------
        elif mode == "2":
            print("Seleccion de modo: Video")
            sleep(2)
            in_dir = INPUT_FEED_DIR
            out_dir = OUTPUT_FEED_DIR
            os.makedirs(out_dir, exist_ok=True)

            vext = ('.mp4', '.avi', '.mov', '.mkv')
            files = sorted([f for f in glob.glob(in_dir + "/*") if f.lower().endswith(vext)])

            if not files:
                print("No hay videos en INPUT_FEED_DIR")
                input("Enter para continuar...")
                return main()
            else:
                for vid in files:
                    print(f"Procesando video (OCR colombiano multiplaca): {vid}")
                    try:
                        # CORRECCIÓN: 'input_video' se cambia a 'video_path'
                        run_video_detection(
                            model=model,
                            video_path=vid, 
                            # Cambié 'output_folder=out_dir' por 'out_video_path'
                            # ya que la función process_video solo acepta out_video_path.
                            # El manejo del path de salida se hace dentro de process_video si es None.
                            out_video_path=None, 
                            display=True, # show_window se llama 'display' en process_video
                            iou_thresh=0.45, # nms_thresh se llama 'iou_thresh'
                            min_area=1400,
                            # Los argumentos 'min_conf', 'max_area', 'allow_vertical', y 'ocr_lang' 
                            # no existen en la definición de process_video que proporcionaste, 
                            # por lo que deben ser removidos o ajustados a los parámetros de process_video.
                            max_missed=5,
                            confirm_frames=3,
                            aspect_ratio_min=2.0,
                            aspect_ratio_max=6.0
                        )
                    except Exception as e:
                        print("Error procesando", vid, e)
                        sleep(2)
                        

            input("Enter para continuar...")
            return main()

        # ------------------ WEBCAM -------------------
        elif mode == "3":
            print("Seleccion de modo: Webcam")
            input("Presione Enter para continuar...")
            from src.inference.predict_webcam import list_cameras
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

            run_webcam(model, cam_index=cam)

            input("Enter para continuar...")
            return main()

        else:
            print("Opción inválida.")
            return main()

    # =======================================================
    # 5. SALIR
    # =======================================================
    elif option == "5":
        print("Hasta luego!")
        exit(0)

    else:
        print("Opción inválida.")
        return main()


if __name__ == "__main__":
    print_header()
    main()
