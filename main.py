# /main.py
'''
Punto de entrada principal del proyecto
'''
from src.utils.gpu_checker import check_gpu_status
from src.utils.preprocess_and_augmentation import preprocess_and_save_data_modular
from src.utils.view_web_dataset import view_web_dataset
from src.train import train_model
# Importar funciones de inferencia existentes
from src.inference.predict_image import infer_image
from src.inference.predict_video import process_video, load_model_safe as load_model_safe_vid, find_latest_model_in_models_dir
from src.inference.predict_webcam import run_webcam
from src.infer_video import run_video_detection
from src.config import INPUT_FEED_DIR, OUTPUT_FEED_DIR
import glob
import os

def clean_screen():
    #borrar pantalla
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
    print("     3. Entrenar modelo nuevo")
    print("     4. Inferir placas (Imagen / Video / Webcam)")
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
        # Nuevo sub-menú: elegir entre inferir imagen, video o webcam
        print("Seleccione modo de inferencia:")
        print(" 1. Imagen")
        print(" 2. Video")
        print(" 3. Webcam")
        mode = input("Elija (1/2/3): ")

        # Cargar modelo una vez; priorizar la ruta del último modelo existente
        try:
            auto_model = find_latest_model_in_models_dir()
            if auto_model:
                model = load_model_safe_vid(auto_model)
            else:
                model = load_model_safe_vid(None)
        except Exception as e:
            print("No se pudo cargar el modelo:", e)
            main()
            return

        if mode == "1":
            # Procesar por defecto todas las imágenes en INPUT_FEED_DIR
            img_path = INPUT_FEED_DIR
            out_dir = OUTPUT_FEED_DIR
            os.makedirs(out_dir, exist_ok=True)

            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            files = [f for f in sorted(glob.glob(os.path.join(img_path, '*'))) if f.lower().endswith(exts)]
            if not files:
                print("No se encontraron imágenes en INPUT_FEED_DIR.")
            for f in files:
                base = os.path.basename(f)
                out_p = os.path.join(out_dir, f"det_{base}")
                print(f"Procesando imagen: {f} -> {out_p}")
                try:
                    infer_image(model, f, out_path=out_p)
                except Exception as e:
                    print('Error procesando', f, e)

            input('Presione enter para continuar...')
            main()
        elif mode == "2":
            # Procesar por defecto todos los videos en INPUT_FEED_DIR
            video_path = INPUT_FEED_DIR
            out_dir = OUTPUT_FEED_DIR
            os.makedirs(out_dir, exist_ok=True)

            vexts = ('.mp4', '.avi', '.mov', '.mkv')
            files = [f for f in sorted(glob.glob(os.path.join(video_path, '*'))) if f.lower().endswith(vexts)]
            if not files:
                print("No se encontraron videos en INPUT_FEED_DIR.")
            for f in files:
                base = os.path.basename(f)
                print(f"Procesando video: {f} (se guardará en {out_dir} con timestamp)")
                try:
                    # dejar que process_video cree el nombre del video con timestamp y mostrar en tiempo real
                    # Parámetros: aspect_ratio_min relajado para aceptar placas, confirm_frames para estabilidad
                    process_video(model, f, out_video_path=None, display=True,
                                  confirm_frames=3, iou_thresh=0.45, min_area=1500,
                                  aspect_ratio_min=1.0, aspect_ratio_max=10.0)
                except Exception as e:
                    print('Error procesando', f, e)

            input('Presione enter para continuar...')
            main()
        elif mode == "3":
            # Listar cámaras y permitir selección interactiva
            from src.inference.predict_webcam import list_cameras
            cams = list_cameras(max_index=8)
            if not cams:
                print("No se detectaron cámaras disponibles.")
                input('Presione enter para continuar...')
                main()
                return
            print("Cámaras detectadas:")
            for idx in cams:
                print(f" - {idx}")
            sel = input(f"Seleccione índice de cámara (enter para {cams[0]}): ")
            try:
                cam_index = int(sel) if sel.strip() != "" else cams[0]
            except Exception:
                cam_index = cams[0]
            # Guardado automático: guardar video y frames con confianza >= 90%
            save = True
            run_webcam(model, cam_index=cam_index, save_output=save, save_video=True, save_high_conf=True, high_conf_thresh=1.0)
            input('Presione enter para continuar...')
            main()
        else:
            print("Opción inválida en sub-menú de inferencia.")
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

