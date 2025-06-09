import os
import os.path
import yaml
import numpy as np
import cv2
from itertools import combinations
# Assuming src.charuco_calibration and src.extrinsic_calibration are in PYTHONPATH
from src.charuco_calibration import CharucoCalibration
from src.extrinsic_calibration import ExtrinsicCalib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_images(path, camera_disk_ids):
    """
    Carrega as imagens de cada câmera a partir do diretório especificado.
    'path' é o base_path.
    'camera_disk_ids' é uma lista de strings, cada uma no formato 'DISK_FOLDER_NAME/SIDE', 
    por exemplo, 'Passive/left'.
    A função espera que as imagens estejam em path/DISK_FOLDER_NAME/SIDE/*.jpg (ou outro formato de imagem)
    """
    images = {}
    for cam_id_on_disk in camera_disk_ids:  # ex: 'Passive/left'
        image_dir_path = os.path.join(path, cam_id_on_disk) # ex: base_path/Passive/left
        
        if not os.path.isdir(image_dir_path):
            print(f"Aviso: Diretório de imagens não encontrado: {image_dir_path}. Pulando {cam_id_on_disk}.")
            images[cam_id_on_disk] = []
            continue

        loaded_imgs = []
        try:
            image_files = sorted(os.listdir(image_dir_path))
            for img_name in image_files:
                full_img_path = os.path.join(image_dir_path, img_name)
                if os.path.isfile(full_img_path):
                    img = cv2.imread(full_img_path)
                    if img is not None:
                        loaded_imgs.append(img)
                    else:
                        print(f"Aviso: Não foi possível ler a imagem: {full_img_path}")
            images[cam_id_on_disk] = loaded_imgs
        except Exception as e:
            print(f"Erro ao carregar imagens de {image_dir_path}: {e}")
            images[cam_id_on_disk] = []
            
    return images

def charuco_calibration(base_path: str, save_path: str = 'camera-calibration/camera_parameters') -> None:
    """
    Realiza a calibração de múltiplos sistemas e câmeras usando um tabuleiro Charuco.
    Inclui calibração intrínseca e estéreo.
    
    :param base_path: Caminho principal onde estão as pastas dos sistemas (ex.: Passive, Active).
    :param save_path: Caminho para salvar os parâmetros de calibração estéreo.
    """
    charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))

    systems_config = [
        {'logical_name': 'SM2 Passivo', 'disk_folder': 'Passive'}, # Mapeia para a pasta 'Passive' no disco
        {'logical_name': 'SM3 e SM4 Ativo', 'disk_folder': 'Active'}  # Mapeia para a pasta 'Active' no disco
    ]

    for config in systems_config:
        logical_name = config['logical_name']
        disk_folder = config['disk_folder']
        # system_data_path agora aponta para a pasta base do sistema (ex: base_path/Passive)
        # A função load_image da CharucoCalibration deve lidar com as subpastas 'left' e 'right' internamente.
        system_data_path = os.path.join(base_path, disk_folder)

        if not os.path.exists(system_data_path):
            print(f"Diretório de dados '{disk_folder}' para o sistema '{logical_name}' não encontrado em {base_path}. Pulando...")
            continue

        print(f"Processando sistema '{logical_name}' (dados de '{system_data_path}')...")

        # A função load_image espera o caminho para a pasta do sistema (ex: 'Passive')
        # e internamente procura por 'Passive/left' e 'Passive/right'.
        charuco.load_image(images_path=system_data_path, visualize=False)

        charuco.detect_charuco_board(camera='left', visualize=False)
        charuco.detect_charuco_board(camera='right', visualize=False)

        charuco.calibration_camera(camera='left')
        charuco.calibration_camera(camera='right')

        print(f"Realizando calibração estéreo para o sistema '{logical_name}'...")
        charuco.stereo_calibration()

        save_file = os.path.join(save_path, f"{logical_name}_stereo_charuco.yaml")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        charuco.save_camera_params(save_file)
        print(f"Parâmetros estéreo do sistema '{logical_name}' salvos em {save_file}")

    print("Calibração Charuco concluída para todos os sistemas configurados.")


def charuco_markers(base_path: str, Calibration=False, mad_multiplier=3):
    """
    Define os parâmetros das câmeras e calcula as transformações relativas.
    Automatiza a calibração intrínseca se necessário.
    
    :param base_path: Caminho principal onde estão as pastas dos sistemas (ex.: Passive, Active).
    :param Calibration: Se True, realiza a calibração intrínseca automaticamente.
    :param mad_multiplier: Fator multiplicativo para o Desvio Absoluto Mediano (MAD).
    """
    extrinsic = ExtrinsicCalib(marker_size=(28, 18))
    
    cam_dict = {
        'SM2 Passivo': {
            'disk_folder': 'Passive', # Pasta no disco para dados do SM2 Passivo
            'ids': ['SM2 Passivo/left', 'SM2 Passivo/right'], # IDs lógicos
            'params': [ 
                {'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03], [0.0000e+00, 4.3180e+03, 1.0937e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]], 'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]},
                {'kk': [[4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]], 'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]}
            ]
        },
        'SM3 e SM4 Ativo': {
            'disk_folder': 'Active', # Pasta no disco para dados do SM3 e SM4 Ativo
            'ids': ['SM3 e SM4 Ativo/left', 'SM3 e SM4 Ativo/right'], # IDs lógicos
            'params': [ 
                {'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03], [0.0000e+00, 4.3180e+03, 1.0937e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]], 'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]},
                {'kk': [[4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]], 'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]}
            ]
        }
    }

    for system_logical_name, system_config in cam_dict.items():
        disk_folder_for_data = system_config['disk_folder'] # ex: 'Passive' ou 'Active'
        
        if Calibration:
            charuco_calib_obj = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))
            print(f"Calibrando sistema '{system_logical_name}' (usando dados da pasta: '{disk_folder_for_data}')...")
            # system_actual_data_path é o caminho para a pasta do sistema (ex: base_path/Passive)
            system_actual_data_path = os.path.join(base_path, disk_folder_for_data)
            
            calibration_successful_for_system = False
            if not os.path.exists(system_actual_data_path):
                print(f"  Atenção: Diretório de dados {system_actual_data_path} não encontrado.")
            else:
                try:
                    # load_image espera o caminho para a pasta do sistema (ex: 'Passive')
                    charuco_calib_obj.load_image(images_path=system_actual_data_path, visualize=False)
                    charuco_calib_obj.detect_charuco_board(camera='left', visualize=False)
                    charuco_calib_obj.calibration_camera(camera='left')
                    charuco_calib_obj.detect_charuco_board(camera='right', visualize=False)
                    charuco_calib_obj.calibration_camera(camera='right')
                    calibration_successful_for_system = True
                except Exception as e:
                    print(f"  Erro durante a calibração intrínseca para '{system_logical_name}' (dados de '{disk_folder_for_data}'): {e}")

            if calibration_successful_for_system:
                print(f"  Calibração intrínseca para '{system_logical_name}' bem-sucedida.")
                left_cam_logical_id = system_config['ids'][0]
                right_cam_logical_id = system_config['ids'][1]
                extrinsic.set_camera_params(left_cam_logical_id, 
                                            charuco_calib_obj.camera_params['left']['kk'], 
                                            charuco_calib_obj.camera_params['left']['kc'])
                extrinsic.set_camera_params(right_cam_logical_id, 
                                            charuco_calib_obj.camera_params['right']['kk'], 
                                            charuco_calib_obj.camera_params['right']['kc'])
            else:
                print(f"  Falha na calibração intrínseca para '{system_logical_name}'. Usando parâmetros definidos manualmente.")
                for idx, cam_id_logical in enumerate(system_config['ids']):
                    extrinsic.set_camera_params(cam_id_logical, 
                                                system_config['params'][idx]['kk'], 
                                                system_config['params'][idx]['kc'])
        else: # Not Calibration: use hardcoded params from cam_dict
            for idx, cam_id_logical in enumerate(system_config['ids']):
                extrinsic.set_camera_params(cam_id_logical, 
                                            system_config['params'][idx]['kk'], 
                                            system_config['params'][idx]['kc'])

    # Preparar para carregar imagens
    physical_ids_to_load_imgs = []
    for system_config in cam_dict.values():
        disk_folder = system_config['disk_folder'] # ex: 'Passive'
        physical_ids_to_load_imgs.append(f"{disk_folder}/left")  # ex: 'Passive/left'
        physical_ids_to_load_imgs.append(f"{disk_folder}/right") # ex: 'Passive/right'
    
    physical_ids_to_load_imgs = sorted(list(set(physical_ids_to_load_imgs)))
    # images_loaded_from_disk terá chaves como 'Passive/left', 'Active/right', etc.
    images_loaded_from_disk = load_images(base_path, camera_disk_ids=physical_ids_to_load_imgs)

    # Mapear imagens carregadas para IDs lógicos e adicionar ao cam_dict
    for system_logical_name, system_config in cam_dict.items():
        disk_folder = system_config['disk_folder'] # ex: 'Passive'
        system_config['images_per_cam'] = {} 

        logical_left_id = system_config['ids'][0] # ex: 'SM2 Passivo/left'
        logical_right_id = system_config['ids'][1]# ex: 'SM2 Passivo/right'
        
        # Chaves para buscar em images_loaded_from_disk
        physical_left_id_on_disk = f"{disk_folder}/left"   # ex: 'Passive/left'
        physical_right_id_on_disk = f"{disk_folder}/right" # ex: 'Passive/right'
        
        system_config['images_per_cam'][logical_left_id] = images_loaded_from_disk.get(physical_left_id_on_disk, [])
        system_config['images_per_cam'][logical_right_id] = images_loaded_from_disk.get(physical_right_id_on_disk, [])

        if not system_config['images_per_cam'][logical_left_id]:
            print(f"Alerta: Nenhuma imagem carregada para {logical_left_id} (origem: {base_path}/{physical_left_id_on_disk})")
        if not system_config['images_per_cam'][logical_right_id]:
            print(f"Alerta: Nenhuma imagem carregada para {logical_right_id} (origem: {base_path}/{physical_right_id_on_disk})")


    # Calcular transformações relativas intra-sistema (left vs right)
    print("\nCalculando transformações relativas intra-sistema:")
    for system_logical_name, system_config in cam_dict.items():
        cam1_logical_id = system_config['ids'][0]
        cam2_logical_id = system_config['ids'][1]
        
        cam1_images = system_config['images_per_cam'].get(cam1_logical_id, [])
        cam2_images = system_config['images_per_cam'].get(cam2_logical_id, [])

        if not cam1_images or not cam2_images:
            print(f"Aviso: Imagens ausentes para o sistema '{system_logical_name}'. Pulando par ({cam1_logical_id}, {cam2_logical_id}).")
            continue
        
        cam1_data = {'id': cam1_logical_id, 'images': cam1_images}
        cam2_data = {'id': cam2_logical_id, 'images': cam2_images}
        process_camera_pair(cam1_data, cam2_data, extrinsic, mad_multiplier=mad_multiplier)

    # Calcular transformações relativas inter-sistemas (left de sistema A vs left de sistema B)
    print("\nCalculando transformações relativas inter-sistemas (entre câmeras 'left'):")
    all_system_logical_names = list(cam_dict.keys())
    for sys_name1, sys_name2 in combinations(all_system_logical_names, 2):
        cam1_logical_id = cam_dict[sys_name1]['ids'][0] 
        cam2_logical_id = cam_dict[sys_name2]['ids'][0] 

        cam1_images = cam_dict[sys_name1]['images_per_cam'].get(cam1_logical_id, [])
        cam2_images = cam_dict[sys_name2]['images_per_cam'].get(cam2_logical_id, [])

        if not cam1_images or not cam2_images:
            print(f"Aviso: Imagens ausentes. Pulando par inter-sistema ({cam1_logical_id}, {cam2_logical_id}).")
            continue
            
        cam1_data = {'id': cam1_logical_id, 'images': cam1_images}
        cam2_data = {'id': cam2_logical_id, 'images': cam2_images}
        process_camera_pair(cam1_data, cam2_data, extrinsic, mad_multiplier=mad_multiplier)

def process_camera_pair(cam1, cam2, extrinsic, mad_multiplier=1):
    """
    Processa um par de câmeras, calcula as transformações relativas e filtra imagens válidas.
    
    :param cam1: Dicionário com informações da primeira câmera ('id', 'images').
    :param cam2: Dicionário com informações da segunda câmera ('id', 'images').
    :param extrinsic: Objeto ExtrinsicCalib para realizar os cálculos.
    :param mad_multiplier: Fator multiplicativo para o Desvio Absoluto Mediano (MAD).
    :return: Lista de índices das imagens válidas.
    """
    transformations = []
    print(f"Transformação relativa entre '{cam1['id']}' e '{cam2['id']}':")
    
    num_images_cam1 = len(cam1['images'])
    num_images_cam2 = len(cam2['images'])

    if num_images_cam1 == 0 or num_images_cam2 == 0:
        print(f"  Imagens não disponíveis para uma ou ambas as câmeras. Pulando.")
        return []

    k_num = min(num_images_cam1, num_images_cam2)
    if k_num == 0 :
        print(f"  Nenhum par de imagens para processar entre '{cam1['id']}' e '{cam2['id']}'.")
        return []
    if num_images_cam1 != num_images_cam2:
        print(f"  Aviso: Número de imagens difere ({num_images_cam1} vs {num_images_cam2}). Usando {k_num} pares.")

    valid_image_pairs_count = 0
    for k in range(k_num):
        img1 = cam1['images'][k]
        img2 = cam2['images'][k]

        if img1 is None or img2 is None:
            print(f"  Imagem {k}: Uma ou ambas as imagens são None. Pulando par.")
            continue

        corners1, ids1 = extrinsic.detect_charuco_board(img1)
        corners2, ids2 = extrinsic.detect_charuco_board(img2)

        if corners1 is not None and ids1 is not None and len(ids1) > 0 and \
           corners2 is not None and ids2 is not None and len(ids2) > 0:
            
            obj1, img1_points = extrinsic.match_charuco_points(corners1, ids1)
            obj2, img2_points = extrinsic.match_charuco_points(corners2, ids2)

            T1, T2 = None, None # Inicializa T1 e T2
            if obj1 is not None and len(obj1) > 0 and img1_points is not None and len(img1_points) > 0:
                T1 = extrinsic.pose_estimation_charuco(camera=cam1['id'], obj_points=obj1, img_points=img1_points)
            
            if obj2 is not None and len(obj2) > 0 and img2_points is not None and len(img2_points) > 0:
                T2 = extrinsic.pose_estimation_charuco(camera=cam2['id'], obj_points=obj2, img_points=img2_points)


            if T1 is not None and T2 is not None:
                T_relative = extrinsic.calculate_relative_pose(T1, T2)
                if T_relative is not None:
                    transformations.append(T_relative[:3, 3]) 
                    valid_image_pairs_count +=1
                else:
                    print(f"  Imagem {k}: Transformação relativa não pôde ser calculada.")
            else:
                missing_pose_info = []
                if T1 is None: missing_pose_info.append(f"T1 para {cam1['id']}")
                if T2 is None: missing_pose_info.append(f"T2 para {cam2['id']}")
                print(f"  Imagem {k}: Estimativa de pose falhou para ({', '.join(missing_pose_info)}).")
        else:
            missing_marker_info = []
            if corners1 is None or ids1 is None or len(ids1) == 0: missing_marker_info.append(f"câmera {cam1['id']}")
            if corners2 is None or ids2 is None or len(ids2) == 0: missing_marker_info.append(f"câmera {cam2['id']}")
            print(f"  Imagem {k}: Marcadores Charuco não detectados ou insuficientes em ({', '.join(missing_marker_info)}).")
    
    if not transformations:
        print(f"  Nenhuma transformação relativa válida calculada para o par ('{cam1['id']}', '{cam2['id']}').")
        return []

    median_transformation = np.median(transformations, axis=0)
    mean_transformation = np.mean(transformations, axis=0)
    std_deviation = np.std(transformations, axis=0)
    print(f"  Estatísticas baseadas em {valid_image_pairs_count} pares de imagens válidas:")
    print(f"  Mediana das translações (mm): {median_transformation}")
    print(f"  Média das translações (mm): {mean_transformation}")
    print(f"  Desvio padrão das translações (mm): {std_deviation}")

    # # Plotar a transformação mediana
    # x, y, z = median_transformation
    # fig = plt.figure(figsize=(8,6)) # Ajuste o tamanho da figura se necessário
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, color='r', s=100, label=f'Mediana T: ({x:.2f}, {y:.2f}, {z:.2f}) mm')
    
    # trans_array = np.array(transformations)
    # ax.scatter(trans_array[:,0], trans_array[:,1], trans_array[:,2], color='b', alpha=0.3, label=f'Todas as Translações ({valid_image_pairs_count})')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.legend()
    # ax.set_title(f"Transformações Relativas: '{cam1['id']}' para '{cam2['id']}'", fontsize=10)
    # ax.plot([0],[0],[0], 'ko', markersize=5, label="Origem (Câm1 Ref.)") 
    # ax.quiver(0, 0, 0, x, y, z, length=np.linalg.norm([x,y,z]), normalize=True, color='r', arrow_length_ratio=0.1, label="Vetor Mediana")
    # ax.legend() # Chamar legend novamente para incluir o quiver
    # plt.tight_layout() # Ajusta o padding para evitar sobreposição de legendas/títulos
    # plt.show()

    valid_indices = extrinsic.filter_images_by_precision(transformations, mad_multiplier=mad_multiplier)
    print(f"  Índices de imagens filtradas (precisão < {mad_multiplier}): {valid_indices}")
    print()
    return valid_indices

if __name__ == '__main__':
    # ATUALIZE ESTE CAMINHO para o diretório que contém as pastas 'Passive' e 'Active'
    base_path = '/home/tiago/Downloads/20250430' 

    # Para realizar a calibração intrínseca e estéreo e salvar os arquivos YAML:
    # print("Iniciando calibração Charuco (intrínseca e estéreo)...")
    # charuco_calibration(base_path, save_path='camera-calibration/output_params_new_structure')

    # Para usar os parâmetros (hardcoded ou calibrados na hora) e calcular transformações extrínsecas:
    print("\nIniciando cálculo de marcadores Charuco e transformações extrínsecas...")
    charuco_markers(base_path, Calibration=True, mad_multiplier=1.5) 
