import os
import os.path
import yaml
import numpy as np
import cv2
from itertools import combinations
from src.charuco_calibration import CharucoCalibration
from src.extrinsic_calibration import ExtrinsicCalib

def load_images(path, cameras=('left', 'right')):
    """
    Carrega as imagens de cada câmera a partir do diretório especificado.
    """
    images = {}
    for camera in cameras:
        camera_path = os.path.join(path, camera)
        images[camera] = [cv2.imread(os.path.join(camera_path, img)) for img in sorted(os.listdir(camera_path))]
    return images

def charuco_calibration(base_path: str, save_path: str = 'camera-calibration/camera_parameters') -> None:
    """
    Realiza a calibração de múltiplos sistemas e câmeras usando um tabuleiro Charuco.
    Inclui calibração intrínseca e estéreo.
    
    :param base_path: Caminho principal onde estão as pastas dos sistemas (ex.: SM2, SM3, SM4).
    :param save_path: Caminho para salvar os parâmetros de calibração estéreo.
    """
    # Instanciar a classe CharucoCalibration
    charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))

    # Iterar sobre os sistemas (SM2, SM3, SM4)
    for system in ['SM2', 'SM3', 'SM4']:
        system_path = os.path.join(base_path, system)
        if not os.path.exists(system_path):
            print(f"Diretório do sistema {system} não encontrado em {base_path}. Pulando...")
            continue

        print(f"Processando sistema {system}...")

        # Carregar imagens de ambas as câmeras (left e right)
        charuco.load_image(images_path=system_path, visualize=False)

        # Detectar o tabuleiro ChArUco para ambas as câmeras
        charuco.detect_charuco_board(camera='left', visualize=False)
        charuco.detect_charuco_board(camera='right', visualize=False)

        # Calibrar ambas as câmeras
        charuco.calibration_camera(camera='left')
        charuco.calibration_camera(camera='right')

        # Realizar calibração estéreo
        print(f"Realizando calibração estéreo para o sistema {system}...")
        charuco.stereo_calibration()

        # Salvar os parâmetros estéreo no formato YAML
        save_file = os.path.join(save_path, f"{system}_stereo_charuco.yaml")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        charuco.save_camera_params(save_file)
        print(f"Parâmetros estéreo do sistema {system} salvos em {save_file}")

    print("Calibração concluída para todos os sistemas.")

# Configurar os parâmetros das câmeras no objeto ExtrinsicCalib
def charuco_markers(base_path: str, Calibration=False):
    """
    Define os parâmetros das câmeras diretamente no código e calcula as transformações relativas.
    Automatiza a calibração intrínseca se necessário.
    
    :param base_path: Caminho principal onde estão as pastas dos sistemas (ex.: SM2, SM3, SM4).
    :param Calibration: Se True, realiza a calibração intrínseca automaticamente.
    """
    # Inicializa o objeto ExtrinsicCalib
    extrinsic = ExtrinsicCalib(marker_size=(28, 18))
    

    # Dicionário de câmeras e parâmetros
    cam_dict = {
        'SM2': {
            'ids': ['SM2/left', 'SM2/right'],
            'params': [
                {
                    'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03], [0.0000e+00, 4.3180e+03, 1.0937e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]
                },
                {
                    'kk': [[4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]
                }
            ]
        },
        'SM3': {
            'ids': ['SM3/left', 'SM3/right'],
            'params': [
                {
                    'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03], [0.0000e+00, 4.3180e+03, 1.0937e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]
                },
                {
                    'kk': [[4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]
                }
            ]
        },
        'SM4': {
            'ids': ['SM4/left', 'SM4/right'],
            'params': [
                {
                    'kk': [[ 1355.0238, 0.0, 761.5210 ], [ 0.0, 1354.4458, 594.4952 ], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [ -0.2140, 0.1692, 0.0001, -0.0006, 0.0 ]
                },
                {
                    'kk': [[ 1366.5948, 0.0, 771.2434 ], [ 0.0, 1366.3036, 602.9112 ], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [ -0.2127, 0.1492, -0.0011, -0.0001, 0.0 ]
                }
            ]
        }
    }

    for cam_key, params in cam_dict.items():
        system = cam_key.split('/')[0]  # Obter o sistema (ex.: SM2, SM3, SM4)

        if Calibration:
            charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))

            # Automatizar a calibração intrínseca para o sistema
            print(f"Calibrando sistema {system}...")
            system_path = os.path.join(base_path, system)
            print(f"Path {system_path}")
            charuco.load_image(images_path=system_path, visualize=False)

            # Calibrar as câmeras 'left' e 'right'
            for camera in ['left', 'right']:
                charuco.detect_charuco_board(camera=camera, visualize=False)
                charuco.calibration_camera(camera=camera)

                # Obter os parâmetros intrínsecos calculados
                cam_id = f"{system}/{camera}"
                charuco.camera_params[cam_id] = charuco.camera_params.pop(camera)
                extrinsic.set_camera_params(cam_id, 
                                            charuco.camera_params[cam_id]['kk'], 
                                            charuco.camera_params[cam_id]['kc'])
        else:
            # Usar os parâmetros fixos do dicionário
            for idx, cam_id in enumerate(params['ids']):
                extrinsic.set_camera_params(cam_id, params['params'][idx]['kk'], params['params'][idx]['kc'])

    # Lista de câmeras (baseada nos IDs do dicionário)
    cameras = [cam_id for cam in cam_dict.values() for cam_id in cam['ids']]

    # Caminho para as imagens
    images = load_images(base_path, cameras=cameras)

    # Adicionar as imagens ao dicionário de câmeras
    for cam_key, params in cam_dict.items():
        for idx, cam_id in enumerate(params['ids']):
            if 'images' not in cam_dict[cam_key]:
                cam_dict[cam_key]['images'] = {}
            cam_dict[cam_key]['images'][cam_id] = images[cam_id]

    # Calcular as transformações relativas apenas entre 'left' e 'right' de cada câmera
    threshold = 10
    for cam_key, params in cam_dict.items():
        cam1 = {'id': params['ids'][0], 'images': cam_dict[cam_key]['images'][params['ids'][0]]}
        cam2 = {'id': params['ids'][1], 'images': cam_dict[cam_key]['images'][params['ids'][1]]}
        process_camera_pair(cam1, cam2, extrinsic, threshold=threshold)

    # Calcular as transformações relativas entre 'left' de cada par de câmeras
    for cam1_key, cam2_key in combinations(cam_dict.keys(), 2):
        cam1 = {'id': cam_dict[cam1_key]['ids'][0], 'images': cam_dict[cam1_key]['images'][cam_dict[cam1_key]['ids'][0]]}
        cam2 = {'id': cam_dict[cam2_key]['ids'][0], 'images': cam_dict[cam2_key]['images'][cam_dict[cam2_key]['ids'][0]]}
        process_camera_pair(cam1, cam2, extrinsic, threshold=threshold)

def process_camera_pair(cam1, cam2, extrinsic, threshold=1):
    """
    Processa um par de câmeras, calcula as transformações relativas e filtra imagens válidas.
    
    :param cam1: Dicionário com informações da primeira câmera.
    :param cam2: Dicionário com informações da segunda câmera.
    :param extrinsic: Objeto ExtrinsicCalib para realizar os cálculos.
    :param threshold: Limite de precisão para filtrar imagens.
    :return: Lista de índices das imagens válidas.
    """
    transformations = []
    print(f"Transformação relativa entre {cam1['id']} e {cam2['id']}:")
    
    if len(cam1['images']) < len(cam2['images']):
        k_num = len(cam1['images'])
    else:
        k_num = len(cam2['images'])
    
    for k in range(k_num):
        img1 = cam1['images'][k]
        img2 = cam2['images'][k]

        # Detectar marcadores Charuco
        corners1, ids1 = extrinsic.detect_charuco_board(img1)
        corners2, ids2 = extrinsic.detect_charuco_board(img2)

        # Verificar se os cantos foram detectados
        if corners1 is not None and corners2 is not None:
            obj1, img1_points = extrinsic.match_charuco_points(corners1, ids1)
            obj2, img2_points = extrinsic.match_charuco_points(corners2, ids2)

            # Estimar poses
            T1 = extrinsic.pose_estimation_charuco(camera=cam1['id'], obj_points=obj1, img_points=img1_points)
            T2 = extrinsic.pose_estimation_charuco(camera=cam2['id'], obj_points=obj2, img_points=img2_points)

            # Calcular transformação relativa
            if T1 is not None and T2 is not None:
                T_relative = extrinsic.calculate_relative_pose(T1, T2)
                transformations.append(T_relative[:3, 3])  # Armazena apenas a translação
            else:
                print(f"Imagem {k}: Transformação não disponível.")
        else:
            print(f"Imagem {k}: Marcadores não detectados.")

    # Calcular estatísticas
    median_transformation = np.median(transformations, axis=0)
    mean_transformation = np.mean(transformations, axis=0)
    std_deviation = np.std(transformations, axis=0)
    print(f"Mediana das transformações: {median_transformation}")
    print(f"Média das transformações: {mean_transformation}")
    print(f"Desvio padrão das transformações: {std_deviation}")

    # Filtrar imagens com base na precisão
    valid_indices = extrinsic.filter_images_by_precision(transformations, threshold=threshold)
    print(f"Imagens válidas para o par ({cam1['id']}, {cam2['id']}) com base no limite de precisão ({threshold}): {valid_indices}")
    print()
    return valid_indices

if __name__ == '__main__':
    base_path = '/home/tiago/Downloads/20250430'
    # charuco_calibration(base_path)
    charuco_markers(base_path, Calibration=True)