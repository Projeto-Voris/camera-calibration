import os.path
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

def charuco_calibration():
    """
    Realiza a calibração das câmeras usando um tabuleiro Charuco.
    """
    charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))
    charuco.create_charucoboard(min_margin_mm=5, paper_size=(297, 420), dpi=600, save_as_pdf=False)

    images_path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S1'
    charuco.load_image(images_path=images_path, visualize=False)
    charuco.detect_charuco_board(camera='left', visualize=True)
    charuco.detect_charuco_board(camera='right', visualize=True)
    charuco.calibration_camera(camera='left')
    charuco.calibration_camera(camera='right')
    charuco.stereo_calibration()
    charuco.save_camera_params(path='./camera_parameters/sm3_charuco.yaml')
    print('debug')
    charuco.homogeneous_matrix(camera='left', visualize=False)
    print('debugger')

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
    
    for k in range(len(cam1['images'])):
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
                print(f"Imagem {k}: {T_relative[:3, 3]}")
            else:
                print(f"Imagem {k}: Transformação não disponível.")
        else:
            print(f"Imagem {k}: Marcadores não detectados.")

    # Filtrar imagens com base na precisão
    valid_indices = extrinsic.filter_images_by_precision(transformations, threshold=threshold)
    print(f"Imagens válidas para o par ({cam1['id']}, {cam2['id']}) com base no limite de precisão ({threshold}): {valid_indices}")
    return valid_indices

def charuco_markers():
    """
    Define os parâmetros das câmeras diretamente no código e calcula as transformações relativas.
    """
    # Inicializa o objeto ExtrinsicCalib
    extrinsic = ExtrinsicCalib(marker_size=(28, 18))

    # Criar o dicionário de câmeras com os parâmetros definidos diretamente no código 
    cam_dict = { 
        'cam1': {
            'id': 'left',
            'kk': [[ 4436.4, 0.0, 1016.4], [ 0.0, 4434.3, 731.19 ], [ 0.0, 0.0, 1.0 ]],
            'kc': [  -0.22937, -0.46029, 0.0010781, -0.00059945, 0.67818 ]
        },
        'cam2': {
            'id': 'right',
            'kk': [[ 4418.1, 0.0, 1040.4 ], [ 0.0, 4416.5, 759.65 ], [0, 0, 1]],
            'kc': [-0.23152, -0.40006, 0.00079416, -0.00045294, 0.15779 ]
        },
        'cam3': {
            'id': 'top',
            'kk': [[ 4418.1, 0.0, 1040.4 ], [ 0.0, 4416.5, 759.65 ], [0, 0, 1]],
            'kc': [-0.23152, -0.40006, 0.00079416, -0.00045294, 0.15779 ]
        }
    }

    # Configurar os parâmetros das câmeras no objeto ExtrinsicCalib
    for cam_key, params in cam_dict.items():
        extrinsic.set_camera_params(params['id'], params['kk'], params['kc'])

    # Lista de câmeras (baseada nos IDs do dicionário)
    cameras = [cam_dict['cam1']['id'], cam_dict['cam2']['id'], cam_dict['cam3']['id']]

    # Caminho para as imagens
    path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S1'
    images = load_images(path, cameras=cameras)

    # Adicionar as imagens ao dicionário de câmeras
    for cam_key, params in cam_dict.items():
        cam_dict[cam_key]['images'] = images[params['id']]

    # Calcular as transformações relativas entre todas as combinações de câmeras
    threshold = 1
    for cam1_key, cam2_key in combinations(cam_dict.keys(), 2):
        cam1 = cam_dict[cam1_key]
        cam2 = cam_dict[cam2_key]
        process_camera_pair(cam1, cam2, extrinsic, threshold=threshold)

if __name__ == '__main__':
    charuco_markers()
