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

    images_path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/SM2'
    charuco.load_image(images_path=images_path, visualize=False)
    charuco.detect_charuco_board(camera='left', visualize=False)
    charuco.detect_charuco_board(camera='right', visualize=False)
    charuco.calibration_camera(camera='left')
    charuco.calibration_camera(camera='right')
    charuco.stereo_calibration()
    charuco.save_camera_params(path='camera-calibration/camera_parameters/sm2_charuco.yaml')

    charuco.homogeneous_matrix(camera='left', visualize=False)

    return 

def charuco_markers(Calibration=False):
    """
    Define os parâmetros das câmeras diretamente no código e calcula as transformações relativas.
    """
    # Inicializa o objeto ExtrinsicCalib
    extrinsic = ExtrinsicCalib(marker_size=(28, 18))

    cam_dict = { 
        'cam1': {
            'ids': ['SM2/left', 'SM2/right'],
            'params': [
                {
                    'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03],[0.0000e+00, 4.3180e+03, 1.0937e+03],[0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]
                },
                {
                    'kk': [ [4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]
                }
            ]
        },
        'cam2': {
            'ids': ['SM3/left', 'SM3/right'],
            'params': [
                {
                    'kk': [[4.3200e+03, 0.0000e+00, 1.0562e+03],[0.0000e+00, 4.3180e+03, 1.0937e+03],[0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-1.6623e-01, -1.6184e+00, -1.2382e-03, -1.5350e-03, 8.5337e+00]
                },
                {
                    'kk': [ [4.3930e+03, 0.0000e+00, 9.7039e+02], [0.0000e+00, 4.4234e+03, 1.1391e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-2.2510e-01, -1.0044e+00, -2.3892e-03, 6.9123e-03, 5.3801e+00]
                }
            ]
        },
        'cam3': {
            'ids': ['SM4/left', 'SM4/right'],
            'params': [
                {
                    'kk': [[1.3172e+03, 0.0000e+00, 7.8616e+02], [0.0000e+00, 1.3193e+03, 6.2468e+02],[0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-2.0376e-01, 1.3796e-01, 8.2274e-04, 8.9460e-04, -2.7476e-02]
                },
                {
                    'kk': [[1.4093e+03, 0.0000e+00, 7.9993e+02], [0.0000e+00, 1.4089e+03, 5.9945e+02], [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                    'kc': [-3.2045e-01, 2.8344e+00, 6.7285e-03, 3.9213e-04, -1.9141e+01]
                }
            ]
        }
    }

    # Configurar os parâmetros das câmeras no objeto ExtrinsicCalib
    for cam_key, params in cam_dict.items():
        for idx, cam_id in enumerate(params['ids']):
            extrinsic.set_camera_params(cam_id, params['params'][idx]['kk'], params['params'][idx]['kc'])

    # Lista de câmeras (baseada nos IDs do dicionário)
    cameras = [cam_id for cam in cam_dict.values() for cam_id in cam['ids']]

    # Caminho para as imagens
    path = '/home/tiago/Downloads/20250430-20250506T115956Z-001/20250430'
    images = load_images(path, cameras=cameras)

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
    charuco_markers()
    # charuco_calibration()