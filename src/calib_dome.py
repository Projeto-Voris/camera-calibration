
import os
import cv2
import yaml
import numpy as np

def cortar_regiao_interesse(imagem):
    """
    Abre uma janela para o usuário selecionar uma ROI usando o mouse.
    Retorna a matriz da imagem cortada.
    """
    # Define um nome para a janela de seleção
    nome_janela = "Selecione a ROI - [Enter/Espaco] Confirma | [C] Cancela"
    
    # Abre a interface nativa de seleção do OpenCV
    # showCrosshair=True mostra uma mira no centro da seleção
    # fromCenter=False permite arrastar do canto superior esquerdo ao inferior direito
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1200, 760)  # Ajuste o tamanho da janela conforme necessário
    roi = cv2.selectROI(nome_janela, imagem, showCrosshair=True, fromCenter=False)
    
    # Fecha a janela de seleção logo após o usuário confirmar ou cancelar
    cv2.destroyWindow(nome_janela)
    
    # A variável 'roi' retorna uma tupla com: (x_inicial, y_inicial, largura, altura)
    x, y, w, h = roi
    
    # Verifica se o usuário realmente selecionou uma área válida
    if w > 0 and h > 0:
        # Recorta a matriz utilizando fatiamento (slicing) do NumPy [y:y+h, x:x+w]
        imagem_cortada = imagem[int(y):int(y+h), int(x):int(x+w)]
        print(f"ROI selecionada com sucesso! Dimensões: {w}x{h} pixels.")
        cv2.imshow("ROI Selecionada", imagem_cortada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return imagem_cortada
    else:
        print("Nenhuma seleção feita ou operação cancelada. Retornando a imagem original.")
        return imagem

def load_ros2_calibration(yaml_path):
    """Carrega os parâmetros intrínsecos e de retificação do YAML do ROS2."""
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    
    # Extrai matrizes (ajuste as chaves conforme o seu arquivo do ROS2)
    K = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(calib_data['distortion_coefficients']['data'])
    R = np.array(calib_data['rectification_matrix']['data']).reshape(3, 3)
    P = np.array(calib_data['projection_matrix']['data']).reshape(3, 4)
    
    return K, D, R, P

def get_point_to_line_distance(point, line_params):
    """
    Calcula a distância ortogonal de um ponto a uma reta gerada pelo cv2.fitLine.
    line_params: (vx, vy, x0, y0) onde (vx, vy) é o vetor diretor e (x0, y0) um ponto da reta.
    """
    vx, vy, x0, y0 = line_params
    x, y = point
    # Fórmula da distância de um ponto a uma reta bidimensional
    return abs(vx * (y - y0) - vy * (x - x0)) / np.sqrt(vx**2 + vy**2)

def find_corners_shitomasi(img_color, max_corners=100, quality_level=0.15, min_distance=15):
    """
    Encontra cantos na imagem usando o método Shi-Tomasi (goodFeaturesToTrack).
    
    Parâmetros:
    - img_color: Imagem original colorida (BGR).
    - max_corners: Número máximo de cantos a detectar.
    - quality_level: Qualidade mínima aceitável do canto (0 a 1).
    - min_distance: Distância mínima em pixels entre os cantos.
    
    Retorna:
    - corners_refined: Array NumPy com as coordenadas (x, y) dos cantos refinados.
    - vis_img: Cópia da imagem original com os cantos desenhados.
    """
    # Trabalhar em uma cópia para não alterar a imagem original
    vis_img = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR) if len(img_color.shape) == 3 else img_color.copy()
    gray = img_color.copy()

    # 1. Busca inicial de cantos
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, 
                                      qualityLevel=quality_level, 
                                      minDistance=min_distance)

    if corners is None:
        print("Aviso: Nenhum canto detectado pelo Shi-Tomasi.")
        return None, vis_img

    # Converter para float32 para o refinamento subpixel
    corners = np.float32(corners)
    
    # 2. Refinamento Subpixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    
    # 3. Desenhar os pontos encontrados
    for corner in corners_refined:
        x, y = corner.ravel()
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 255, 0), -1)

    return corners_refined, vis_img

def find_corners_harris(img_color, block_size=2, ksize=3, k=0.04, threshold_rel=0.01):
    """
    Encontra cantos na imagem usando o método de Harris.
    
    Parâmetros:
    - img_color: Imagem original colorida (BGR).
    - block_size: Tamanho da vizinhança para detecção.
    - ksize: Abertura do operador Sobel (deve ser ímpar).
    - k: Parâmetro livre do detector de Harris.
    - threshold_rel: Porcentagem do valor máximo para considerar como canto (ex: 0.01 = 1%).
    
    Retorna:
    - corners_refined: Array NumPy com as coordenadas (x, y) dos cantos refinados.
    - vis_img: Cópia da imagem original com os cantos desenhados.
    """
    vis_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if len(img_color.shape) == 3 else img_color.copy()
    gray = img_color.copy()

    # O Harris exige que a imagem de entrada seja float32
    gray_float = np.float32(gray)

    # 1. Calcular o mapa de resposta do Harris
    dst = cv2.cornerHarris(gray_float, blockSize=block_size, ksize=ksize, k=k)

    # 2. Aplicar limiar para extrair as regiões dos cantos
    threshold = threshold_rel * dst.max()
    corner_mask = np.uint8(dst > threshold)

    # 3. Extrair as coordenadas dos centroides das regiões detectadas
    # connectedComponentsWithStats encontra as "manchas" que passaram no threshold
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_mask)

    # Se apenas o fundo (índice 0) foi encontrado, não há cantos
    if len(centroids) <= 1:
        print("Aviso: Nenhum canto detectado pelo Harris com o threshold atual.")
        return None, vis_img

    # Ignorar o índice 0, que representa o fundo da imagem
    corners = np.float32(centroids[1:])

    # 4. Refinamento Subpixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # 5. Desenhar os pontos encontrados
    for corner in corners_refined:
        x, y = corner.ravel()
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)

    return corners_refined, vis_img

def chessboard_evaluate(img, yaml_path, chessboard_size=(9, 6)):
    """
    Avalia o alinhamento domo-pupila medindo a retitude das linhas verticais.
    chessboard_size: (colunas internas, linhas internas)
    """
    # 1. Carregar calibração e Imagem
    K, D, R, P = load_ros2_calibration(yaml_path)    
    # 2. Retificar a imagem (Remover distorção intrínseca da lente)
    # Nota: Queremos avaliar o erro do DOMO. Portanto, remover a distorção da lente isolada é ideal.
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P[:3, :3], (w, h), cv2.CV_32FC1)
    rectified_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    rectified_gray = rectified_img.copy()

    # 3. Encontrar vértices do tabuleiro
    ret, corners = cv2.findChessboardCorners(rectified_gray, chessboard_size, None)
    
    if not ret:
        print("Erro: Não foi possível encontrar o tabuleiro de xadrez na imagem retificada.")
        return None

    # Refinamento subpixel dos cantos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(rectified_gray, corners, (11, 11), (-1, -1), criteria)
    # Print the detected corners for debugging
    print(f"Detected corners (subpixel refined): {corners.reshape(-1, 2)}")
    cv2.drawChessboardCorners(rectified_img, chessboard_size, corners, ret)
    cv2.imshow("Detected Chessboard Corners", rectified_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Redimensionar para a matriz do tabuleiro (linhas, colunas, 2)
    # O OpenCV organiza por linhas (da esquerda para direita, de cima para baixo)
    corners_grid = corners.reshape(chessboard_size[1], chessboard_size[0], 2)
    
    total_errors = []
    vis_img = rectified_img.copy()

    print(f"\n--- Análise de Retas Verticais (Total de Colunas: {chessboard_size[0]}) ---")
    
    # 4. Iterar sobre as COLUNAS verticais do tabuleiro
    for col in range(chessboard_size[0]):
        # Extrai todos os pontos daquela coluna específica
        pts_col = corners_grid[:, col, :] # Formato (N_linhas, 2)
        
        # Ajusta uma reta usando Mínimos Quadrados (DIST_L2)
        # Retorna [vx, vy, x0, y0]
        line_params = cv2.fitLine(pts_col, cv2.DIST_L2, 0, 0.01, 0.01)
        
        col_errors = []
        for pt in pts_col:
            # Calcular erro em pixel do vértice até a reta ajustada
            err = get_point_to_line_distance(pt, line_params)
            col_errors.append(err)
            total_errors.append(err)
            
            # Desenhar o vértice avaliado
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
            
        # Desenhar a reta ajustada para visualização estendida
        vx, vy, x0, y0 = line_params
        pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
        pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
        cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)
        
        print(f"Coluna {col}: Erro Médio = {np.mean(col_errors):.3f} px | Erro Máx = {np.max(col_errors):.3f} px")

    # Metrics Globais
    rmse = np.sqrt(np.mean(np.square(total_errors)))
    max_err = np.max(total_errors)
    
    print("\n--- Resultado Global ---")
    print(f"Erro Médio Quadrático (RMSE): {rmse:.3f} pixels")
    print(f"Erro Máximo Encontrado: {max_err:.3f} pixels")
    
    # Exibir resultado visual
    cv2.imshow("Avaliacao de Alinhamento (Linhas Ajustadas)", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return rmse, max_err

def find_free_vertices_in_roi(img_color, max_corners=150, quality_level=0.15, min_distance=15):
    """
    Permite selecionar uma ROI e busca vértices livremente (sem tamanho de grid)
    usando o Shi-Tomasi (goodFeaturesToTrack) restrito à área selecionada.
    """
    # 1. Selecionar a Região de Interesse (ROI)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR) if len(img_color.shape) != 3 else img_color.copy()

    cv2.namedWindow("Selecione o Tabuleiro", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecione o Tabuleiro", 1200, 760)
    roi = cv2.selectROI("Selecione o Tabuleiro", img_color, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Selecione o Tabuleiro")
    
    x, y, w, h = roi
    
    # Se o usuário não selecionou nada (apertou C ou fechou), encerra a função
    if w == 0 or h == 0:
        print("Nenhuma região selecionada. Abortando.")
        return None, img_color.copy()

    # 2. Criar uma máscara preta e pintar de branco apenas a área selecionada
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255 # 255 significa que o algoritmo só pode olhar para cá

    # 3. Buscar vértices LIVREMENTE, usando a máscara para ignorar o fundo
    corners = cv2.goodFeaturesToTrack(gray, 
                                      maxCorners=max_corners, 
                                      qualityLevel=quality_level, 
                                      minDistance=min_distance,
                                      mask=mask) # A máscara é o segredo aqui

    if corners is None:
        print("Nenhum vértice foi encontrado dentro da área selecionada.")
        return None, img_color.copy()

    # 4. Refinamento Subpixel para precisão geométrica
    corners = np.float32(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # 5. Desenhar os pontos encontrados
    vis_img = img_color.copy()
    
    # Desenhar o retângulo da ROI para referência visual
    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    for corner in corners_refined:
        cx, cy = corner.ravel()
        cv2.circle(vis_img, (int(cx), int(cy)), 4, (0, 255, 0), -1)

    print(f"Sucesso! {len(corners_refined)} vértices encontrados de forma livre.")
    
    return corners_refined, vis_img
# Exemplo de uso:
path = '/home/daniel/Pictures/20260715_1457_stereo_images'
left_imgs = sorted(os.listdir(os.path.join(path, 'left')))[:-2]
right_imgs = sorted(os.listdir(os.path.join(path, 'right')))[:-2]
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))  # Inicializa CLAHE para uso posterior
for left, right in zip(left_imgs, right_imgs):
    img_l = cv2.imread(os.path.join(path, 'left', left), cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(os.path.join(path, 'right', right), cv2.IMREAD_GRAYSCALE)
    img_l = clahe.apply(img_l)
    img_r = clahe.apply(img_r)
    # croped_l = cortar_regiao_interesse(img_l)
    # croped_r = cortar_regiao_interesse(img_r)

    print(f"\n--- Avaliando Imagem: {left} e {right} ---")
    # print(f"Haaris")
    # _, vis_l = find_corners_harris(croped_l)
    # _, vis_r = find_corners_harris(croped_r)
    # cv2.imshow("Harris - Left", vis_l)
    # cv2.imshow("Harris - Right", vis_r)
    # cv2.waitKey(0)
    # print(f"Shi-Tomasi")
    # _, vis_l = find_corners_shitomasi(croped_l)
    # _, vis_r = find_corners_shitomasi(croped_r)
    # cv2.imshow("Shi-Tomasi - Left", vis_l)
    # cv2.imshow("Shi-Tomasi - Right", vis_r)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    _, vis = find_free_vertices_in_roi(img_l)
    cv2.imshow("Free Vertices - Left", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # chessboard_evaluate(croped_l, os.path.join(path, 'left.yaml'), chessboard_size=(9, 6))
    # chessboard_evaluate(croped_r, os.path.join(path, 'right.yaml'), chessboard_size=(9, 6))

# evaluate_alignment('camera_meia_agua.png', 'ost_camera_info.yaml', chessboard_size=(9, 6))