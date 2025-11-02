# salve como solver_interativo_setas.py
import cv2
import numpy as np
import kociemba
import time
import sys
import os  # Importa o módulo 'os' para verificar se o arquivo existe
from scipy import stats  # Para ajudar na estabilidade da detecção

print("--- Iniciando Script ---")  # DEBUG 1

# --- 1. Carregar Valores Calibrados ---
# (Nenhuma alteração nesta seção)
try:
    if os.path.exists("calibrated_colors.py"):
        from calibrated_colors import calibrated_values
        color_ranges = calibrated_values
        print("DEBUG: Valores carregados com sucesso.")  # DEBUG 2
        # Verifica se todas as 6 cores foram carregadas
        if len(color_ranges) != 6:
            print("\n!!! ATENCAO !!!")
            print(f"O arquivo 'calibrated_colors.py' contem apenas {len(color_ranges)} cores.")
            print("Execute o 'calibrador_com_save.py' novamente e calibre TODAS as 6 cores.")
            print("O programa pode falhar se uma cor estiver faltando.")
            print("Cores carregadas:", list(color_ranges.keys()))
            print("!!! ATENCAO !!!\n")
        else:
            print("DEBUG: Arquivo 'calibrated_colors.py' contem 6 cores.")  # DEBUG 2.1

    else:
        # Define valores de ERRO se o arquivo não existir
        print("\n!!! ERRO FATAL !!!")
        print("Arquivo 'calibrated_colors.py' não encontrado na pasta atual.")
        print("Execute o 'calibrador_com_save.py' primeiro para calibrar as cores.")
        print("!!! ERRO FATAL !!!\n")
        exit()  # Termina a execução

except ImportError:
    print("\n!!! ERRO FATAL !!!")
    print("Erro ao importar 'calibrated_colors.py'. O arquivo pode estar corrompido ou mal formatado.")
    print("Delete o arquivo 'calibrated_colors.py' e execute o calibrador novamente.")
    print("!!! ERRO FATAL !!!\n")
    exit()
except Exception as e:
    print(f"\n!!! ERRO FATAL !!!")
    print(f"Erro inesperado ao carregar calibração: {e}")
    print("Verifique o arquivo 'calibrated_colors.py' ou execute o calibrador novamente.")
    print("!!! ERRO FATAL !!!\n")
    exit()
# --- FIM DO CARREGAMENTO ---


# --- 2. Configurações e Mapeamentos ---
print("DEBUG: Definindo Configurações...")  # DEBUG 3
# Posições dos centros dos 9 quadrados (x, y) - Ajuste se necessário
grid_centers = [
    (250, 170), (320, 170), (390, 170),  # Linha de cima
    (250, 240), (320, 240), (390, 240),  # Linha do meio
    (250, 310), (320, 310), (390, 310)   # Linha de baixo
]
grid_map = np.array(range(9)).reshape(3, 3)  # Mapa 0-8 para posições lógicas

# Ordem das faces para escanear e padrão Kociemba
faces_order = ['U', 'R', 'F', 'D', 'L', 'B']

# Nomes neutros (sem cores fixas)
face_names_pt = {
    'U': 'Cima', 'R': 'Direita', 'F': 'Frente',
    'D': 'Baixo', 'L': 'Esquerda', 'B': 'Trás'
}

# Mapeamento interno: Número da cor (baseado no centro) para letra Kociemba
# Será preenchido após escanear
num_to_kociemba_letter = {}
kociemba_letter_to_num = {}
print("DEBUG: Configurações definidas.")  # DEBUG 3 (Fim)


# --- 3. Funções Auxiliares ---
# (Nenhuma alteração nesta seção)
print("DEBUG: Definindo Funções Auxiliares...")  # DEBUG 4

def get_color_name(hsv_pixel):
    """ Retorna a letra da cor ('U', 'R', 'F'...) ou '?' """
    h, s, v = hsv_pixel

    # Itera sobre os ranges carregados do arquivo
    for color_name, (lower, upper) in color_ranges.items():
        h_min, s_min, v_min = lower
        h_max, s_max, v_max = upper

        # Lógica para checar se o pixel está no range (incluindo wrap-around do HUE)
        hue_match = False
        if h_min <= h_max:  # Range normal de HUE
            if h_min <= h <= h_max:
                hue_match = True
        else:  # Range de HUE que dá a volta (ex: vermelho Hmin=170, Hmax=10)
            if (h_min <= h <= 179) or (0 <= h <= h_max):
                hue_match = True

        # Se o HUE bateu, checa Saturação e Valor
        if hue_match:
            if s_min <= s <= s_max and v_min <= v <= v_max:
                return color_name  # Encontrou a cor!

    return '?'  # Nenhuma cor encontrada

def draw_preview_grid(frame, centers):
    """ Desenha a grade de 9 quadrados """
    if frame is None:
        return None  # Adiciona checagem
    for (x, y) in centers:
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (255, 255, 255), 1)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    return frame

def detect_face_from_webcam(frame, centers, color_map):
    """
    Detecta a face usando amostragem de pixels e retorna a matriz 1x9 numérica.
    Retorna None se a detecção falhar.
    """
    if frame is None:
        print("DEBUG: detect_face_from_webcam recebeu frame Nulo.")
        return None
    try:  # Adiciona try-except para a conversão de cor
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"DEBUG: Erro ao converter frame para HSV: {e}")
        return None  # Frame inválido

    detected_colors_letters = []
    detected_colors_numbers = []

    for idx, (x, y) in enumerate(centers):
        # Verifica limites antes de acessar pixel
        if y >= hsv_frame.shape[0] or x >= hsv_frame.shape[1] or y < 0 or x < 0:
            print(f"DEBUG: Coordenada {idx} ({x},{y}) fora dos limites ({hsv_frame.shape[0]}x{hsv_frame.shape[1]}).")
            return None  # Coordenada inválida

        pixel_hsv = hsv_frame[y, x]
        color_letter = get_color_name(pixel_hsv)
        detected_colors_letters.append(color_letter)

        if color_letter == '?':
            return None  # Falha na detecção

        # Converte letra para número usando o mapa criado no scan
        color_num = color_map.get(color_letter)  # Pega o número associado à letra
        if color_num is None:  # Verifica se a letra realmente existe no mapa
            print(f"DEBUG: ERRO CRÍTICO no Mapeamento durante RESOLUÇÃO - Cor '{color_letter}' (HSV={pixel_hsv}) detectada em ({x},{y}), mas não está no mapeamento 'kociemba_letter_to_num'.")
            print("DEBUG: Mapeamento:", kociemba_letter_to_num)
            return None  # Retorna None se o mapeamento falhar durante a resolução
        detected_colors_numbers.append(color_num)

    if len(detected_colors_numbers) == 9:
        return np.array([detected_colors_numbers])  # Retorna como array NumPy 1x9
    else:
        print(f"DEBUG: Detectou {len(detected_colors_numbers)} cores em vez de 9.")
        return None


# Funções de Rotação 2D
def rotate_cw(face_1x9):
    if face_1x9 is None or face_1x9.shape != (1, 9):
        print("DEBUG: rotate_cw recebeu face inválida.")
        return None
    face = face_1x9.reshape(3, 3)
    rotated = np.rot90(face, k=-1)
    return rotated.flatten().reshape(1, 9)

def rotate_ccw(face_1x9):
    if face_1x9 is None or face_1x9.shape != (1, 9):
        print("DEBUG: rotate_ccw recebeu face inválida.")
        return None
    face = face_1x9.reshape(3, 3)
    rotated = np.rot90(face, k=1)
    return rotated.flatten().reshape(1, 9)

print("DEBUG: Funções auxiliares definidas.")  # DEBUG 4 (Fim)


# --- 4. Funções Interativas com Setas ---
print("DEBUG: Definindo Funções Interativas...")  # DEBUG 5

def wait_for_move(video, expected_front_face, state_before_front, move_name, arrow_coords):
    print(f"DEBUG: Entrou em wait_for_move para {move_name}")

    if expected_front_face is None:
        print("DEBUG: ERRO - expected_front_face é None em wait_for_move.")
        return False  # Não podemos comparar com None

    print(f"Faça o movimento: {move_name}")
    detected_faces_buffer = []

    while True:
        is_ok, frame = video.read()
        if not is_ok:
            print("DEBUG: Erro ao ler webcam durante wait_for_move.")
            return False

        if frame is None:
            print("DEBUG: Frame nulo durante wait_for_move.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_with_grid = draw_preview_grid(frame.copy(), grid_centers)
        if frame_with_grid is None:
            continue  # Checa se draw_preview_grid falhou

        # Detecta a face atual - AGORA USA O MAPEAMENTO COMPLETO
        current_face_state_num = detect_face_from_webcam(frame, grid_centers, kociemba_letter_to_num)

        # Mostra instrução
        cv2.putText(frame_with_grid, f"Faca o movimento: {move_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if current_face_state_num is not None:
            # --- Desenha as letras detectadas DURANTE a espera ---
            try:
                current_letters = [num_to_kociemba_letter.get(n, '?') for n in current_face_state_num[0]]
                if len(current_letters) == 9:
                    for i, (x, y) in enumerate(grid_centers):
                        text_pos = (x - 10, y + 5)
                        cv2.putText(frame_with_grid, current_letters[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame_with_grid, current_letters[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e_draw_wait:
                print(f"Debug: Erro menor ao desenhar letras em wait_for_move: {e_draw_wait}")
            # --- Fim do desenho das letras ---

            detected_faces_buffer.append(current_face_state_num.tolist())
            if len(detected_faces_buffer) > 5:
                detected_faces_buffer.pop(0)

            # Verifica estabilidade
            if len(detected_faces_buffer) >= 3 and all(np.array_equal(np.array(f), expected_front_face) for f in detected_faces_buffer[-3:]):
                print("DEBUG: MOVIMENTO DETECTADO E ESTAVEL!")
                cv2.putText(frame_with_grid, "OK!", (frame.shape[1] // 2 - 30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Resolvendo...", frame_with_grid)
                cv2.waitKey(500)
                return True

            # Desenha seta se estiver no estado anterior
            elif len(detected_faces_buffer) >= 1 and state_before_front is not None and np.array_equal(np.array(detected_faces_buffer[-1]), state_before_front):
                for p1, p2 in arrow_coords:
                    try:  # Adiciona try-except para desenho da seta
                        p1_int = (int(p1[0]), int(p1[1]))
                        p2_int = (int(p2[0]), int(p2[1]))
                        cv2.arrowedLine(frame_with_grid, p1_int, p2_int, (0, 0, 0), 7, tipLength=0.2)
                        cv2.arrowedLine(frame_with_grid, p1_int, p2_int, (0, 0, 255), 4, tipLength=0.2)
                    except Exception as draw_err:
                        print(f"DEBUG: Erro ao desenhar seta: {draw_err}, P1={p1}, P2={p2}")

        else:
            detected_faces_buffer = []
            cv2.putText(frame_with_grid, "Ajuste o cubo na grade", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Garante que a janela existe antes de mostrar
        if cv2.getWindowProperty("Resolvendo...", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow("Resolvendo...", frame_with_grid)
        else:
            print("DEBUG: Janela 'Resolvendo...' não está visível.")
            break  # Sai do loop se a janela foi fechada

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            print("DEBUG: 'q' pressionado em wait_for_move.")
            return False
    # Se saiu do loop por outro motivo (ex: janela fechada)
    print("DEBUG: Saindo de wait_for_move sem sucesso.")
    return False


# Coordenadas das setas
center_points = {i: (grid_centers[i][0], grid_centers[i][1]) for i in range(9)}
arrows = {
    "R": [(center_points[8], center_points[2])], "R'": [(center_points[2], center_points[8])],
    "L": [(center_points[0], center_points[6])], "L'": [(center_points[6], center_points[0])],
    "U": [(center_points[2], center_points[0])], "U'": [(center_points[0], center_points[2])],
    "D": [(center_points[6], center_points[8])], "D'": [(center_points[8], center_points[6])],
    "F": [((center_points[8][0]-10, center_points[8][1]), (center_points[6][0], center_points[6][1]+10)),
          ((center_points[6][0], center_points[6][1]-10), (center_points[0][0]+10, center_points[0][1])),
          ((center_points[0][0]+10, center_points[0][1]), (center_points[2][0], center_points[2][1]-10)),
          ((center_points[2][0], center_points[2][1]+10), (center_points[8][0]-10, center_points[8][1]))],
    "F'": [((center_points[6][0], center_points[6][1]+10), (center_points[8][0]-10, center_points[8][1])),
           ((center_points[0][0]+10, center_points[0][1]), (center_points[6][0], center_points[6][1]-10)),
           ((center_points[2][0], center_points[2][1]-10), (center_points[0][0]+10, center_points[0][1])),
           ((center_points[8][0]-10, center_points[8][1]), (center_points[2][0], center_points[2][1]+10))],
    "B": [], "B'": [], "B2": [],
    # <--- CORRIGIDO: Setas de giro do cubo estavam trocadas ---
    "TURN_R": [(center_points[6], center_points[8]), (center_points[3], center_points[5]), (center_points[0], center_points[2])],
    "TURN_L": [(center_points[8], center_points[6]), (center_points[5], center_points[3]), (center_points[2], center_points[0])],
}

# --- Funções de Rotação Lógica ---

def right_cw(video, u, r, f, d, l, b, *args):
    # R (CW): U -> F -> D -> B(inv) -> U
    if any(face is None for face in [u, r, f, d, b]): print("DEBUG: right_cw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    d_before = np.copy(d)
    b_before = np.copy(b)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [2, 5, 8]] = f_before[0, [2, 5, 8]]
    expected_f[0, [2, 5, 8]] = d_before[0, [2, 5, 8]]
    expected_d[0, [2, 5, 8]] = b_before[0, [6, 3, 0]]
    expected_b[0, [6, 3, 0]] = u_before[0, [2, 5, 8]]

    rotated_r = rotate_cw(r)
    if rotated_r is None: print("DEBUG: rotate_cw(r) falhou."); return u, r, f, d, l, b
    expected_r = rotated_r
    
    if wait_for_move(video, expected_f, f_before, "R", arrows["R"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def right_ccw(video, u, r, f, d, l, b, *args):
    # R' (CCW): U -> B(inv) -> D -> F -> U
    if any(face is None for face in [u, r, f, d, b]): print("DEBUG: right_ccw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    d_before = np.copy(d)
    b_before = np.copy(b)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [2, 5, 8]] = b_before[0, [6, 3, 0]]
    expected_f[0, [2, 5, 8]] = u_before[0, [2, 5, 8]]
    expected_d[0, [2, 5, 8]] = f_before[0, [2, 5, 8]]
    expected_b[0, [6, 3, 0]] = d_before[0, [2, 5, 8]]

    rotated_r = rotate_ccw(r)
    if rotated_r is None: print("DEBUG: rotate_ccw(r) falhou."); return u, r, f, d, l, b
    expected_r = rotated_r
    
    if wait_for_move(video, expected_f, f_before, "R'", arrows["R'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def left_cw(video, u, r, f, d, l, b, *args):
    # L (CW): U -> B(inv) -> D -> F -> U
    if any(face is None for face in [u, l, f, d, b]): print("DEBUG: left_cw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    d_before = np.copy(d)
    b_before = np.copy(b)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [0, 3, 6]] = b_before[0, [8, 5, 2]]
    expected_f[0, [0, 3, 6]] = u_before[0, [0, 3, 6]]
    expected_d[0, [0, 3, 6]] = f_before[0, [0, 3, 6]]
    expected_b[0, [8, 5, 2]] = d_before[0, [0, 3, 6]]

    rotated_l = rotate_cw(l)
    if rotated_l is None: print("DEBUG: rotate_cw(l) falhou."); return u, r, f, d, l, b
    expected_l = rotated_l
    
    if wait_for_move(video, expected_f, f_before, "L", arrows["L"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def left_ccw(video, u, r, f, d, l, b, *args):
    # L' (CCW): U -> F -> D -> B(inv) -> U
    if any(face is None for face in [u, l, f, d, b]): print("DEBUG: left_ccw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    d_before = np.copy(d)
    b_before = np.copy(b)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [0, 3, 6]] = f_before[0, [0, 3, 6]]
    expected_f[0, [0, 3, 6]] = d_before[0, [0, 3, 6]]
    expected_d[0, [0, 3, 6]] = b_before[0, [8, 5, 2]]
    expected_b[0, [8, 5, 2]] = u_before[0, [0, 3, 6]]

    rotated_l = rotate_ccw(l)
    if rotated_l is None: print("DEBUG: rotate_ccw(l) falhou."); return u, r, f, d, l, b
    expected_l = rotated_l
    
    if wait_for_move(video, expected_f, f_before, "L'", arrows["L'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def up_cw(video, u, r, f, d, l, b, *args):
    # U (CW): F -> L -> B -> R -> F
    if any(face is None for face in [u, r, f, l, b]): print("DEBUG: up_cw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    r_before = np.copy(r)
    l_before = np.copy(l)
    b_before = np.copy(b)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    
    expected_f[0, [0, 1, 2]] = r_before[0, [0, 1, 2]]
    expected_r[0, [0, 1, 2]] = b_before[0, [0, 1, 2]]
    expected_b[0, [0, 1, 2]] = l_before[0, [0, 1, 2]]
    expected_l[0, [0, 1, 2]] = f_before[0, [0, 1, 2]]

    rotated_u = rotate_cw(u)
    if rotated_u is None: print("DEBUG: rotate_cw(u) falhou."); return u, r, f, d, l, b
    expected_u = rotated_u
    
    if wait_for_move(video, expected_f, f_before, "U", arrows["U"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def up_ccw(video, u, r, f, d, l, b, *args):
    # U' (CCW): F -> R -> B -> L -> F
    if any(face is None for face in [u, r, f, l, b]): print("DEBUG: up_ccw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    r_before = np.copy(r)
    l_before = np.copy(l)
    b_before = np.copy(b)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_f[0, [0, 1, 2]] = l_before[0, [0, 1, 2]]
    expected_l[0, [0, 1, 2]] = b_before[0, [0, 1, 2]]
    expected_b[0, [0, 1, 2]] = r_before[0, [0, 1, 2]]
    expected_r[0, [0, 1, 2]] = f_before[0, [0, 1, 2]]

    rotated_u = rotate_ccw(u)
    if rotated_u is None: print("DEBUG: rotate_ccw(u) falhou."); return u, r, f, d, l, b
    expected_u = rotated_u
    
    if wait_for_move(video, expected_f, f_before, "U'", arrows["U'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def down_cw(video, u, r, f, d, l, b, *args):
    # D (CW): F -> R -> B -> L -> F
    if any(face is None for face in [d, r, f, l, b]): print("DEBUG: down_cw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    r_before = np.copy(r)
    l_before = np.copy(l)
    b_before = np.copy(b)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    
    expected_f[0, [6, 7, 8]] = l_before[0, [6, 7, 8]]
    expected_l[0, [6, 7, 8]] = b_before[0, [6, 7, 8]]
    expected_b[0, [6, 7, 8]] = r_before[0, [6, 7, 8]]
    expected_r[0, [6, 7, 8]] = f_before[0, [6, 7, 8]]
    
    rotated_d = rotate_cw(d)
    if rotated_d is None: print("DEBUG: rotate_cw(d) falhou."); return u, r, f, d, l, b
    expected_d = rotated_d
    
    if wait_for_move(video, expected_f, f_before, "D", arrows["D"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def down_ccw(video, u, r, f, d, l, b, *args):
    # D' (CCW): F -> L -> B -> R -> F
    if any(face is None for face in [d, r, f, l, b]): print("DEBUG: down_ccw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    r_before = np.copy(r)
    l_before = np.copy(l)
    b_before = np.copy(b)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_f[0, [6, 7, 8]] = r_before[0, [6, 7, 8]]
    expected_r[0, [6, 7, 8]] = b_before[0, [6, 7, 8]]
    expected_b[0, [6, 7, 8]] = l_before[0, [6, 7, 8]]
    expected_l[0, [6, 7, 8]] = f_before[0, [6, 7, 8]]
    
    rotated_d = rotate_ccw(d)
    if rotated_d is None: print("DEBUG: rotate_ccw(d) falhou."); return u, r, f, d, l, b
    expected_d = rotated_d
    
    if wait_for_move(video, expected_f, f_before, "D'", arrows["D'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def front_cw(video, u, r, f, d, l, b, *args):
    # F (CW): U -> R -> D(inv) -> L -> U
    if any(face is None for face in [u, r, f, d, l]): print("DEBUG: front_cw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    r_before = np.copy(r)
    d_before = np.copy(d)
    l_before = np.copy(l)
    
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [6, 7, 8]] = l_before[0, [8, 5, 2]] # U[6,7,8] <- L[8,5,2]
    expected_r[0, [0, 3, 6]] = u_before[0, [6, 7, 8]] # R[0,3,6] <- U[6,7,8]
    expected_d[0, [0, 1, 2]] = r_before[0, [6, 3, 0]] # D[0,1,2] <- R[6,3,0]
    expected_l[0, [2, 5, 8]] = d_before[0, [0, 1, 2]] # L[2,5,8] <- D[0,1,2]

    rotated_f = rotate_cw(f)
    if rotated_f is None: print("DEBUG: rotate_cw(f) falhou."); return u, r, f, d, l, b
    expected_f = rotated_f
    
    if wait_for_move(video, expected_f, f_before, "F", arrows["F"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def front_ccw(video, u, r, f, d, l, b, *args):
    # F' (CCW): U -> L -> D(inv) -> R -> U
    if any(face is None for face in [u, r, f, d, l]): print("DEBUG: front_ccw recebeu None."); return u, r, f, d, l, b
    
    f_before = np.copy(f)
    u_before = np.copy(u)
    r_before = np.copy(r)
    d_before = np.copy(d)
    l_before = np.copy(l)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [6, 7, 8]] = r_before[0, [0, 3, 6]]
    expected_r[0, [6, 3, 0]] = d_before[0, [0, 1, 2]]
    expected_d[0, [0, 1, 2]] = l_before[0, [2, 5, 8]]
    expected_l[0, [8, 5, 2]] = u_before[0, [6, 7, 8]]

    rotated_f = rotate_ccw(f)
    if rotated_f is None: print("DEBUG: rotate_ccw(f) falhou."); return u, r, f, d, l, b
    expected_f = rotated_f
    
    if wait_for_move(video, expected_f, f_before, "F'", arrows["F'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: 
        return u, r, f, d, l, b

def back_cw(video, u, r, f, d, l, b, *args):
    # B (CW): U -> L -> D(inv) -> R -> U
    # Esta função só é usada para cálculo interno, não para wait_for_move
    if any(face is None for face in [u, r, d, l, b]): print("DEBUG: back_cw recebeu None."); return u, r, f, d, l, b
    
    u_before = np.copy(u)
    r_before = np.copy(r)
    d_before = np.copy(d)
    l_before = np.copy(l)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)

    expected_u[0, [0, 1, 2]] = r_before[0, [2, 5, 8]]
    expected_r[0, [2, 5, 8]] = d_before[0, [8, 7, 6]]
    expected_d[0, [8, 7, 6]] = l_before[0, [6, 3, 0]]
    expected_l[0, [6, 3, 0]] = u_before[0, [0, 1, 2]]

    rotated_b = rotate_cw(b)
    if rotated_b is None: print("DEBUG: rotate_cw(b) falhou."); return u, r, f, d, l, b
    expected_b = rotated_b
    
    return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b


def back_ccw(video, u, r, f, d, l, b, *args):
    # B' (CCW): U -> R -> D(inv) -> L -> U
    # Esta função só é usada para cálculo interno, não para wait_for_move
    if any(face is None for face in [u, r, d, l, b]): print("DEBUG: back_ccw recebeu None."); return u, r, f, d, l, b
    
    u_before = np.copy(u)
    r_before = np.copy(r)
    d_before = np.copy(d)
    l_before = np.copy(l)

    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    
    expected_u[0, [0, 1, 2]] = l_before[0, [6, 3, 0]]
    expected_l[0, [6, 3, 0]] = d_before[0, [8, 7, 6]]
    expected_d[0, [8, 7, 6]] = r_before[0, [2, 5, 8]]
    expected_r[0, [2, 5, 8]] = u_before[0, [0, 1, 2]]

    rotated_b = rotate_ccw(b)
    if rotated_b is None: print("DEBUG: rotate_ccw(b) falhou."); return u, r, f, d, l, b
    expected_b = rotated_b
    
    return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b

# <--- CORRIGIDO: Funções de giro do cubo com lógica e nomes trocados ---

# Y (CW, Horário): F->R, R->B, B->L, L->F
def turn_cube_Y(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l, b]): print("DEBUG: turn_Y (Girar Direita) recebeu None."); return u, r, f, d, l, b
    print("Vire o cubo todo para a DIREITA (Y)")
    f_before = np.copy(f) # Salva a face F (Front) original

    # Lógica Y (Horário): F->R, R->B, B->L, L->F
    new_f, new_r, new_b, new_l = r, b, l, f

    rotated_u = rotate_cw(u)  # U gira horário
    rotated_d = rotate_ccw(d) # D gira anti-horário (visto de D)
    if rotated_u is None or rotated_d is None: print("DEBUG: rotação U/D falhou em turn_Y."); return u, r, f, d, l, b
    new_u, new_d = rotated_u, rotated_d

    print("Mostre a nova FACE FRONTAL (que era a Direita)")
    # Esperamos ver a face 'R' original, que é 'new_f'
    if wait_for_move(video, new_f, f_before, "VIRAR P/ DIREITA: Mostre a nova face F", arrows["TURN_R"]):
        return new_u, new_r, new_f, new_d, new_l, new_b
    else: return u, r, f, d, l, b

# Y' (CCW, Anti-horário): F->L, L->B, B->R, R->F
def turn_cube_Y_prime(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l, b]): print("DEBUG: turn_Y' (Girar Esquerda) recebeu None."); return u, r, f, d, l, b
    print("Vire o cubo todo para a ESQUERDA (Y')")
    f_before = np.copy(f) # Salva a face F (Front) original

    # Lógica Y' (Anti-horário): F->L, L->B, B->R, R->F
    new_f, new_r, new_b, new_l = l, f, r, b

    rotated_u = rotate_ccw(u) # U gira anti-horário
    rotated_d = rotate_cw(d)  # D gira horário (visto de D)
    if rotated_u is None or rotated_d is None: print("DEBUG: rotação U/D falhou em turn_Y'."); return u, r, f, d, l, b
    new_u, new_d = rotated_u, rotated_d

    print("Mostre a nova FACE FRONTAL (que era a Esquerda)")
    # Esperamos ver a face 'L' original, que é 'new_f'
    if wait_for_move(video, new_f, f_before, "VIRAR P/ ESQUERDA: Mostre a nova face F", arrows["TURN_L"]):
        return new_u, new_r, new_f, new_d, new_l, new_b
    else: return u, r, f, d, l, b


# <--- INÍCIO DAS NOVAS FUNÇÕES WRAPPER (CORRIGIDAS) ---
# A lógica de conversão correta é:
# B  = Y  R' Y'
# B' = Y  R  Y'
# B2 = Y  R2 Y'

def handle_back_cw(video, u, r, f, d, l, b, *args):
    """
    Executa um movimento B (CW) usando a conversão: Y  R' Y'
    (Gira Direita, R anti-horário, Gira Esquerda)
    """
    print("INFO: Movimento 'B' (Trás) solicitado. Reorientando o cubo...")
    
    # 1. Turn Cube Y (Girar para Direita)
    # new_F = old_R, new_R = old_B, new_B = old_L, new_L = old_F
    new_u, new_r, new_f, new_d, new_l, new_b = turn_cube_Y(video, u, r, f, d, l, b, *args)
    if np.array_equal(new_f, f): # Checa se a rotação Y foi interrompida
        print("DEBUG: Rotação Y interrompida.")
        return u, r, f, d, l, b # Retorna estado original para a main() detectar

    # 2. Executar R' (movimento B convertido)
    print("INFO: Executando 'R'' (que é o 'B' original)")
    u2, r2, f2, d2, l2, b2 = right_ccw(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)

    if np.array_equal(f2, new_f): # Interrompeu no R'
        print("DEBUG: Movimento 'R'' (convertido) interrompido. Desfazendo rotação Y...")
        # Gira de volta ANTES de retornar
        u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)
        return u, r, f, d, l, b # Retorna estado original TOTAL

    # 3. Turn Cube Y' (Girar de volta)
    print("INFO: Retornando à face frontal original...")
    u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, u2, r2, f2, d2, l2, b2, *args)

    if np.array_equal(f3, f2): # Interrompeu no Y' de volta
        print("DEBUG: Rotação Y' (retorno) interrompida.")
        return u, r, f, d, l, b 

    return u3, r3, f3, d3, l3, b3 # Retorna estado final corrigido

def handle_back_ccw(video, u, r, f, d, l, b, *args):
    """
    Executa um movimento B' (CCW) usando a conversão: Y  R  Y'
    (Gira Direita, R horário, Gira Esquerda)
    """
    print("INFO: Movimento 'B'' (Trás) solicitado. Reorientando o cubo...")
    
    # 1. Turn Cube Y (Girar para Direita)
    new_u, new_r, new_f, new_d, new_l, new_b = turn_cube_Y(video, u, r, f, d, l, b, *args)
    if np.array_equal(new_f, f): # Interrompeu no turn_Y
        print("DEBUG: Rotação Y interrompida.")
        return u, r, f, d, l, b 

    # 2. Executar R (movimento B' convertido)
    print("INFO: Executando 'R' (que é o 'B'' original)")
    u2, r2, f2, d2, l2, b2 = right_cw(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)

    if np.array_equal(f2, new_f): # Interrompeu no R
        print("DEBUG: Movimento 'R' (convertido) interrompido. Desfazendo rotação Y...")
        u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)
        return u, r, f, d, l, b 

    # 3. Turn Cube Y' (Girar de volta)
    print("INFO: Retornando à face frontal original...")
    u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, u2, r2, f2, d2, l2, b2, *args)

    if np.array_equal(f3, f2): # Interrompeu no Y' de volta
        print("DEBUG: Rotação Y' (retorno) interrompida.")
        return u, r, f, d, l, b 

    return u3, r3, f3, d3, l3, b3

def handle_back_2(video, u, r, f, d, l, b, *args):
    """
    Executa um movimento B2 usando a conversão: Y  R2  Y'
    (Gira Direita, R2, Gira Esquerda)
    """
    print("INFO: Movimento 'B2' (Trás) solicitado. Reorientando o cubo...")

    # 1. Turn Cube Y (Girar para Direita)
    new_u, new_r, new_f, new_d, new_l, new_b = turn_cube_Y(video, u, r, f, d, l, b, *args)
    if np.array_equal(new_f, f): # Interrompeu no turn_Y
        print("DEBUG: Rotação Y interrompida.")
        return u, r, f, d, l, b 

    # 2. Executar R2 (movimento B2 convertido)
    print("INFO: Executando 'R2' (que é o 'B2' original)")
    # R2 = R + R
    u_temp, r_temp, f_temp, d_temp, l_temp, b_temp = right_cw(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)
    
    if np.array_equal(f_temp, new_f): # Checa interrupção no primeiro R
        print("DEBUG: Movimento 'R' (1/2 de R2) interrompido. Desfazendo rotação Y...")
        u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, new_u, new_r, new_f, new_d, new_l, new_b, *args)
        return u, r, f, d, l, b # Retorna estado original TOTAL
    
    # Segundo R
    u2, r2, f2, d2, l2, b2 = right_cw(video, u_temp, r_temp, f_temp, d_temp, l_temp, b_temp, *args)

    if np.array_equal(f2, f_temp): # Checa interrupção no segundo R
        print("DEBUG: Movimento 'R' (2/2 de R2) interrompido. Desfazendo rotação Y...")
        u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, u_temp, r_temp, f_temp, d_temp, l_temp, b_temp, *args)
        return u, r, f, d, l, b

    # 3. Turn Cube Y' (Girar de volta)
    print("INFO: Retornando à face frontal original...")
    u3, r3, f3, d3, l3, b3 = turn_cube_Y_prime(video, u2, r2, f2, d2, l2, b2, *args)

    if np.array_equal(f3, f2): # Interrompeu no Y' de volta
        print("DEBUG: Rotação Y' (retorno) interrompida.")
        return u, r, f, d, l, b

    return u3, r3, f3, d3, l3, b3
# <--- FIM DAS NOVAS FUNÇÕES WRAPPER ---


# Mapeamento de strings de movimento para funções
move_functions = {
    "R": right_cw, "R'": right_ccw,
    "L": left_cw,  "L'": left_ccw,
    "U": up_cw,    "U'": up_ccw,
    "D": down_cw,  "D'": down_ccw,
    "F": front_cw, "F'": front_ccw,
    # As funções B originais não são mais necessárias aqui
}

# Adiciona movimentos duplos (X2) dinamicamente para R,L,U,D,F
for move in list(move_functions.keys()):
    if "'" not in move:
        func = move_functions[move]
        move_functions[move + "2"] = lambda v, u, r, f, d, l, b, *a, _func=func: \
                                        _func(v, *_func(v, u, r, f, d, l, b, *a), *a)

# Adiciona os handlers de 'B' manualmente
move_functions["B"] = handle_back_cw
move_functions["B'"] = handle_back_ccw
move_functions["B2"] = handle_back_2



print("DEBUG: Funções interativas definidas.")  # DEBUG 5 (Fim)


# --- 5. Função Principal ---
def main():
    global num_to_kociemba_letter, kociemba_letter_to_num

    print("DEBUG: Entrando na função main()")  # DEBUG 6
    video = cv2.VideoCapture(0)  # Tenta IP Camera primeiro
    if not video.isOpened():
        print("DEBUG: Tentando webcam 1...")
        video = cv2.VideoCapture(1)
        if not video.isOpened():
            print("Erro fatal: Nenhuma webcam encontrada.")
            return
    print("DEBUG: Webcam aberta com sucesso.")  # DEBUG 7

    # Guarda o estado das 6 faces (LISTAS DE LETRAS)
    cube_state_letters = {face: None for face in faces_order}
    # O cube_state_num será criado DEPOIS do scan
    cube_state_num = {face: None for face in faces_order}

    detected_faces_buffer = []
    current_face_index = 0
    scan_complete = False
    solution_moves = []
    current_move_index = 0
    kociemba_string_generated = ""

    # Reseta mapeamentos no início
    num_to_kociemba_letter = {}
    kociemba_letter_to_num = {}

    cv2.namedWindow("Resolvendo...")
    print("DEBUG: Janela 'Resolvendo...' criada.")  # DEBUG 7.1

    print("--- Solver Interativo de Cubo Mágico ---")
    print("Instruções de Scan:")
    print("1. Certifique-se que o arquivo 'calibrated_colors.py' existe e está correto!")
    print("2. Mostre cada face do cubo alinhada com a grade.")
    print("3. Mantenha a face estável por ~1 segundo para leitura.")
    print("4. Siga as instruções no topo da tela.")
    print("5. Pressione [Q] para sair a qualquer momento.")

    while True:
        is_ok, frame = video.read()
        if not is_ok:
            print("DEBUG: Falha ao ler frame. Tentando de novo...")
            time.sleep(1); is_ok, frame = video.read()
            if not is_ok:
                print("DEBUG: Falha ao ler frame novamente. Saindo.")
                break
        if frame is None:
            print("DEBUG: Frame Nulo.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_with_grid = draw_preview_grid(frame.copy(), grid_centers)
        if frame_with_grid is None:
            print("DEBUG: draw_preview_grid falhou.")
            continue

        # --- Fase de Scan ---
        # (Nenhuma alteração nesta seção)
        if not scan_complete:
            if current_face_index >= len(faces_order):
                print("DEBUG: Erro - Índice de face inválido. Reiniciando scan.")
                current_face_index = 0
                scan_complete = False
                cube_state_letters = {f: None for f in faces_order}
                detected_faces_buffer = []

            face_code_to_scan = faces_order[current_face_index]
            face_name = face_names_pt.get(face_code_to_scan, "Desconhecida")
            text = f"Scan ({current_face_index+1}/6): Mostre {face_name} ({face_code_to_scan})"
            cv2.putText(frame_with_grid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tenta detectar a face atual (apenas letras por enquanto)
            detected_letters = []
            valid_detection = True
            try:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            except cv2.error as e:
                print(f"DEBUG: Erro HSV: {e}.")
                valid_detection = False

            if valid_detection:
                for (x, y) in grid_centers:
                    if y >= hsv_frame.shape[0] or x >= hsv_frame.shape[1] or y < 0 or x < 0:
                        valid_detection = False
                        break
                    pixel_hsv = hsv_frame[y, x]
                    color_letter = get_color_name(pixel_hsv)
                    if color_letter == '?':
                        valid_detection = False
                        break
                    detected_letters.append(color_letter)

            # --- Adiciona Texto de Debug Visual ---
            if valid_detection and len(detected_letters) == 9:
                # Desenha as letras detectadas ANTES de checar estabilidade
                for i, (x, y) in enumerate(grid_centers):
                    color_letter_detected = detected_letters[i]
                    text_pos = (x - 10, y + 5)
                    cv2.putText(frame_with_grid, color_letter_detected, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame_with_grid, color_letter_detected, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Estabilidade (3 leituras idênticas consecutivas)
                detected_faces_buffer.append(detected_letters)
                if len(detected_faces_buffer) > 10:
                    detected_faces_buffer.pop(0)

                if len(detected_faces_buffer) >= 3 and all(f == detected_letters for f in detected_faces_buffer[-3:]):
                    center_color_letter = detected_letters[4]
                    print(f"Face {face_code_to_scan} escaneada (centro={center_color_letter}): {detected_letters}")

                    # SALVA EXATAMENTE O QUE FOI VISTO PARA ESTA POSIÇÃO (U,R,F,D,L,B)
                    cube_state_letters[face_code_to_scan] = detected_letters

                    current_face_index += 1
                    detected_faces_buffer = []
                    cv2.putText(frame_with_grid, "OK!", (frame.shape[1] // 2 - 30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Resolvendo...", frame_with_grid)
                    cv2.waitKey(700)

                    # --- Lógica de Mapeamento e Solução ---
                    if current_face_index == len(faces_order):
                        scan_complete = True
                        print("\nDEBUG: Scan completo. Iniciando Mapeamento e Geração da Solução...")
                        kociemba_string = ""
                        try:
                            # --- 1. Criar Mapeamento Final (cor->numero e cor<->posicao) ---
                            print("DEBUG: Criando mapeamento baseado nos centros...")
                            # zera mapas globais
                            kociemba_letter_to_num = {}
                            num_to_kociemba_letter = {}
                            map_letra_para_posicao = {}
                            map_posicao_para_letra = {}

                            for i, face_code_kociemba in enumerate(faces_order):
                                if cube_state_letters[face_code_kociemba] is None:
                                    raise ValueError(f"Estado da face {face_code_kociemba} não foi escaneado.")
                                center_letter = cube_state_letters[face_code_kociemba][4]
                                assigned_num = i + 1
                                if center_letter in kociemba_letter_to_num:
                                    raise ValueError(f"Erro de Mapeamento: Cor central '{center_letter}' duplicada!")
                                kociemba_letter_to_num[center_letter] = assigned_num
                                num_to_kociemba_letter[assigned_num] = center_letter
                                map_letra_para_posicao[center_letter] = face_code_kociemba
                                map_posicao_para_letra[face_code_kociemba] = center_letter
                                print(f"DEBUG: Mapeado Centro '{center_letter}' -> Num {assigned_num} (Posição {face_code_kociemba})")

                            if len(kociemba_letter_to_num) != 6:
                                raise ValueError("Mapeamento incompleto.")
                            print("DEBUG: Mapeamento Cor -> Posição Kociemba:", map_letra_para_posicao)

                            # --- 2. Converter Estado para Números (para a interação) ---
                            print("DEBUG: Convertendo estado (letras) para estado (números)...")
                            for face_code in faces_order:
                                letras = cube_state_letters[face_code]
                                numeros = [kociemba_letter_to_num[l] for l in letras]
                                cube_state_num[face_code] = np.array([numeros])
                                print(f"DEBUG: Estado Numérico {face_code}: {cube_state_num[face_code]}")

                            # --- 3. Gerar String Kociemba (CORRETO) ---
                            print("DEBUG: Gerando string Kociemba (correção)...")
                            kociemba_string = ""
                            # Para cada POSIÇÃO (U,R,F,D,L,B), percorra as 9 letras lidas nessa posição.
                            # Para cada cor lida, escreva a LETRA da POSIÇÃO do centro daquela cor.
                            for face_code_posicao in faces_order:
                                letras_da_face_na_posicao = cube_state_letters[face_code_posicao]
                                for letra_peca in letras_da_face_na_posicao:
                                    posicao_daquela_cor = map_letra_para_posicao.get(letra_peca)
                                    if posicao_daquela_cor is None:
                                        raise ValueError(f"Cor '{letra_peca}' sem mapeamento de posição!")
                                    kociemba_string += posicao_daquela_cor

                            kociemba_string_generated = kociemba_string
                            print(f"String Kociemba Final: {kociemba_string}")
                            if len(kociemba_string) != 54:
                                raise ValueError(f"String Kociemba com tamanho incorreto: {len(kociemba_string)}")

                            # --- 4. Chamar Kociemba ---
                            print("DEBUG: Chamando kociemba.solve...")
                            solution = kociemba.solve(kociemba_string)
                            solution_moves = solution.split()
                            print(f"Solucao ({len(solution_moves)} mov): {solution}")
                            current_move_index = 0

                        except ValueError as ve:
                            print(f"DEBUG: Erro de Valor ao Mapear/Gerar Solução: {ve}")
                            scan_complete = False
                            current_face_index = 0
                            cube_state_letters = {f: None for f in faces_order}
                            cube_state_num = {f: None for f in faces_order}
                            num_to_kociemba_letter = {}
                            kociemba_letter_to_num = {}
                            detected_faces_buffer = []
                            time.sleep(4)
                        except Exception as e:
                            print(f"DEBUG: Erro inesperado no Mapeamento/Solução: {e}")
                            scan_complete = False
                            current_face_index = 0
                            cube_state_letters = {f: None for f in faces_order}
                            cube_state_num = {f: None for f in faces_order}
                            num_to_kociemba_letter = {}
                            kociemba_letter_to_num = {}
                            detected_faces_buffer = []
                            time.sleep(4)

                else:  # Não estável ainda
                    needed = 3 - len(detected_faces_buffer) if len(detected_faces_buffer) < 3 else 0
                    if not all(f == detected_letters for f in detected_faces_buffer[-3:]):
                        needed = 3
                    if needed > 0:
                        cv2.putText(frame_with_grid, f"Mantenha estavel... ({needed})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            else:  # Detecção inválida (letra '?' ou número errado de cores)
                detected_faces_buffer = []
                cv2.putText(frame_with_grid, "Ajuste na grade!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # --- Adiciona Texto '?' se a detecção falhou ---
                try:
                    if 'hsv_frame' in locals() and hsv_frame is not None:
                        temp_letters = []
                        temp_valid = True
                        for (x_t, y_t) in grid_centers:
                            if y_t < hsv_frame.shape[0] and x_t < hsv_frame.shape[1] and y_t >= 0 and x_t >= 0:
                                temp_letter = get_color_name(hsv_frame[y_t, x_t])
                                temp_letters.append(temp_letter)
                            else:
                                temp_letters.append('X')
                                temp_valid = False
                        if temp_valid and len(temp_letters) == 9:
                            for i, (x, y) in enumerate(grid_centers):
                                text_pos = (x - 10, y + 5)
                                cv2.putText(frame_with_grid, temp_letters[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                cv2.putText(frame_with_grid, temp_letters[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e_draw:
                    print(f"Debug: Erro menor ao tentar desenhar letras em detecção inválida: {e_draw}")

        # --- Fase de Resolução Interativa ---
        # (Nenhuma alteração nesta seção, pois a lógica foi movida
        #  para as funções 'handle_back' e o dicionário 'move_functions')
        elif current_move_index < len(solution_moves):
            move = solution_moves[current_move_index]
            move_func = move_functions.get(move)

            if move_func:
                current_u = cube_state_num.get('U'); current_r = cube_state_num.get('R')
                current_f = cube_state_num.get('F'); current_d = cube_state_num.get('D')
                current_l = cube_state_num.get('L'); current_b = cube_state_num.get('B')

                if any(face is None for face in [current_u, current_r, current_f, current_d, current_l, current_b]):
                    print("DEBUG: ERRO CRÍTICO - Estado numérico incompleto antes de aplicar movimento!")
                    break

                # Esta impressão agora mostrará 'B', 'B'' ou 'B2' corretamente.
                print(f"\nDEBUG: Chamando função para movimento {move} ({current_move_index+1}/{len(solution_moves)})")
                
                # A move_func será a 'handle_back_cw' (por exemplo) se move == 'B'
                new_u, new_r, new_f, new_d, new_l, new_b = move_func(
                    video, current_u, current_r, current_f, current_d, current_l, current_b,
                    kociemba_letter_to_num,
                    grid_centers[0][0] - 70, grid_centers[0][1] - 70, 70, 70
                )

                # Verifica interrupção (a função wrapper handle_back JÁ retorna o estado
                # original (current_u, ...) se for interrompida em qualquer etapa)
                interrupted = (np.array_equal(new_u, current_u) and np.array_equal(new_r, current_r) and
                               np.array_equal(new_f, current_f) and np.array_equal(new_d, current_d) and
                               np.array_equal(new_l, current_l) and np.array_equal(new_b, current_b))

                if interrupted: # A checagem de 'wait_for_move' é desnecessária
                    print("DEBUG: Execução interrompida pelo usuário ('q').")
                    break

                if not interrupted:
                    cube_state_num['U'] = new_u; cube_state_num['R'] = new_r
                    cube_state_num['F'] = new_f; cube_state_num['D'] = new_d
                    cube_state_num['L'] = new_l; cube_state_num['B'] = new_b
                    current_move_index += 1
                    print(f"DEBUG: Movimento {move} concluído.")

            else:
                print(f"DEBUG: Erro - Movimento '{move}' desconhecido.")
                current_move_index += 1
        else:  # Fim da solução
            print("DEBUG: Fim da solução.")
            cv2.putText(frame_with_grid, "CUBO RESOLVIDO!", (frame.shape[1] // 2 - 150, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Resolvendo...", frame_with_grid)
            cv2.waitKey(3000)
            break

        # Mostra o frame final do loop
        if cv2.getWindowProperty("Resolvendo...", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow("Resolvendo...", frame_with_grid)
        else:
            print("DEBUG: Janela 'Resolvendo...' foi fechada. Saindo.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("DEBUG: 'q' pressionado no loop principal.")
            break

    # --- Fim ---
    print("DEBUG: Saindo do loop while.")  # DEBUG 13
    video.release()
    cv2.destroyAllWindows()
    print("DEBUG: Recursos liberados.")  # DEBUG 14
    for _ in range(5): cv2.waitKey(1)
    print("--- Fim do Script ---")  # DEBUG 15


if __name__ == "__main__":
    main()