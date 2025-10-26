# salve como solver_interativo_setas_debug.py
import cv2
import numpy as np
import kociemba
import time
import sys
import os # Importa o módulo 'os' para verificar se o arquivo existe
from scipy import stats # Para ajudar na estabilidade da detecção

print("--- Iniciando Script ---") # DEBUG 1

# --- 1. Carregar Valores Calibrados ---
# Tenta importar os valores salvos pelo calibrador_com_save.py
try:
    if os.path.exists("calibrated_colors.py"):
        from calibrated_colors import calibrated_values
        color_ranges = calibrated_values
        print("DEBUG: Valores carregados com sucesso.") # DEBUG 2
        # Verifica se todas as 6 cores foram carregadas
        if len(color_ranges) != 6:
             print("\n!!! ATENCAO !!!")
             print(f"O arquivo 'calibrated_colors.py' contem apenas {len(color_ranges)} cores.")
             print("Execute o 'calibrador_com_save.py' novamente e calibre TODAS as 6 cores.")
             print("O programa pode falhar se uma cor estiver faltando.")
             print("Cores carregadas:", list(color_ranges.keys()))
             print("!!! ATENCAO !!!\n")
             # time.sleep(5) # Pausa para ler
             # exit() # Descomente para sair se faltar cor
        else:
            print("DEBUG: Arquivo 'calibrated_colors.py' contem 6 cores.") # DEBUG 2.1

    else:
        # Define valores de ERRO se o arquivo não existir
        print("\n!!! ERRO FATAL !!!")
        print("Arquivo 'calibrated_colors.py' não encontrado na pasta atual.")
        print("Execute o 'calibrador_com_save.py' primeiro para calibrar as cores.")
        print("!!! ERRO FATAL !!!\n")
        exit() # Termina a execução

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
print("DEBUG: Definindo Configurações...") # DEBUG 3
# Posições dos centros dos 9 quadrados (x, y) - Ajuste se necessário
grid_centers = [
    (250, 170), (320, 170), (390, 170), # Linha de cima
    (250, 240), (320, 240), (390, 240), # Linha do meio
    (250, 310), (320, 310), (390, 310)  # Linha de baixo
]
grid_map = np.array(range(9)).reshape(3, 3) # Mapa 0-8 para posições lógicas

# Ordem das faces para escanear e padrão Kociemba
faces_order = ['U', 'R', 'F', 'D', 'L', 'B']
face_names_pt = { # Nomes para instruções
    'U': 'Cima (Branca)', 'R': 'Direita (Vermelha)', 'F': 'Frente (Verde)',
    'D': 'Baixo (Amarela)', 'L': 'Esquerda (Laranja)', 'B': 'Trás (Azul)'
}

# Mapeamento interno: Número da cor (baseado no centro) para letra Kociemba
# Será preenchido após escanear
num_to_kociemba_letter = {}
kociemba_letter_to_num = {}
print("DEBUG: Configurações definidas.") # DEBUG 3 (Fim)


# --- 3. Funções Auxiliares ---
print("DEBUG: Definindo Funções Auxiliares...") # DEBUG 4

def get_color_name(hsv_pixel):
    """ Retorna a letra da cor ('U', 'R', 'F'...) ou '?' """
    h, s, v = hsv_pixel

    # Itera sobre os ranges carregados do arquivo
    for color_name, (lower, upper) in color_ranges.items():
        h_min, s_min, v_min = lower
        h_max, s_max, v_max = upper

        # Lógica para checar se o pixel está no range (incluindo wrap-around do HUE)
        hue_match = False
        if h_min <= h_max: # Range normal de HUE
            if h >= h_min and h <= h_max:
                hue_match = True
        else: # Range de HUE que dá a volta (ex: vermelho Hmin=170, Hmax=10)
            if (h >= h_min and h <= 179) or (h >= 0 and h <= h_max):
                hue_match = True

        # Se o HUE bateu, checa Saturação e Valor
        if hue_match:
            if s >= s_min and s <= s_max and v >= v_min and v <= v_max:
                return color_name # Encontrou a cor!

    return '?' # Nenhuma cor encontrada

def draw_preview_grid(frame, centers):
    """ Desenha a grade de 9 quadrados """
    if frame is None: return None # Adiciona checagem
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
    try: # Adiciona try-except para a conversão de cor
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"DEBUG: Erro ao converter frame para HSV: {e}")
        return None # Frame inválido

    detected_colors_letters = []
    detected_colors_numbers = []

    for idx, (x, y) in enumerate(centers):
        # Verifica limites antes de acessar pixel
        if y >= hsv_frame.shape[0] or x >= hsv_frame.shape[1] or y < 0 or x < 0:
            print(f"DEBUG: Coordenada {idx} ({x},{y}) fora dos limites ({hsv_frame.shape[0]}x{hsv_frame.shape[1]}).")
            return None # Coordenada inválida

        pixel_hsv = hsv_frame[y, x]
        color_letter = get_color_name(pixel_hsv)
        detected_colors_letters.append(color_letter)

        if color_letter == '?':
            # print(f"DEBUG: Cor não detectada em ({x},{y}), HSV={pixel_hsv}") # Muito verbose
            return None # Falha na detecção

        # Converte letra para número usando o mapa criado no scan
        color_num = color_map.get(color_letter) # Pega o número associado à letra
        if color_num is None: # Verifica se a letra realmente existe no mapa
            # É NORMAL dar erro aqui durante o scan, pois o mapa ainda não está completo
            # print(f"DEBUG: ERRO (Esperado no Scan) - Cor '{color_letter}' não está no mapeamento 'kociemba_letter_to_num'.")
            # print("DEBUG: Mapeamento atual:", kociemba_letter_to_num)
            # Retorna None SE ESTIVER NA FASE DE RESOLUÇÃO, mas não no scan
            # No entanto, a função detect_face_from_webcam só é chamada com color_map
            # preenchido DURANTE a resolução. Durante o scan, usamos outra lógica.
            # Se chegou aqui na RESOLUÇÃO, é um erro grave.
             print(f"DEBUG: ERRO CRÍTICO no Mapeamento durante RESOLUÇÃO - Cor '{color_letter}' (HSV={pixel_hsv}) detectada em ({x},{y}), mas não está no mapeamento 'kociemba_letter_to_num'.")
             print("DEBUG: Mapeamento:", kociemba_letter_to_num)
             return None # Retorna None se o mapeamento falhar durante a resolução
        detected_colors_numbers.append(color_num)

    if len(detected_colors_numbers) == 9:
        return np.array([detected_colors_numbers]) # Retorna como array NumPy 1x9
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

print("DEBUG: Funções auxiliares definidas.") # DEBUG 4 (Fim)


# --- 4. Funções Interativas com Setas ---
print("DEBUG: Definindo Funções Interativas...") # DEBUG 5

def wait_for_move(video, expected_front_face, state_before_front, move_name, arrow_coords):
    print(f"DEBUG: Entrou em wait_for_move para {move_name}")
    # print(f"DEBUG: Esperando face: {expected_front_face}") # Muito verbose
    # print(f"DEBUG: Estado anterior: {state_before_front}") # Muito verbose

    if expected_front_face is None:
         print("DEBUG: ERRO - expected_front_face é None em wait_for_move.")
         return False # Não podemos comparar com None

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
        if frame_with_grid is None: continue # Checa se draw_preview_grid falhou

        # Detecta a face atual - AGORA USA O MAPEAMENTO COMPLETO
        current_face_state_num = detect_face_from_webcam(frame, grid_centers, kociemba_letter_to_num)

        # Mostra instrução
        cv2.putText(frame_with_grid, f"Faca o movimento: {move_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if current_face_state_num is not None:
            # --- Desenha as letras detectadas DURANTE a espera ---
            # Converte números de volta para letras para exibição
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
                 # print("DEBUG: Estado anterior detectado, desenhando seta.") # Muito verbose
                 for p1, p2 in arrow_coords:
                     try: # Adiciona try-except para desenho da seta
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
            break # Sai do loop se a janela foi fechada

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
    "R": [ (center_points[8], center_points[2]) ], "R'": [ (center_points[2], center_points[8]) ],
    "L": [ (center_points[0], center_points[6]) ], "L'": [ (center_points[6], center_points[0]) ],
    "U": [ (center_points[2], center_points[0]) ], "U'": [ (center_points[0], center_points[2]) ],
    "D": [ (center_points[6], center_points[8]) ], "D'": [ (center_points[8], center_points[6]) ],
    "F": [ ((center_points[8][0]-10, center_points[8][1]), (center_points[6][0], center_points[6][1]+10)),
           ((center_points[6][0], center_points[6][1]-10), (center_points[0][0]+10, center_points[0][1])),
           ((center_points[0][0]+10, center_points[0][1]), (center_points[2][0], center_points[2][1]-10)),
           ((center_points[2][0], center_points[2][1]+10), (center_points[8][0]-10, center_points[8][1])) ],
    "F'": [ ((center_points[6][0], center_points[6][1]+10), (center_points[8][0]-10, center_points[8][1])),
            ((center_points[0][0]+10, center_points[0][1]), (center_points[6][0], center_points[6][1]-10)),
            ((center_points[2][0], center_points[2][1]-10), (center_points[0][0]+10, center_points[0][1])),
            ((center_points[8][0]-10, center_points[8][1]), (center_points[2][0], center_points[2][1]+10)) ],
     "B": [], "B'": [], "B2": [],
     "TURN_R": [ (center_points[8], center_points[6]), (center_points[5], center_points[3]), (center_points[2], center_points[0]) ],
     "TURN_L": [ (center_points[6], center_points[8]), (center_points[3], center_points[5]), (center_points[0], center_points[2]) ],
}

# --- Funções de Rotação Lógica (Modificadas para checar Nones) ---
def right_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, b]): print("DEBUG: right_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [2, 5, 8]] = d[0, [2, 5, 8]]
    expected_d[0, [2, 5, 8]] = b[0, [6, 3, 0]]
    expected_b[0, [6, 3, 0]] = u[0, [2, 5, 8]]
    expected_u[0, [2, 5, 8]] = f_before[0, [2, 5, 8]]
    rotated_r = rotate_cw(r)
    if rotated_r is None: print("DEBUG: rotate_cw(r) falhou."); return u, r, f, d, l, b
    expected_r = rotated_r
    if wait_for_move(video, expected_f, f_before, "R", arrows["R"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def right_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, b]): print("DEBUG: right_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [2, 5, 8]] = u[0, [2, 5, 8]]
    expected_u[0, [2, 5, 8]] = b[0, [6, 3, 0]]
    expected_b[0, [6, 3, 0]] = d[0, [2, 5, 8]]
    expected_d[0, [2, 5, 8]] = f_before[0, [2, 5, 8]]
    rotated_r = rotate_ccw(r)
    if rotated_r is None: print("DEBUG: rotate_ccw(r) falhou."); return u, r, f, d, l, b
    expected_r = rotated_r
    if wait_for_move(video, expected_f, f_before, "R'", arrows["R'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def left_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, l, f, d, b]): print("DEBUG: left_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [0, 3, 6]] = u[0, [0, 3, 6]]
    expected_u[0, [0, 3, 6]] = b[0, [8, 5, 2]]
    expected_b[0, [8, 5, 2]] = d[0, [0, 3, 6]]
    expected_d[0, [0, 3, 6]] = f_before[0, [0, 3, 6]]
    rotated_l = rotate_cw(l)
    if rotated_l is None: print("DEBUG: rotate_cw(l) falhou."); return u, r, f, d, l, b
    expected_l = rotated_l
    if wait_for_move(video, expected_f, f_before, "L", arrows["L"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def left_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, l, f, d, b]): print("DEBUG: left_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [0, 3, 6]] = d[0, [0, 3, 6]]
    expected_d[0, [0, 3, 6]] = b[0, [8, 5, 2]]
    expected_b[0, [8, 5, 2]] = u[0, [0, 3, 6]]
    expected_u[0, [0, 3, 6]] = f_before[0, [0, 3, 6]]
    rotated_l = rotate_ccw(l)
    if rotated_l is None: print("DEBUG: rotate_ccw(l) falhou."); return u, r, f, d, l, b
    expected_l = rotated_l
    if wait_for_move(video, expected_f, f_before, "L'", arrows["L'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def up_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, l, b]): print("DEBUG: up_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [0, 1, 2]] = r[0, [0, 1, 2]]
    expected_r[0, [0, 1, 2]] = b[0, [0, 1, 2]]
    expected_b[0, [0, 1, 2]] = l[0, [0, 1, 2]]
    expected_l[0, [0, 1, 2]] = f_before[0, [0, 1, 2]]
    rotated_u = rotate_cw(u)
    if rotated_u is None: print("DEBUG: rotate_cw(u) falhou."); return u, r, f, d, l, b
    expected_u = rotated_u
    if wait_for_move(video, expected_f, f_before, "U", arrows["U"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def up_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, l, b]): print("DEBUG: up_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [0, 1, 2]] = l[0, [0, 1, 2]]
    expected_l[0, [0, 1, 2]] = b[0, [0, 1, 2]]
    expected_b[0, [0, 1, 2]] = r[0, [0, 1, 2]]
    expected_r[0, [0, 1, 2]] = f_before[0, [0, 1, 2]]
    rotated_u = rotate_ccw(u)
    if rotated_u is None: print("DEBUG: rotate_ccw(u) falhou."); return u, r, f, d, l, b
    expected_u = rotated_u
    if wait_for_move(video, expected_f, f_before, "U'", arrows["U'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def down_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [d, r, f, l, b]): print("DEBUG: down_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [6, 7, 8]] = l[0, [6, 7, 8]]
    expected_l[0, [6, 7, 8]] = b[0, [6, 7, 8]]
    expected_b[0, [6, 7, 8]] = r[0, [6, 7, 8]]
    expected_r[0, [6, 7, 8]] = f_before[0, [6, 7, 8]]
    rotated_d = rotate_cw(d)
    if rotated_d is None: print("DEBUG: rotate_cw(d) falhou."); return u, r, f, d, l, b
    expected_d = rotated_d
    if wait_for_move(video, expected_f, f_before, "D", arrows["D"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def down_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [d, r, f, l, b]): print("DEBUG: down_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    expected_f[0, [6, 7, 8]] = r[0, [6, 7, 8]]
    expected_r[0, [6, 7, 8]] = b[0, [6, 7, 8]]
    expected_b[0, [6, 7, 8]] = l[0, [6, 7, 8]]
    expected_l[0, [6, 7, 8]] = f_before[0, [6, 7, 8]]
    rotated_d = rotate_ccw(d)
    if rotated_d is None: print("DEBUG: rotate_ccw(d) falhou."); return u, r, f, d, l, b
    expected_d = rotated_d
    if wait_for_move(video, expected_f, f_before, "D'", arrows["D'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def front_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l]): print("DEBUG: front_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    u_before = np.copy(u)
    expected_u[0, [6, 7, 8]] = l[0, [8, 5, 2]]
    expected_l[0, [2, 5, 8]] = d[0, [0, 1, 2]]
    expected_d[0, [0, 1, 2]] = r[0, [6, 3, 0]]
    expected_r[0, [0, 3, 6]] = u_before[0, [6, 7, 8]]
    rotated_f = rotate_cw(f)
    if rotated_f is None: print("DEBUG: rotate_cw(f) falhou."); return u, r, f, d, l, b
    expected_f = rotated_f
    if wait_for_move(video, expected_f, f_before, "F", arrows["F"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def front_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l]): print("DEBUG: front_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    u_before = np.copy(u)
    expected_u[0, [6, 7, 8]] = r[0, [0, 3, 6]]
    expected_r[0, [6, 3, 0]] = d[0, [0, 1, 2]]
    expected_d[0, [0, 1, 2]] = l[0, [2, 5, 8]]
    expected_l[0, [8, 5, 2]] = u_before[0, [6, 7, 8]]
    rotated_f = rotate_ccw(f)
    if rotated_f is None: print("DEBUG: rotate_ccw(f) falhou."); return u, r, f, d, l, b
    expected_f = rotated_f
    if wait_for_move(video, expected_f, f_before, "F'", arrows["F'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def back_cw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, d, l, b]): print("DEBUG: back_cw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    u_before = np.copy(u)
    expected_u[0, [0, 1, 2]] = r[0, [2, 5, 8]]
    expected_r[0, [2, 5, 8]] = d[0, [8, 7, 6]]
    expected_d[0, [8, 7, 6]] = l[0, [6, 3, 0]]
    expected_l[0, [6, 3, 0]] = u_before[0, [0, 1, 2]]
    rotated_b = rotate_cw(b)
    if rotated_b is None: print("DEBUG: rotate_cw(b) falhou."); return u, r, f, d, l, b
    expected_b = rotated_b
    if wait_for_move(video, expected_f, f_before, "B", arrows["B"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b

def back_ccw(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, d, l, b]): print("DEBUG: back_ccw recebeu None."); return u, r, f, d, l, b
    f_before = np.copy(f)
    expected_u, expected_r, expected_f, expected_d, expected_l, expected_b = np.copy(u), np.copy(r), np.copy(f), np.copy(d), np.copy(l), np.copy(b)
    u_before = np.copy(u)
    expected_u[0, [0, 1, 2]] = l[0, [6, 3, 0]]
    expected_l[0, [6, 3, 0]] = d[0, [8, 7, 6]]
    expected_d[0, [8, 7, 6]] = r[0, [2, 5, 8]]
    expected_r[0, [2, 5, 8]] = u_before[0, [0, 1, 2]]
    rotated_b = rotate_ccw(b)
    if rotated_b is None: print("DEBUG: rotate_ccw(b) falhou."); return u, r, f, d, l, b
    expected_b = rotated_b
    if wait_for_move(video, expected_f, f_before, "B'", arrows["B'"]):
        return expected_u, expected_r, expected_f, expected_d, expected_l, expected_b
    else: return u, r, f, d, l, b


# Funções para virar o cubo inteiro (simplificado)
def turn_cube_Y(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l, b]): print("DEBUG: turn_Y recebeu None."); return u, r, f, d, l, b
    print("Vire o cubo todo para a DIREITA (Y)")
    new_f, new_r, new_b, new_l = l, f, r, b
    rotated_u = rotate_cw(u)
    rotated_d = rotate_ccw(d)
    if rotated_u is None or rotated_d is None: print("DEBUG: rotação U/D falhou em turn_Y."); return u, r, f, d, l, b
    new_u, new_d = rotated_u, rotated_d
    print("Mostre a nova FACE FRONTAL (que era a Esquerda)")
    if wait_for_move(video, new_f, l, "VIRAR P/ DIREITA: Mostre a nova face F", arrows["TURN_R"]):
        return new_u, new_r, new_f, new_d, new_l, new_b
    else: return u, r, f, d, l, b


def turn_cube_Y_prime(video, u, r, f, d, l, b, *args):
    if any(face is None for face in [u, r, f, d, l, b]): print("DEBUG: turn_Y' recebeu None."); return u, r, f, d, l, b
    print("Vire o cubo todo para a ESQUERDA (Y')")
    new_f, new_l, new_b, new_r = r, f, l, b
    rotated_u = rotate_ccw(u)
    rotated_d = rotate_cw(d)
    if rotated_u is None or rotated_d is None: print("DEBUG: rotação U/D falhou em turn_Y'."); return u, r, f, d, l, b
    new_u, new_d = rotated_u, rotated_d
    print("Mostre a nova FACE FRONTAL (que era a Direita)")
    if wait_for_move(video, new_f, r, "VIRAR P/ ESQUERDA: Mostre a nova face F", arrows["TURN_L"]):
        return new_u, new_r, new_f, new_d, new_l, new_b
    else: return u, r, f, d, l, b

# Mapeamento de strings de movimento para funções
move_functions = {
    "R": right_cw, "R'": right_ccw,
    "L": left_cw,  "L'": left_ccw,
    "U": up_cw,    "U'": up_ccw,
    "D": down_cw,  "D'": down_ccw,
    "F": front_cw, "F'": front_ccw,
    "B": back_cw,  "B'": back_ccw,
}
# Adiciona movimentos duplos (X2) dinamicamente
for move in list(move_functions.keys()):
    if "'" not in move:
        func = move_functions[move]
        move_functions[move + "2"] = lambda v, u, r, f, d, l, b, *a, _func=func: \
                                     _func(v, *_func(v, u, r, f, d, l, b, *a), *a)

print("DEBUG: Funções interativas definidas.") # DEBUG 5 (Fim)


# --- 5. Função Principal ---
def main():
    global num_to_kociemba_letter, kociemba_letter_to_num

    print("DEBUG: Entrando na função main()") # DEBUG 6
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("DEBUG: Tentando webcam 1...")
        video = cv2.VideoCapture(1)
        if not video.isOpened():
              print("Erro fatal: Nenhuma webcam encontrada.")
              return
    print("DEBUG: Webcam aberta com sucesso.") # DEBUG 7

    # Guarda o estado das 6 faces (agora como LISTAS DE LETRAS)
    cube_state_letters = {face: None for face in faces_order} # MODIFICADO
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
    print("DEBUG: Janela 'Resolvendo...' criada.") # DEBUG 7.1

    print("--- Solver Interativo de Cubo Mágico ---")
    print("Instruções de Scan:")
    print("1. Certifique-se que o arquivo 'calibrated_colors.py' existe e está correto!")
    print("2. Mostre cada face do cubo alinhada com a grade.")
    print("3. Mantenha a face estável por ~1 segundo para leitura.")
    print("4. Siga as instruções no topo da tela.")
    print("5. Pressione [Q] para sair a qualquer momento.")

    while True:
        # print("DEBUG: Inicio do loop while.") # DEBUG 8 (Muito verbose)
        is_ok, frame = video.read()
        # print(f"DEBUG: video.read() retornou is_ok={is_ok}") # DEBUG 9 (Muito verbose)
        if not is_ok:
            print("DEBUG: Falha ao ler frame. Tentando de novo...")
            time.sleep(1); is_ok, frame = video.read()
            if not is_ok: print("DEBUG: Falha ao ler frame novamente. Saindo."); break
        if frame is None: print("DEBUG: Frame Nulo."); time.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        frame_with_grid = draw_preview_grid(frame.copy(), grid_centers)
        if frame_with_grid is None: print("DEBUG: draw_preview_grid falhou."); continue

        # --- Fase de Scan ---
        if not scan_complete:
             if current_face_index >= len(faces_order):
                 print("DEBUG: Erro - Índice de face inválido. Reiniciando scan.")
                 current_face_index = 0; scan_complete = False; cube_state_letters = {f: None for f in faces_order}; detected_faces_buffer = []

             face_code_to_scan = faces_order[current_face_index]
             face_name = face_names_pt.get(face_code_to_scan, "Desconhecida")
             text = f"Scan ({current_face_index+1}/6): Mostre {face_name} ({face_code_to_scan})"
             cv2.putText(frame_with_grid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

             # Tenta detectar a face atual (apenas letras por enquanto)
             detected_letters = []
             valid_detection = True
             try: hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
             except cv2.error as e: print(f"DEBUG: Erro HSV: {e}."); valid_detection = False

             if valid_detection:
                 for (x, y) in grid_centers:
                     if y >= hsv_frame.shape[0] or x >= hsv_frame.shape[1] or y < 0 or x < 0: valid_detection = False; break
                     pixel_hsv = hsv_frame[y, x]
                     color_letter = get_color_name(pixel_hsv)
                     if color_letter == '?': valid_detection = False; break
                     detected_letters.append(color_letter)

             # --- Adiciona Texto de Debug Visual (MODIFICADO) ---
             if valid_detection and len(detected_letters) == 9:
                 # Desenha as letras detectadas ANTES de checar estabilidade
                 for i, (x, y) in enumerate(grid_centers):
                     color_letter_detected = detected_letters[i]
                     text_pos = (x - 10, y + 5)
                     cv2.putText(frame_with_grid, color_letter_detected, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                     cv2.putText(frame_with_grid, color_letter_detected, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                 # --- Continua com a lógica de estabilidade e verificação do centro ---
                 detected_faces_buffer.append(detected_letters)
                 if len(detected_faces_buffer) > 10:
                     detected_faces_buffer.pop(0)

                 # Verifica estabilidade (3 detecções)
                 if len(detected_faces_buffer) >= 3 and all(f == detected_letters for f in detected_faces_buffer[-3:]):
                     center_color_letter = detected_letters[4]
                     if center_color_letter == face_code_to_scan:
                         print(f"Face {face_code_to_scan} escaneada (letras): {detected_letters}")

                         # Salva a LISTA DE LETRAS detectada
                         cube_state_letters[face_code_to_scan] = detected_letters # MODIFICADO

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
                              # --- Bloco try...except CORRIGIDO para mapeamento e geração da string ---
                              try:
                                  # --- 1. Criar Mapeamento Final ---
                                  print("DEBUG: Criando mapeamento baseado nos centros...")
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
                                  if len(kociemba_letter_to_num) != 6: raise ValueError("Mapeamento incompleto.")
                                  print("DEBUG: Mapeamento Cor -> Posição Kociemba:", map_letra_para_posicao)

                                  # --- 2. Converter Estado para Números ---
                                  print("DEBUG: Convertendo estado (letras) para estado (números)...")
                                  for face_code in faces_order:
                                       letras = cube_state_letters[face_code]
                                       numeros = [kociemba_letter_to_num[l] for l in letras]
                                       cube_state_num[face_code] = np.array([numeros])
                                       print(f"DEBUG: Estado Numérico {face_code}: {cube_state_num[face_code]}")

                                  # --- 3. Gerar String Kociemba ---
                                  print("DEBUG: Gerando string Kociemba (Lógica Corrigida)...")
                                  kociemba_string = ""
                                  for face_code_posicao in faces_order:
                                      letra_centro_da_posicao = map_posicao_para_letra.get(face_code_posicao)
                                      if letra_centro_da_posicao is None: raise ValueError(f"Cor central não encontrada para posição {face_code_posicao}")
                                      letras_da_face_escaneada = cube_state_letters.get(letra_centro_da_posicao)
                                      if letras_da_face_escaneada is None: raise ValueError(f"Estado não encontrado para cor central {letra_centro_da_posicao}")
                                      for letra_peca in letras_da_face_escaneada:
                                          posicao_kociemba_da_peca = map_letra_para_posicao.get(letra_peca)
                                          if posicao_kociemba_da_peca is None: raise ValueError(f"Cor '{letra_peca}' sem mapeamento de posição!")
                                          kociemba_string += posicao_kociemba_da_peca
                                  kociemba_string_generated = kociemba_string
                                  print(f"String Kociemba Final: {kociemba_string}")
                                  if len(kociemba_string) != 54: raise ValueError(f"String Kociemba com tamanho incorreto: {len(kociemba_string)}")

                                  # --- 4. Chamar Kociemba ---
                                  print("DEBUG: Chamando kociemba.solve...")
                                  solution = kociemba.solve(kociemba_string)
                                  solution_moves = solution.split()
                                  print(f"Solucao ({len(solution_moves)} mov): {solution}")
                                  current_move_index = 0

                              except ValueError as ve:
                                    print(f"DEBUG: Erro de Valor ao Mapear/Gerar Solução: {ve}")
                                    scan_complete = False; current_face_index = 0; cube_state_letters = {f: None for f in faces_order}; cube_state_num = {f: None for f in faces_order}; num_to_kociemba_letter = {}; kociemba_letter_to_num = {}; detected_faces_buffer = []
                                    time.sleep(4)
                              except Exception as e:
                                  print(f"DEBUG: Erro inesperado no Mapeamento/Solução: {e}")
                                  scan_complete = False; current_face_index = 0; cube_state_letters = {f: None for f in faces_order}; cube_state_num = {f: None for f in faces_order}; num_to_kociemba_letter = {}; kociemba_letter_to_num = {}; detected_faces_buffer = []
                                  time.sleep(4)
                              # --- Fim do Bloco try...except CORRIGIDO ---

                     else: # Centro errado
                         cv2.putText(frame_with_grid, f"Centro errado! Mostre {face_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                         detected_faces_buffer = []
                 else: # Não estável
                     needed = 3 - len(detected_faces_buffer) if len(detected_faces_buffer) < 3 else 0
                     if not all(f == detected_letters for f in detected_faces_buffer[-3:]) : needed = 3
                     if needed > 0 :
                          cv2.putText(frame_with_grid, f"Mantenha estavel... ({needed})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

             else: # Detecção inválida (letra '?' ou número errado de cores)
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
             # --- Fim das Modificações ---

        # --- Fase de Resolução Interativa ---
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

                 print(f"\nDEBUG: Chamando função para movimento {move} ({current_move_index+1}/{len(solution_moves)})")
                 new_u, new_r, new_f, new_d, new_l, new_b = move_func(
                     video, current_u, current_r, current_f, current_d, current_l, current_b,
                     kociemba_letter_to_num,
                     grid_centers[0][0] - 70, grid_centers[0][1] - 70, 70, 70
                 )

                 # Verifica interrupção
                 interrupted = (np.array_equal(new_u, current_u) and np.array_equal(new_r, current_r) and
                                np.array_equal(new_f, current_f) and np.array_equal(new_d, current_d) and
                                np.array_equal(new_l, current_l) and np.array_equal(new_b, current_b))

                 if interrupted and move_func != wait_for_move:
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
        else: # Fim da solução
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
    print("DEBUG: Saindo do loop while.") # DEBUG 13
    video.release()
    cv2.destroyAllWindows()
    print("DEBUG: Recursos liberados.") # DEBUG 14
    for _ in range(5): cv2.waitKey(1)
    print("--- Fim do Script ---") # DEBUG 15

if __name__ == "__main__":
    main()