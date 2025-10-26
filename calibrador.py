import cv2
import numpy as np
import os
import time

def nada(x):
    """Função 'dummy' para os trackbars"""
    pass

# --- Mapeamento Nome -> Letra Kociemba ---
name_to_kociemba = {
    "branco": "U", "vermelho": "R", "verde": "F",
    "amarelo": "D", "laranja": "L", "azul": "B"
}
kociemba_to_name = {v: k for k, v in name_to_kociemba.items()}

# Dicionário para guardar os ranges calibrados
saved_ranges = {}
output_filename = "calibrated_colors.py"
last_save_message = ""
last_save_time = 0

# --- Carregar valores salvos anteriormente ---
if os.path.exists(output_filename):
    try:
        from calibrated_colors import calibrated_values
        saved_ranges = calibrated_values
        print(f"Valores carregados de '{output_filename}':")
        for k, v in saved_ranges.items():
             print(f"- {kociemba_to_name.get(k, k)} ({k}): Min{v[0]}, Max{v[1]}")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar '{output_filename}'. Será sobrescrito. Erro: {e}")
        saved_ranges = {}

# Tenta iniciar a webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Webcam 0 indisponível. Tentando índice 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erro fatal: Nenhuma webcam encontrada.")
        exit()

# Cria janelas
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 300)
cv2.namedWindow("Original (Mostre a cor aqui)")
cv2.namedWindow("Mascara (Isole a cor em branco)")
cv2.namedWindow("Resultado (Cor isolada)") # <-- Janela reativada

# Cria trackbars
cv2.createTrackbar("H_min", "Trackbars", 0, 179, nada)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nada)
cv2.createTrackbar("S_min", "Trackbars", 0, 255, nada)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nada)
cv2.createTrackbar("V_min", "Trackbars", 0, 255, nada)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nada)

# --- Define valores iniciais dos trackbars (se já foram salvos) ---
if saved_ranges:
    try:
        first_key = next(iter(saved_ranges))
        if len(saved_ranges[first_key]) == 2 and len(saved_ranges[first_key][0]) == 3 and len(saved_ranges[first_key][1]) == 3:
            cv2.setTrackbarPos("H_min", "Trackbars", saved_ranges[first_key][0][0])
            cv2.setTrackbarPos("S_min", "Trackbars", saved_ranges[first_key][0][1])
            cv2.setTrackbarPos("V_min", "Trackbars", saved_ranges[first_key][0][2])
            cv2.setTrackbarPos("H_max", "Trackbars", saved_ranges[first_key][1][0])
            cv2.setTrackbarPos("S_max", "Trackbars", saved_ranges[first_key][1][1])
            cv2.setTrackbarPos("V_max", "Trackbars", saved_ranges[first_key][1][2])
    except Exception as e:
        print(f"Aviso: Não foi possível definir os valores iniciais dos trackbars. Erro: {e}")


print("\n--- Calibrador de Cor HSV (com Save) ---")
print("1. Ajuste os controles para isolar uma cor na janela 'Mascara'.")
print("2. Pressione a tecla [s] para salvar.")
print("3. Digite o NOME da cor no CONSOLE (branco, vermelho, verde, amarelo, laranja, azul) e pressione Enter.")
print("4. Repita para as 6 cores.")
print("5. Pressione [q] para sair e SALVAR TUDO no arquivo 'calibrated_colors.py'.")
print("\nCores já salvas nesta sessão:")
for k, v in saved_ranges.items():
    print(f"- {kociemba_to_name.get(k, k)} ({k})")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pega valores dos trackbars
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    # Cria máscara
    limite_inferior = np.array([h_min, s_min, v_min])
    limite_superior = np.array([h_max, s_max, v_max])
    if h_min > h_max:
        mask1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max]))
        mask2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max]))
        mascara = mask1 | mask2
    else:
        mascara = cv2.inRange(hsv, limite_inferior, limite_superior)

    resultado = cv2.bitwise_and(frame, frame, mask=mascara) # <-- Linha reativada

    # Mostra mensagem de save
    if time.time() - last_save_time < 3:
        cv2.putText(frame, last_save_message, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostra janelas
    cv2.imshow("Original (Mostre a cor aqui)", frame)
    cv2.imshow("Mascara (Isole a cor em branco)", mascara)
    cv2.imshow("Resultado (Cor isolada)", resultado) # <-- Linha reativada

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s'):
        print("\n-------------------------")
        color_name_input = input("Digite o nome da cor (branco, vermelho, etc.): ").lower().strip()
        print("-------------------------")

        if color_name_input in name_to_kociemba:
            kociemba_letter = name_to_kociemba[color_name_input]
            current_range = ([h_min, s_min, v_min], [h_max, s_max, v_max])
            saved_ranges[kociemba_letter] = current_range

            last_save_message = f"'{color_name_input.capitalize()}' ({kociemba_letter}) salvo!"
            print(last_save_message)
            print(f"  Valores: Min{current_range[0]}, Max{current_range[1]}")
            last_save_time = time.time()

            print("\nCores salvas até agora:")
            for k, v in saved_ranges.items():
                print(f"- {kociemba_to_name.get(k, k)} ({k})")
        else:
            last_save_message = f"Nome '{color_name_input}' invalido!"
            print(f"Erro: Nome de cor inválido. Use: {list(name_to_kociemba.keys())}")
            last_save_time = time.time()

# --- Fim do loop - Salvar no arquivo ---
cap.release()
cv2.destroyAllWindows()

if saved_ranges:
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("# Arquivo gerado automaticamente pelo calibrador_com_save.py\n")
            f.write("# Contem os ranges HSV calibrados para as cores do cubo\n\n")
            f.write("calibrated_values = {\n")
            num_entries = len(saved_ranges)
            count = 0
            for key, value in saved_ranges.items():
                count += 1
                f.write(f"    '{key}': ([{value[0][0]}, {value[0][1]}, {value[0][2]}], [{value[1][0]}, {value[1][1]}, {value[1][2]}]){',' if count < num_entries else ''}\n")
            f.write("}\n")
        print(f"\nValores salvos com sucesso no arquivo: '{output_filename}'")
        print("\nConteúdo salvo:")
        with open(output_filename, 'r', encoding='utf-8') as f:
             print(f.read())
    except Exception as e:
        print(f"\nErro ao salvar o arquivo '{output_filename}': {e}")
else:
    print("\nNenhum valor foi salvo.")