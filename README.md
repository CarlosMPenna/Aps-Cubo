# 🤖 Solver Interativo de Cubo Mágico

Este projeto é um solucionador de Cubo Mágico que utiliza Visão Computacional (OpenCV) para ler o estado de um cubo físico através de uma webcam.

O grande diferencial deste solver é que ele não apenas _calcula_ a solução, mas **guia-o interativamente** em cada etapa. O programa exibe setas no ecrã para cada movimento (R, U, F', etc.) e **espera pela confirmação visual** de que executou o movimento antes de avançar para o próximo.

## ✨ Funcionalidades Principais

- **Leitura por Webcam:** Detecta as 6 faces do cubo usando a câmara.
- **Calibração de Cor:** Inclui uma ferramenta de calibração (`calibrador.py`) para ajustar os ranges de cor HSV, permitindo que o programa funcione em diferentes condições de iluminação e com diferentes câmaras.
- **Guia Interativo Passo a Passo:** Após o scan, o programa guia o utilizador com setas visuais para cada movimento da solução.
- **Verificação de Movimento:** O programa "assiste" enquanto faz o movimento e só avança para o próximo passo quando o movimento correto é detetado.
- **Lógica de Rotação (Y-turn):** Possui uma solução inteligente para movimentos na face de Trás (B), pedindo ao utilizador que rode o cubo inteiro (movimento Y) para que o movimento 'B' possa ser executado como um 'R' ou 'L' e ser verificado pela câmara.

## 🛠️ Tecnologias e Bibliotecas

Este projeto é escrito em **Python 3** e utiliza as seguintes bibliotecas:

- **OpenCV (`opencv-python`):** Para captura de vídeo, processamento de imagem e desenho das setas/grelhas no ecrã.
- **Kociemba (`kociemba`):** A biblioteca que fornece o algoritmo para calcular a solução mais curta para o cubo.
- **NumPy (`numpy`):** Usada para manipulação eficiente de arrays e matrizes de imagem.
- **SciPy (`scipy`):** Utilizada para estabilizar as deteções.

## 📂 Estrutura dos Ficheiros

- `solver_interativo_setas.py`: O programa principal do solucionador interativo.
- `calibrador.py`: A ferramenta que deve ser executada primeiro para calibrar as cores.
- `calibrated_colors.py`: **(Ficheiro Gerado)** Este ficheiro é criado pelo calibrador e armazena os valores de cor HSV que o solver principal irá usar.

## 🚀 Como Executar o Projeto

Siga estes passos para configurar e executar o projeto no seu computador.

### 1. Instalação das Dependências

Primeiro, precisa de instalar todas as bibliotecas necessárias. Pode fazer isso usando `pip`:

```bash
pip install opencv-python numpy kociemba scipy
```

### 2. Passo 1: Calibrar as Cores (MUITO IMPORTANTE!)

Você deve executar o calibrador antes de usar o solver, pois cada webcam e ambiente de iluminação é diferente.

Execute o script de calibração:

```bash
python calibrador_com_save.py
```

- Uma janela com "Trackbars" e a imagem da sua câmara será aberta.

- Mostre uma cor do cubo para a câmara (ex: a face Verde).

- Ajuste os sliders (H_min, H_max, S_min, S_max, V_min, V_max) até que apenas a cor verde apareça em branco na janela "Mascara".

- Pressione a tecla [ s ] no teclado.

- Na consola (terminal), digite o nome da cor (ex: verde) e pressione Enter.

- Repita este processo para TODAS as 6 cores (branco, vermelho, verde, amarelo, laranja, azul).

- Após calibrar as 6 cores, pressione [ q ] para sair.

- Isso criará o ficheiro calibrated_colors.py na pasta do projeto.

### 3. Passo 2: Executar o Solver Interativo

Com as cores calibradas, está pronto para resolver!

Execute o script principal:

```bash
python solver_interativo_setas.py
```

Siga as instruções que aparecem na janela da webcam.

O processo tem duas fases:

**Fase 1: Scan**

- O programa pedirá para mostrar as 6 faces do cubo, uma por uma (Cima, Direita, Frente, etc.).

- Alinhe a face do cubo com a grelha de 9 pontos que aparece no ecrã.

- O programa possui uma validação de centro: ele só aceitará a leitura se a peça do centro for da cor correta para a face que ele pediu (ex: ao pedir a "Face Verde", o centro deve ser verde).

- Mantenha o cubo estável para ele registar a face e passar para a próxima.

**Fase 2: Resolução**

- Após escanear as 6 faces, o programa irá calcular a solução usando o kociemba.

- Ele mostrará o primeiro movimento da solução com uma seta (ex: "R").

- Faça o movimento no seu cubo físico.

- O programa estará a "assistir" e, ao detetar que completou o movimento, mostrará "OK!" e avançará para o próximo movimento.

- Siga as setas até ao fim.

- Atenção: Para movimentos na face de Trás (B, B', B2), o programa pedirá para rodar o cubo inteiro (ex: "VIRE P/ ESQUERDA"). Apenas siga as instruções no ecrã.

Ao final, o programa exibirá a mensagem **"CUBO RESOLVIDO!".**
