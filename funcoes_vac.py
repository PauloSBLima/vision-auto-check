"""
#=========================================================
# 👓 01. SOBRE
#=========================================================

MÓDULO DE FUNÇÕES PARA O PROJETO (Vínculo)
===========================================================================================
PROJETO: 🚗 VisionAutoCheck (VAC) — Sistema Inteligente de Detecção para Oficina Automotiva
===========================================================================================
Aqui você encontrará a estrutura das funções preparadas 
para serem utilizadas no módulo principal (vac.ipynb).
Utilizamos as bibliotecas listadas abaixo.
"""
import sys                          # Biblioteca para manipulação de variáveis e funções do sistema operacional (como sys.exit())
import os                           # Biblioteca para manipulação de arquivos e diretórios
from importlib.metadata import (    # Biblioteca para obter informações sobre pacotes instalados, como
    version,                        # Função para obter a versão de um pacote instalado
    PackageNotFoundError            # Exceção lançada quando um pacote não é encontrado
)

import ast                          # Biblioteca para converter strings em objetos Python (útil para ler listas e dicionários de arquivos .txt)
import json                         # Biblioteca para manipulação de arquivos JSON (JavaScript Object Notation) 
from pathlib import Path            # Biblioteca para manipulação de caminhos de arquivos e diretórios de forma independente do sistema operacional
import cv2                          # Biblioteca OpenCV para processamento de imagens e visão computacional
from ultralytics import YOLO        # Biblioteca para trabalhar com modelos YOLO (You Only Look Once) de detecção de objetos

import random                       # Biblioteca para gerar números aleatórios (útil para selecionar amostras aleatórias de imagens)
import matplotlib.pyplot as plt     # Biblioteca para criação de gráficos e visualização de imagens (útil para exibir imagens com bounding boxes)
import yaml                         # Biblioteca para manipulação de arquivos YAML (útil para ler e escrever configurações de projetos)
import numpy as np                  # Biblioteca para manipulação de arrays e operações matemáticas (útil para cálculos com coordenadas de bounding boxes)
import shutil                       # Biblioteca para operações de alto nível em arquivos e diretórios (como copiar arquivos)

from IPython.display import (
    Image, 
    display
)                                   # Biblioteca para exibir imagens diretamente em notebooks Jupyter

from ultralytics import (
    YOLO,                           # Biblioteca para trabalhar com modelos YOLO (You Only Look Once) de detecção de objetos
    YOLOWorld                       # Biblioteca para trabalhar com modelos YOLO-World, que permitem detecção de objetos baseada em texto (prompt text)
)

from roboflow import Roboflow       # Biblioteca para integração com a plataforma Roboflow, que facilita o gerenciamento de datasets e modelos de visão computacional

"""
#=========================================================
# 💡 02. LEMBRETES
#=========================================================

01. Type Hints (dicas de tipo) para melhorar a legibilidade e manutenção do código.

    A explicitação dos tipos de variáveis e parâmetros em funções no Python é chamada de Type Hints (dicas de tipo) ou
    Type Annotations (anotações de tipo). Essa prática (introduzida pelo PEP 484) permite definir o tipo esperado para
    cada argumento da função e o tipo do dado que ela vai retornar

02. Cores:

    Estilo/Cor	Código
    Negrito	    \033[1m
    Vermelho	\033[31m
    Verde	    \033[32m
    Amarelo	    \033[33m
    Azul	    \033[34m
    Reset	    \033[0m

    Ex: ERRO = "\033[1;31m" 
        RESET = "\033[0m"
        print(f"{ERRO}ERRO CRÍTICO:{RESET} O arquivo não foi encontrado.")

 """
#=========================================================
# 💡 03. VARIÁVEIS DE AMBIENTE
#=========================================================

TERMOS_NULOS: tuple = (
    '', 'null', 'none', 'undefined', 'nan', 'nulo', 
    'vazio', 'na', 'n/a', '-', '--', '.', '?'
)

NEGRITO = "\033[1m"
ERRO = "\033[1;31m"
RESET = "\033[0m"
VERMELHO = "\033[31m"
VERDE = "\033[32m"
AMARELO = "\033[33m"
AZUL = "\033[34m"

# Paletas de cores para uso com os gráficos
g_amarelo = "#FFD700"
g_azul = "#1E90FF"
g_azul_claro = "#ADD8E6"
g_branco = "#FFFFFF"
g_cinza = "#808080"
g_laranja = "#FFA500"
g_preto = "#000000"
g_rosa = "#FF69B4"
g_roxo = "#800080"
g_salmao = "#FA8072"
g_verde = "#32CD32"
g_verde_claro = "#32CD32"
g_vermelho = "#FF0000"
g_vermelho_claro = "#E57373"

# Padrão brasileiro: ponto no milhar
"""
Força o arredondamento do número para zero casas decimais (um número inteiro), mesmo que o número 
original seja um float com centavos.

Se você passar 1250.75, o resultado será "1.251"
"""
formato_ptBR_0 = lambda x: f"{x:,.0f}".replace(",", ".")
formato_ptBR_2 = lambda x: f"{x:,.2f}".replace(".", ",") # Alterada , e .

# Mapeamento de módulos cujo nome de importação difere do nome no PyPI
MAPEAMENTO_EXCECOES = {
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'yaml': 'PyYAML',
    'bs4': 'beautifulsoup4'
}

#============================================================================================================================
#=========================================================
# 📂 02. FUNÇÕES PARA YOLO
#=========================================================

def gerar_requirements_enxuto(diretorio="."):
    modulos_encontrados = set()
    
    # Mapeamento oficial de pacotes do PyPI que importam para o VAC
    mapa_pypi = {
        "cv2": "opencv-python",
        "yaml": "PyYAML",
        "PIL": "Pillow",
        "ultralytics": "ultralytics",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "questionary": "questionary",
        "pandas": "pandas",
        "numpy": "numpy",
        "gdown": "gdown",
        "requests": "requests",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "joblib": "joblib",
    }
    
    # Whitelist estrita focada exclusivamente em Visão Computacional, YOLO e API
    pacotes_permitidos = set(mapa_pypi.values())
    
    # 1. Varredura física nos arquivos .py da pasta
    for root, dirs, files in os.walk(diretorio):
        if any(p.startswith('.') or p in ['venv', '__pycache__', 'dataset', 'runs', 'models'] for p in root.split(os.sep)):
            continue
            
        for file in files:
            caminho_arquivo = os.path.join(root, file)
            try:
                if file.endswith('.py'):
                    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                        for linha in f:
                            linha_limpa = linha.strip()
                            if linha_limpa.startswith("import ") or linha_limpa.startswith("from "):
                                if linha_limpa.startswith("from "):
                                    partes = linha_limpa.split()
                                    if len(partes) > 1:
                                        modulos_encontrados.add(partes[1])
                                elif linha_limpa.startswith("import "):
                                    partes = linha_limpa.replace("import ", "").split(",")
                                    for p in partes:
                                        modulos_encontrados.add(p.strip().split()[0])
            except Exception:
                pass

    # 2. Varredura no ambiente atual, mas filtrando estritamente pela nossa whitelist de pacotes externos
    for mod_name in list(sys.modules.keys()):
        base_mod = mod_name.split('.')[0]
        # Traduz se estiver no mapa
        pacote_mapeado = mapa_pypi.get(base_mod, base_mod)
        if pacote_mapeado in pacotes_permitidos or base_mod in mapa_pypi:
            modulos_encontrados.add(pacote_mapeado)

    # Converte e filtra garantindo apenas o que é da whitelist
    requisitos = set()
    for mod in modulos_encontrados:
        pacote = mapa_pypi.get(mod, mod)
        if pacote in pacotes_permitidos:
            requisitos.add(pacote)
        
    # Escreve o arquivo requirements.txt limpo
    caminho_req = os.path.join(diretorio, "requirements.txt")
    with open(caminho_req, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(requisitos)) + "\n")
        
    print(f"{NEGRITO}{VERDE}✓ Arquivo 'requirements.txt' gerado com sucesso em: {os.path.abspath(caminho_req)}{RESET}")
    print(f"{NEGRITO}{AMARELO}✓ Pacotes limpos incluídos: {sorted(requisitos)}{RESET}")
#============================================================================================================================
def _extrair_imports_de_arquivo(caminho_arquivo: str) -> set:
    """Extrai todos os nomes de módulos importados em um arquivo .py ou .ipynb."""
    modulos = set()
    
    try:
        if caminho_arquivo.endswith('.py'):
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
        elif caminho_arquivo.endswith('.ipynb'):
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            codigo_celulas = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    fonte = cell.get('source', '')
                    if isinstance(fonte, list):
                        fonte = "".join(fonte)
                    # Descarta comandos mágicos (!pip, %matplotlib, etc)
                    linhas_validas = [l for l in fonte.splitlines() if not l.strip().startswith(('!', '%'))]
                    codigo_celulas.append("\n".join(linhas_validas))
            conteudo = "\n".join(codigo_celulas)
        else:
            return modulos

        if not conteudo.strip():
            return modulos

        # Analisa a árvore de sintaxe (AST)
        tree = ast.parse(conteudo)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modulos.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modulos.add(node.module.split('.')[0])
    except Exception as e:
        print(f"{NEGRITO}{AMARELO}⚠️ Aviso ao analisar '{os.path.basename(caminho_arquivo)}': {e}{RESET}")
        
    return modulos
#============================================================================================================================
def gerar_requirements(diretorio_projeto: str = None, caminho_saida: str = "requirements.txt") -> None:
    """
    Varre o diretório do projeto, identifica automaticamente todos os pacotes importados
    em arquivos .py e .ipynb e gera o arquivo requirements.txt com as versões atuais.
    """
    # Se nenhum diretório for passado, usa a pasta onde está este arquivo de funções
    if diretorio_projeto is None or not os.path.exists(diretorio_projeto):
        diretorio_projeto = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    modulos_encontrados = set()
    arquivos_analisados = 0
    
    # 1. Varre os arquivos do diretório do projeto
    for raiz, _, arquivos in os.walk(diretorio_projeto):
        # Ignora pastas de cache, controle de versão e checkpoints
        if any(p in raiz for p in ['__pycache__', '.ipynb_checkpoints', '.git', 'venv', '.venv', 'env']):
            continue
            
        for arq in arquivos:
            if arq.endswith(('.py', '.ipynb')):
                caminho_completo = os.path.join(raiz, arq)
                modulos_extraidos = _extrair_imports_de_arquivo(caminho_completo)
                if modulos_extraidos:
                    modulos_encontrados.update(modulos_extraidos)
                    arquivos_analisados += 1

    print(f"{NEGRITO}{AMARELO}🔍 Encontrados {len(modulos_encontrados)} módulo(s) em {arquivos_analisados} arquivo(s) analisado(s) na pasta: '{diretorio_projeto}'{RESET}.")

    # 2. Identifica pacotes nativos da biblioteca padrão do Python
    modulos_stdlib = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

    # 3. Resolve nomes do PyPI e recupera as versões instaladas
    linhas_requirements = []
    
    for mod in modulos_encontrados:
        # Descarta arquivos locais do projeto (ex: funcoes_cnn) e bibliotecas nativas
        if mod in modulos_stdlib or os.path.exists(os.path.join(diretorio_projeto, f"{mod}.py")):
            continue
            
        nome_pypi = MAPEAMENTO_EXCECOES.get(mod, mod)
        
        try:
            ver = version(nome_pypi)
            linhas_requirements.append(f"{nome_pypi}=={ver}\n")
        except PackageNotFoundError:
            try:
                ver = version(mod)
                linhas_requirements.append(f"{mod}=={ver}\n")
            except PackageNotFoundError:
                pass

    # 4. Define o caminho de saída
    if not os.path.isabs(caminho_saida):
        caminho_saida = os.path.join(diretorio_projeto, caminho_saida)

    # 5. Grava o arquivo de saída
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.writelines(sorted(set(linhas_requirements)))

    print(f"{NEGRITO}{VERDE}✅ Arquivo '{caminho_saida}' gerado com sucesso!{RESET}")
#============================================================================================================================
#=========================================================
# 📂 02. FUNÇÕES PARA YOLO
#=========================================================
#============================================================================================================================
def carregar_mapeamento_classes(caminho_yaml):
    """Lê o arquivo data.yaml e retorna o dicionário/lista de nomes de classes."""
    p = Path(caminho_yaml)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f)
    return dados.get("names", None)
#============================================================================================================================
def gerar_labels_automaticos_dataset(
    pasta_base="dataset_bbox", 
    modelo_base="yolov8s-worldv2.pt", 
    classes=["chave", "chave combinada"], 
    conf_min=0.25
):
    """
    Percorre as pastas de treino e validação do dataset e gera os arquivos .txt
    no formato YOLO. Detecta automaticamente se deve usar YOLO padrão ou YOLO-World.
    """
    print(f"{NEGRITO}{AMARELO}[INFO] Carregando modelo {NEGRITO}{VERDE}{modelo_base}{NEGRITO}{AMARELO} para auto-labeling...{RESET}")
    
    # Identifica dinamicamente se o modelo é YOLO-World pelo nome ou classe
    is_world = "world" in modelo_base.lower()
    
    if is_world:
        modelo = YOLOWorld(modelo_base)
        print(f"{NEGRITO}{AMARELO}[INFO] YOLO-World detectado. Definindo classes de vocabulário: {classes}{RESET}")
        modelo.set_classes(classes)
    else:
        # Se for um modelo YOLO padrão (ex: yolo11n.pt)
        modelo = YOLO(modelo_base)
        print(f"{NEGRITO}{AMARELO}[INFO] YOLO Padrão detectado. Usando classes nativas do modelo.{RESET}")
    
    subpastas = ["train", "val"]
    
    for split in subpastas:
        pasta_imagens = Path(pasta_base) / split / "images"
        pasta_labels = Path(pasta_base) / split / "labels"
        
        if not pasta_imagens.exists():
            print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Pasta de imagens não encontrada: {pasta_imagens}. Pulando...{RESET}")
            continue
            
        pasta_labels.mkdir(parents=True, exist_ok=True)
        
        extensoes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        imagens = [p for p in pasta_imagens.iterdir() if p.suffix in extensoes]
        
        print(f"\n{NEGRITO}{AMARELO}[PROCESSANDO] {split.upper()}: Encontradas {len(imagens)} imagens.{RESET}")
        
        for img_path in imagens:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Não foi possível ler a imagem: {img_path.name}{RESET}")
                continue
                
            resultados = modelo.predict(img, conf=conf_min, verbose=False)
            
            linhas_label = []
            for box in resultados[0].boxes:
                # Coordenadas normalizadas para o formato YOLO (centro_x, centro_y, largura, altura)
                box_norm = box.xywhn[0].cpu().numpy()
                
                # Se for YOLO-World ou YOLO padrão, extrai o ID da classe detectada
                classe_id = int(box.cls[0].item()) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                
                x_cent, y_cent, largura, altura = box_norm
                linhas_label.append(f"{classe_id} {x_cent:.6f} {y_cent:.6f} {largura:.6f} {altura:.6f}\n")
            
            # Salva o arquivo .txt correspondente
            txt_path = pasta_labels / f"{img_path.stem}.txt"
            with open(txt_path, "w") as f:
                f.writelines(linhas_label)
                
        print(f"{NEGRITO}{VERDE}[SUCESSO] ✓ Rótulos gerados e salvos em: {pasta_labels}{RESET}")
#============================================================================================================================
def visualizar_amostras_labels(pasta_base="dataset", split="train", num_amostras=3, salvar_disco=False, caminho_yaml=None):
    """
    Seleciona aleatoriamente imagens e exibe os labels com rótulos compactos (escala 0.3),
    contendo apenas o ID sequencial e o nome da classe.
    """
    pasta_imagens = Path(pasta_base) / split / "images"
    pasta_labels = Path(pasta_base) / split / "labels"
    
    if caminho_yaml is None:
        caminho_yaml = Path(pasta_base) / "data.yaml"
        
    nomes_classes = carregar_mapeamento_classes(caminho_yaml)
    
    pasta_bbox = Path(pasta_base) / split / "bounding_box"
    if salvar_disco:
        pasta_bbox.mkdir(parents=True, exist_ok=True)
    
    extensoes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    imagens = [p for p in pasta_imagens.iterdir() if p.suffix in extensoes] if pasta_imagens.exists() else []
    
    if not imagens:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Nenhuma imagem encontrada em {pasta_imagens}{RESET}")
        return
        
    amostras = random.sample(imagens, min(num_amostras, len(imagens)))
    print(f"{NEGRITO}{AMARELO}[INFO] Exibindo {len(amostras)} amostras aleatórias da pasta '{split}'{RESET}\n")
    
    for img_path in amostras:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w, _ = img.shape
        txt_path = pasta_labels / f"{img_path.stem}.txt"
        
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                linhas = f.readlines()
                
            obj_contador = 0  # Contador para o ID sequencial do objeto
            
            for linha in linhas:
                partes = linha.strip().split()
                if not partes:
                    continue
                
                obj_contador += 1
                classe_id = int(partes[0])
                
                # Identifica o nome da classe
                if nomes_classes and isinstance(nomes_classes, list) and classe_id < len(nomes_classes):
                    nome_rotulo = nomes_classes[classe_id]
                elif nomes_classes and isinstance(nomes_classes, dict) and classe_id in nomes_classes:
                    nome_rotulo = nomes_classes[classe_id]
                else:
                    nome_rotulo = f"cls_{classe_id}"
                
                # Monta o texto compacto apenas com ID e nome da classe
                texto_label = f"{obj_contador:02d} {nome_rotulo}"
                
                xmin, ymin = 0, 0
                
                # CASO 1: Bounding Box Tradicional (5 elementos)
                if len(partes) == 5:
                    x_cent_norm = float(partes[1])
                    y_cent_norm = float(partes[2])
                    larg_norm = float(partes[3])
                    alt_norm = float(partes[4])
                    
                    x_cent = x_cent_norm * w
                    y_cent = y_cent_norm * h
                    largura = larg_norm * w
                    altura = alt_norm * h
                    
                    xmin = int(x_cent - (largura / 2))
                    ymin = int(y_cent - (altura / 2))
                    xmax = int(x_cent + (largura / 2))
                    ymax = int(y_cent + (altura / 2))
                    
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                
                # CASO 2: Segmentação / Polígono (> 5 elementos)
                elif len(partes) > 5:
                    coordenadas = [float(p) for p in partes[1:]]
                    pontos = []
                    
                    for i in range(0, len(coordenadas), 2):
                        if i + 1 < len(coordenadas):
                            px = int(coordenadas[i] * w)
                            py = int(coordenadas[i+1] * h)
                            pontos.append([px, py])
                    
                    if pontos:
                        pts_np = np.array(pontos, dtype=np.int32)
                        cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
                        
                        xmin = min([p[0] for p in pontos])
                        ymin = min([p[1] for p in pontos])
                """
                # --- RÓTULO COMPACTO (Apenas ID e Classe) ---
                escala_fonte = 0.3
                espessura_fonte = 1
                (w_txt, h_txt), baseline = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, espessura_fonte)
                pos_y = max(h_txt + 4, ymin - 4)

                # Fundo compacto do texto
                cv2.rectangle(img, (xmin, pos_y - h_txt - 3), (xmin + w_txt + 4, pos_y + baseline), (0, 255, 0), -1)
                # Texto em preto sobre fundo verde
                cv2.putText(img, texto_label, (xmin + 2, pos_y - 1), cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, (0, 0, 0), espessura_fonte)
                """               

                # --- DINÂMICA DE ESCALA PROPORCIONAL À IMAGEM ---
                # Usa a altura da imagem para calcular uma proporção equilibrada
                escala_fonte = max(0.4, h / 1500.0)  # Ajusta o tamanho da fonte dinamicamente
                espessura_fonte = max(1, int(h / 800.0)) # Ajusta a grossura da linha dinamicamente
                
                (w_txt, h_txt), baseline = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, espessura_fonte)
                pos_y = max(h_txt + 6, ymin - 6)

                # Fundo compacto do texto proporcional
                cv2.rectangle(img, (xmin, pos_y - h_txt - 4), (xmin + w_txt + 6, pos_y + baseline), (0, 255, 0), -1)
                # Texto em preto sobre fundo verde
                cv2.putText(img, texto_label, (xmin + 3, pos_y - 2), cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, (0, 0, 0), espessura_fonte)
        else:
            print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Arquivo .txt não encontrado para: {img_path.name}{RESET}")
            
        if salvar_disco:
            caminho_saida = pasta_bbox / img_path.name
            cv2.imwrite(str(caminho_saida), img)
            print(f"{NEGRITO}{VERDE}[SALVO] Imagem salva com label em: {caminho_saida}{RESET}\n")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Arquivo: {img_path.name}")
        plt.show()
#============================================================================================================================
def exibir_resultado_auto_labeling(pasta_processadas="dataset/train/auto_labeled", num_amostras=3):
    """
    Seleciona aleatoriamente e exibe imagens já processadas pelo auto-labeling
    para fins de relatório e documentação do projeto.
    """
    pasta_path = Path(pasta_processadas)
    extensoes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    imagens = [p for p in pasta_path.iterdir() if p.suffix in extensoes] if pasta_path.exists() else []
    
    if not imagens:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Nenhuma imagem encontrada no diretório: {pasta_path}. Verifique o caminho.{RESET}")
        return
        
    amostras = random.sample(imagens, min(num_amostras, len(imagens)))
    print(f"{NEGRITO}{AMARELO}[INFO] Exibindo {len(amostras)} imagens auto-rotuladas da pasta '{pasta_path.name}'...{RESET}\n")
    
    for img_path in amostras:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Converte de BGR para RGB para exibição correta no Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Auto-Labeling (yolo11n)- {img_path.name}")
        plt.show()
#============================================================================================================================
def treinar_modelo_yolo(
    caminho_yaml="./dataset/data.yaml", 
    modelo_base="yolo11n.pt", 
    epochs=50,
    patience=15,            # Equivalente a 50% do número de épocas, para early stopping 
    imgsz=640, 
    batch=16, 
    nome_projeto="vision_auto_check", 
    nome_experimento="chave_combinada_v1",
    verbose=True
):
    """
    Inicia o treinamento do modelo YOLOv11 utilizando a estrutura de diretórios e o arquivo data.yaml.
    """
    yaml_path = Path(caminho_yaml)
    if not yaml_path.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ Arquivo de configuração não encontrado em: {yaml_path}{RESET}")
        return None
        
    print(f"{NEGRITO}{VERDE}[INFO] 🚀 Carregando modelo base: {modelo_base}...{RESET}")
    model = YOLO(modelo_base)
    
    print(f"{NEGRITO}{AMARELO}[INFO] 🎯 Iniciando treinamento para o projeto '{nome_projeto}' (Experimento: {nome_experimento})...")
    print(f"{NEGRITO}{AMARELO}[INFO] Configurações: Épocas={epochs} | Imgsz={imgsz} | Batch={batch}\n{RESET}")
    
    # Executa o treinamento oficial do Ultralytics
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=nome_projeto,
        name=nome_experimento,
        verbose=True
    )
    
    print(f"{NEGRITO}{VERDE}[SUCESSO] ✨ Treinamento concluído! Os pesos e métricas foram salvos em '{nome_projeto}/{nome_experimento}'{RESET}")
    return results
#============================================================================================================================
def treinar_modelo_yolo(
    caminho_yaml="./dataset/data.yaml", 
    modelo_base="yolo11n.pt", 
    epochs=50,
    patience=15,            # Equivalente a 50% do número de épocas, para early stopping 
    imgsz=640, 
    batch=16, 
    nome_projeto="vision_auto_check", 
    nome_experimento="chave_combinada_v1",
    verbose=True,
    box=7.5,                # Novo: Ganho da perda de localização (rigor nas caixas)
    cls=1.5,                # Novo: Ganho da perda de classificação (evita misturar classes)
    close_mosaic=10         # Novo: Desativa mosaic nas últimas épocas para estabilizar
):
    """
    Inicia o treinamento do modelo YOLOv11 utilizando a estrutura de diretórios e o arquivo data.yaml.
    Suporta fine-tuning, controle de paciência e hiperparâmetros de ajuste para redução de falsos positivos.
    """
    yaml_path = Path(caminho_yaml)
    if not yaml_path.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ Arquivo de configuração não encontrado em: {yaml_path}{RESET}")
        return None
        
    print(f"{NEGRITO}{VERDE}[INFO] 🚀 Carregando modelo base: {modelo_base}...{RESET}")
    model = YOLO(modelo_base)
    
    print(f"{NEGRITO}{AMARELO}[INFO] 🎯 Iniciando treinamento para o projeto '{nome_projeto}' (Experimento: {nome_experimento})...")
    print(f"{NEGRITO}{AMARELO}[INFO] Configurações: Épocas={epochs} | Imgsz={imgsz} | Batch={batch} | Box={box} | Cls={cls}\n{RESET}")
    
    # Executa o treinamento oficial do Ultralytics com os novos hiperparâmetros de rigor
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=nome_projeto,
        name=nome_experimento,
        verbose=verbose,
        box=box,
        cls=cls,
        close_mosaic=close_mosaic
    )
    
    print(f"{NEGRITO}{VERDE}[SUCESSO] ✨ Treinamento concluído! Os pesos e métricas foram salvos em '{nome_projeto}/{nome_experimento}'{RESET}")
    return results
#============================================================================================================================
def gerar_dataset_bbox(origem="dataset", destino="dataset_bbox"):
    """
    Varre o dataset original e cria um novo (dataset_bbox) contendo apenas
    as amostras que possuem estritamente Bounding Boxes (5 valores por linha)
    para os splits de treino/validação, copia o diretório de teste por completo,
    e copia o arquivo data.yaml da raiz.
    """
    origem_path = Path(origem)
    destino_path = Path(destino)
    
    if not origem_path.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ O diretório de origem '{origem}' não foi encontrado.{RESET}")
        return

    extensoes_img = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    
    total_copiados = 0
    total_ignorados = 0

    print(f"{NEGRITO}{AMARELO}[INFO] 🚀 Iniciando a criação do '{destino}' a partir de '{origem}'...{RESET}")

    # 1. Copia o arquivo data.yaml da raiz, se ele existir
    yaml_origem = origem_path / "data.yaml"
    if yaml_origem.exists():
        destino_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(yaml_origem, destino_path / "data.yaml")
        print(f"{NEGRITO}{VERDE}[INFO] 📄 Arquivo 'data.yaml' copiado com sucesso para a raiz de '{destino}'.{RESET}")
    else:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ⚠ Arquivo 'data.yaml' não encontrado na raiz de '{origem}'.{RESET}")

    # Identifica os splits existentes, tratando o 'test' de forma especial
    splits = [p.name for p in origem_path.iterdir() if p.is_dir() and p.name != "data.yaml"]
    
    for split in splits:
        origem_split_dir = origem_path / split
        destino_split_dir = destino_path / split
        
        # Se o split for 'test', copiamos a pasta inteira de uma vez por todas
        if split == "test":
            print(f"\n{NEGRITO}{AMARELO}[INFO] 📁 Copiando o diretório de teste por completo ('{split}')...{RESET}")
            if destino_split_dir.exists():
                shutil.rmtree(destino_split_dir)
            shutil.copytree(origem_split_dir, destino_split_dir)
            print(f"{NEGRITO}{VERDE}[SUCESSO] ✓ Diretório 'test' copiado integralmente.{RESET}")
            continue

        # Para train e val, aplicamos a filtragem rigorosa de Bounding Boxes
        origem_labels = origem_split_dir / "labels"
        origem_images = origem_split_dir / "images"
        
        if not origem_labels.exists() or not origem_images.exists():
            print(f"{NEGRITO}{VERMELHO}[AVISO] ⚠ O split '{split}' não possui pastas 'labels' ou 'images'. Ignorando.{RESET}")
            continue
            
        destino_labels = destino_split_dir / "labels"
        destino_images = destino_split_dir / "images"
        
        # Cria as pastas de destino se não existirem
        destino_labels.mkdir(parents=True, exist_ok=True)
        destino_images.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{NEGRITO}{AMARELO}[INFO] 📁 Processando split com filtragem: [{split}]...{RESET}")
        
        for txt_path in origem_labels.glob("*.txt"):
            is_pure_bbox = True
            linhas_validas = []
            
            # Valida se o arquivo de texto contém apenas linhas de Bounding Box (5 valores)
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for linha in f:
                        linha_limpa = linha.strip()
                        if not linha_limpa:
                            continue
                        valores = linha_limpa.split()
                        # Se houver mais ou menos de 5 valores, é polígono ou formato inválido
                        if len(valores) != 5:
                            is_pure_bbox = False
                            break
                        linhas_validas.append(linha_limpa)
            except Exception as e:
                print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ Falha ao ler {txt_path.name}: {e}{RESET}")
                continue
            
            # Se não for puramente bounding box ou estiver vazio, ignora
            if not is_pure_bbox or not linhas_validas:
                print(f"{NEGRITO}{g_azul_claro}[IGNORADO] {txt_path.name} contém polígonos (segmentação).{RESET}")
                total_ignorados += 1
                continue
            
            # Procura a imagem correspondente na pasta de imagens do split
            img_encontrada = None
            for ext in extensoes_img:
                potencial_img = origem_images / f"{txt_path.stem}{ext}"
                if potencial_img.exists():
                    img_encontrada = potencial_img
                    break
            
            if img_encontrada is None:
                print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Imagem para {txt_path.name} não foi encontrada.{RESET}")
                continue
            
            # Copia o label e a imagem válida para o novo dataset_bbox
            shutil.copy(txt_path, destino_labels / txt_path.name)
            shutil.copy(img_encontrada, destino_images / img_encontrada.name)
            total_copiados += 1

    print(f"\n==========================================")
    print(f"{NEGRITO}{VERDE}[SUCESSO] ✨ Dataset Bounding Box criado com sucesso!{RESET}")
    print(f"{NEGRITO}{AMARELO}  - Total de arquivos copiados (Puros): {total_copiados}{RESET}")
    print(f"{NEGRITO}{AMARELO}  - Total de arquivos ignorados (Segmentação): {total_ignorados}{RESET}")
    print(f"==========================================")
#============================================================================================================================
def exibir_resultados_experimento(nome_projeto="vision_auto_check_bbox", nome_experimento="chave_combinada_v3", base_dir="runs/detect"):
    """
    Exibe de forma automática as principais imagens de resultado geradas pelo YOLO:
    - Predições no lote de validação (val_batch0_pred.jpg)
    - Gráficos de treinamento e perdas (results.png)
    - Matriz de confusão (confusion_matrix.png)
    """
    caminho_exp = os.path.join(base_dir, nome_projeto, nome_experimento)
    
    if not os.path.exists(caminho_exp):
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ O diretório do experimento não foi encontrado em: {caminho_exp}{RESET}")
        return

    arquivos_interesse = {
        "🔍 Predições no Lote de Validação (val_batch0_pred.jpg)": "val_batch0_pred.jpg",
        "📈 Gráficos de Desempenho e Loss (results.png)": "results.png",
        "📊 Matriz de Confusão (confusion_matrix.png)": "confusion_matrix.png"
    }

    for titulo, arquivo in arquivos_interesse.items():
        caminho_arquivo = os.path.join(caminho_exp, arquivo)
        if os.path.exists(caminho_arquivo):
            print(f"\n{titulo}")
            display(Image(filename=caminho_arquivo))
        else:
            print(f"\n{NEGRITO}{VERMELHO}[INFO] ✗ Arquivo '{arquivo}' não encontrado neste experimento.{RESET}")
#============================================================================================================================
def testar_inferencia_aleatoria(
    caminho_modelo="runs/detect/vision_auto_check_bbox/chave_combinada_v4-2/weights/best.pt",
    pasta_imagens="dataset_chave_combinada/test/images",
    conf_min=0.40,
    pasta_saida_nome="teste_inferencia_aleatoria"
):
    """
    Seleciona aleatoriamente uma imagem de uma pasta de teste, executa a inferência 
    com o modelo treinado e salva o resultado visual de forma robusta.
    Retorna o caminho da imagem gerada.
    """
    print(f"\n{NEGRITO}{AMARELO}[INFO] Carregando modelo: {caminho_modelo}{RESET}")
    if not Path(caminho_modelo).exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ O arquivo de pesos não foi encontrado em: {caminho_modelo}{RESET}")
        return None
        
    model = YOLO(caminho_modelo)
    
    pasta_alvo = Path(pasta_imagens)
    if not pasta_alvo.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ A pasta de imagens não foi encontrada: {pasta_alvo}{RESET}")
        return None
        
    extensoes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    imagens = [p for p in pasta_alvo.iterdir() if p.is_file() and p.suffix in extensoes]
    
    if not imagens:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Nenhuma imagem compatível encontrada em '{pasta_alvo}'.{RESET}")
        return None
        
    # Seleciona uma imagem aleatoriamente
    img_escolhida = random.choice(imagens)
    print(f"{NEGRITO}{AMARELO}[INFO] Imagem selecionada aleatoriamente: {img_escolhida.name}{RESET}")
    
    # Define o diretório absoluto de projeto para evitar duplicações
    projeto_abs = Path.cwd() / "runs" / "detect"
    
    # Executa a predição
    resultados = model.predict(
        source=str(img_escolhida), 
        conf=conf_min, 
        save=True, 
        project=str(projeto_abs), 
        name=pasta_saida_nome, 
        exist_ok=True
    )
    
    # Verifica se houve detecções
    num_deteccoes = len(resultados[0].boxes) if resultados else 0
    if num_deteccoes == 0:
        print(f"{NEGRITO}{AMARELO}[AVISO] ⚠ O modelo não detectou nenhuma chave nesta imagem com conf >= {conf_min}.{RESET}")
    else:
        print(f"{NEGRITO}{VERDE}[INFO] ✓ Detecções realizadas com sucesso: {num_deteccoes} objeto(s) encontrado(s).{RESET}")

    # Limpa o nome base da imagem para busca flexível (removendo parênteses e extensão)
    nome_base = img_escolhida.stem.split('(')[0].strip()
    
    # Localiza o arquivo salvo de forma flexível pelo prefixo do nome base
    dir_esperado = projeto_abs / pasta_saida_nome
    imagem_resultado = None
    
    if dir_esperado.exists():
        candidatos_pasta = [p for p in dir_esperado.iterdir() if nome_base in p.name and p.is_file()]
        if candidatos_pasta:
            imagem_resultado = max(candidatos_pasta, key=lambda p: p.stat().st_mtime)
            
    # Fallback via save_dir oficial do YOLO
    if not imagem_resultado or not imagem_resultado.exists():
        if resultados and hasattr(resultados[0], 'save_dir'):
            candidatos_dir = [p for p in Path(resultados[0].save_dir).iterdir() if nome_base in p.name and p.is_file()]
            if candidatos_dir:
                imagem_resultado = max(candidatos_dir, key=lambda p: p.stat().st_mtime)
                
    # Fallback global de busca recursiva caso necessário
    if not imagem_resultado or not imagem_resultado.exists():
        candidatos_glob = list(projeto_abs.glob(f"**/{img_escolhida.name}"))
        if candidatos_glob:
            imagem_resultado = max(candidatos_glob, key=lambda p: p.stat().st_mtime)

    if imagem_resultado and imagem_resultado.exists():
        print(f"{NEGRITO}{VERDE}[SUCESSO] ✓ Predição concluída! Salva em: {imagem_resultado}{RESET}")
        return str(imagem_resultado)
    else:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ A predição rodou, mas o arquivo de saída não pôde ser mapeado automaticamente.{RESET}")
        return None
#============================================================================================================================
def baixar_dataset_roboflow_versao(
    api_key: str = "kVAO3w4YX0RTpvX5cA23",
    workspace_name: str = "plimas-workspace",
    project_name: str = "alicate-yqke5",
    versao: int = 1,
    destination_path: str = "./dataset_alicate_exp",
    format_type: str = "yolov8"
):
    """
    Baixa uma versão específica do dataset do Roboflow utilizando o SDK oficial.
    """
    try:
        print(f"{NEGRITO}{AMARELO}[ROBOFLOW] Conectando ao workspace '{workspace_name}' e projeto '{project_name}' (Versão {versao})...{RESET}")
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace_name)
        project = workspace.project(project_name)
        
        print(f"{NEGRITO}{AMARELO}[ROBOFLOW] Baixando dataset no formato '{format_type}' para: {destination_path}...{RESET}")
        dataset = project.version(versao).download(format_type, location=destination_path)
        
        print(f"{NEGRITO}{VERDE}[SUCESSO] ✓ Dataset baixado com sucesso em: {destination_path}{RESET}")
        return destination_path
        
    except Exception as e:
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ Falha ao baixar a versão do Roboflow: {e}{RESET}")
        raise e
#============================================================================================================================
def unificar_e_reindexar_classe(
    pasta_origem_dataset: str, 
    pasta_destino_unificada: str, 
    mapeamento_classes_mestre: dict
):
    """
    Lê o dataset baixado de uma classe isolada do Roboflow, descobre qual classe é,
    re-indexa o ID numérico do arquivo .txt para o ID correto do dataset unificado 
    e copia as imagens/labels para a pasta definitiva.
    
    Exemplo de mapeamento_classes_mestre:
    {
        "chave_combinada": 0,
        "alicate": 1,
        "martelo": 2,
        "alternador": 3,
        "bateria": 4,
        "filtro_oleo": 5,
        "radiador": 6,
        "mola_suspensao": 7,
        "disco_embreagem": 8,
        "pastilha_freio": 9
        ...
    }
    """
    origem = Path(pasta_origem_dataset)
    destino = Path(pasta_destino_unificada)
    
    # 1. Descobrir qual classe é este dataset olhando o data.yaml original dele
    arquivo_yaml_origem = origem / "data.yaml"
    if not arquivo_yaml_origem.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ data.yaml não encontrado em {origem}{RESET}")
        return

    with open(arquivo_yaml_origem, 'r') as f:
        dados_yaml = yaml.safe_load(f)
    
    # Pega o nome da classe que está no índice 0 do dataset isolado
    nome_classe_isolada = dados_yaml['names'][0]
    
    if nome_classe_isolada not in mapeamento_classes_mestre:
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ A classe '{nome_classe_isolada}' não está no seu dicionário mestre!{RESET}")
        return
        
    novo_id_correto = mapeamento_classes_mestre[nome_classe_isolada]
    print(f"{NEGRITO}{VERDE}[UNIFICAÇÃO] Processando classe '{nome_classe_isolada}' (ID antigo: 0 -> Novo ID mestre: {novo_id_correto})...{RESET}")

    # 2. Processar subpastas (train, valid, test se existirem)
    for split in ['train', 'valid', 'test']:
        img_origem_dir = origem / split / 'images'
        lbl_origem_dir = origem / split / 'labels'
        
        if not img_origem_dir.exists():
            continue
            
        img_dest_dir = destino / split / 'images'
        lbl_dest_dir = destino / split / 'labels'
        
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar imagens e re-escrever os labels com o ID correto
        for img_path in img_origem_dir.glob('*.*'):
            # Copia imagem
            shutil.copy(img_path, img_dest_dir / img_path.name)
            
            # Procura o label correspondente
            lbl_path = lbl_origem_dir / f"{img_path.stem}.txt"
            lbl_dest_path = lbl_dest_dir / f"{img_path.stem}.txt"
            
            if lbl_path.exists():
                linhas_atualizadas = []
                with open(lbl_path, 'r') as arquivo_txt:
                    for linha in arquivo_txt.readlines():
                        partes = linha.strip().split()
                        if partes:
                            # Substitui o ID antigo (primeiro elemento) pelo novo ID mestre
                            partes[0] = str(novo_id_correto)
                            linhas_atualizadas.append(" ".join(partes))
                
                # Salva o arquivo de label com o ID corrigido na pasta unificada
                with open(lbl_dest_path, 'w') as arquivo_txt_dest:
                    arquivo_txt_dest.write("\n".join(linhas_atualizadas) + "\n")
            else:
                # Caso seja uma imagem sem anotação (fundo), cria um txt vazio ou pula
                open(lbl_dest_path, 'w').close()

    print(f"{NEGRITO}{VERDE}[SUCESSO] ✓ Classe '{nome_classe_isolada}' integrada com sucesso ao dataset unificado!{RESET}")
#============================================================================================================================

#============================================================================================================================

#============================================================================================================================

#============================================================================================================================