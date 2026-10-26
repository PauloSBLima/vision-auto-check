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
def gerar_labels_automaticos_dataset(pasta_base="dataset", modelo_base="yolo11n.pt", conf_min=0.25):
    """
    Percorre as pastas de treino e validação do dataset e gera os arquivos .txt
    no formato YOLO utilizando um modelo base para auto-labeling.
    """
    print(f"{NEGRITO}{AMARELO}[INFO] Carregando modelo base {NEGRITO}{VERDE}{modelo_base}{NEGRITO}{AMARELO} para auto-labeling...{RESET}")
    modelo = YOLO(modelo_base)
    
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
                
                # ID 0 para a primeira classe (chave_combinada)
                classe_id = 0 
                
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
            print(f"{NEGRITO}{VERMELHO}[AVISO] Arquivo .txt não encontrado para: {img_path.name}{RESET}")
            
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
def testar_modelo_yolo(caminho_pesos="vision_auto_check/chave_combinada_v1/weights/best.pt", pasta_teste="dataset/test/images", num_amostras=3, conf_min=0.25):
    """
    Carrega o modelo treinado (best.pt), realiza a inferência em uma seleção aleatória 
    de imagens de teste e exibe visualmente o resultado com as bounding boxes.
    """
    pesos_path = Path(caminho_pesos)
    if not pesos_path.exists():
        print(f"{NEGRITO}{VERMELHO}[ERRO] ✗ Arquivo de pesos não encontrado em: {pesos_path}{RESET}")
        print(f"{NEGRITO}{AMARELO}[INFO] Verifique o caminho exato onde o YOLO salvou o experimento.{RESET}")
        return
        
    print(f"{NEGRITO}{AMARELO}[INFO] 🚀 Carregando modelo treinado de: {pesos_path}{RESET}")
    model = YOLO(str(pesos_path))
    
    pasta_path = Path(pasta_teste)
    extensoes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    imagens = [p for p in pasta_path.iterdir() if p.suffix in extensoes] if pasta_path.exists() else []
    
    if not imagens:
        print(f"{NEGRITO}{VERMELHO}[AVISO] ✗ Nenhuma imagem encontrada em {pasta_path}. Verifique o diretório de teste.{RESET}")
        return
        
    # Seleciona de forma randômica a quantidade definida (padrão = 3)
    amostras = random.sample(imagens, min(num_amostras, len(imagens)))
    print(f"{NEGRITO}{AMARELO}[INFO] 🔍 Executando inferência em {len(amostras)} imagens selecionadas aleatoriamente...{RESET}\n")
    
    for img_path in amostras:
        # Executa a predição na imagem
        resultados = model(str(img_path), conf=conf_min, verbose=False)
        
        # O método .plot() desenha as caixas de detecção, classes e confianças
        img_plotada = resultados[0].plot()
        
        # Converte de BGR para RGB para exibir corretamente no Matplotlib
        img_rgb = cv2.cvtColor(img_plotada, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Inferência (best.pt) - {img_path.name}")
        plt.show()
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
            print(f"{NEGRITO}{VERDE}  [SUCESSO] Diretório 'test' copiado integralmente.{RESET}")
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
                print(f"{NEGRITO}{VERMELHO} ✗ [ERRO] Falha ao ler {txt_path.name}: {e}{RESET}")
                continue
            
            # Se não for puramente bounding box ou estiver vazio, ignora
            if not is_pure_bbox or not linhas_validas:
                print(f"{NEGRITO}{g_azul_claro}  [IGNORADO] {txt_path.name} contém polígonos (segmentação).{RESET}")
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
                print(f"{NEGRITO}{VERMELHO} ✗ [AVISO] Imagem para {txt_path.name} não foi encontrada.{RESET}")
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
        print(f"{NEGRITO}{VERMELHO} ✗[ERRO] O diretório do experimento não foi encontrado em: {caminho_exp}{RESET}")
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
            print(f"\n{NEGRITO}{VERMELHO} ✗[INFO] Arquivo '{arquivo}' não encontrado neste experimento.{RESET}")
#============================================================================================================================

#============================================================================================================================
