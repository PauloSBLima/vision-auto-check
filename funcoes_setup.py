"""
#=========================================================
# 👓 01. SOBRE
#=========================================================

MÓDULO DE FUNÇÕES DE SETUP PARA O PROJETO (Vínculo)
===========================================================================================
PROJETO: 🚗 VisionAutoCheck (VAC) — Sistema Inteligente de Detecção para Oficina Automotiva
===========================================================================================
Aqui você encontrará a estrutura das funções preparadas 
para serem utilizadas no módulo principal (setup_inicil_vac.ipynb).
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

import random                       # Biblioteca para gerar números aleatórios (útil para selecionar amostras aleatórias de imagens)
import matplotlib.pyplot as plt     # Biblioteca para criação de gráficos e visualização de imagens (útil para exibir imagens com bounding boxes)
import yaml                         # Biblioteca para manipulação de arquivos YAML (útil para ler e escrever configurações de projetos)
import numpy as np                  # Biblioteca para manipulação de arrays e operações matemáticas (útil para cálculos com coordenadas de bounding boxes)
import shutil                       # Biblioteca para operações de alto nível em arquivos e diretórios (como copiar arquivos)

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
