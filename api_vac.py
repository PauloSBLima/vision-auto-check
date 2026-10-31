"""
API Full-Stack de Visão Computacional com FastAPI e YOLO (VisionAutoCheck - VAC)
Este módulo implementa uma API RESTful para análise de imagens com suporte a múltiplos 
modelos da família YOLO, benchmark dinâmico e YOLO-World (Vocabulário Aberto).
"""
from fastapi import (
    FastAPI,                                        # Criação da aplicação FastAPI
    UploadFile,                                     # Upload de arquivos (imagens e vídeos)
    File,                                           # Especificação de arquivos para upload
    Form,                                           # Recebimento de dados de formulário (opções de modelo)
    HTTPException                                   # Levantamento de exceções HTTP para respostas de erro
)
from fastapi.responses import FileResponse          # Resposta de arquivo para download (vídeo processado)
from pydantic import BaseModel                      # Modelos de dados para validação e documentação automática da API
from ultralytics import YOLO, YOLOWorld             # Importação das classes de modelo YOLO e YOLO-World para inferência
import cv2                                          # Biblioteca OpenCV para processamento de imagens
import numpy as np                                  # Biblioteca NumPy para manipulação de arrays e imagens
import shutil                                       # Biblioteca para operações de arquivos e diretórios
import os                                           # Biblioteca para operações do sistema de arquivos e caminhos
from typing import Optional, List                   # Tipagem para parâmetros opcionais e listas
from enum import Enum                               # Enumeração para opções de modelos suportados

# ==========================================
# 1. MOLDES DE DADOS (PYDANTIC) EXATOS
# ==========================================
class BoxCoordenadas(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class ObjetoDetectado(BaseModel):
    class_name: str
    confidence: float
    box: BoxCoordenadas

class RespostaAPI(BaseModel):
    status: str
    modelo_utilizado: str
    total_objetos: int
    detections: list[ObjetoDetectado]

# ==========================================
# 2. INICIALIZAÇÃO DA API E MAPA SEGURO DE MODELOS
# ==========================================
app = FastAPI(
    title="API VisionAutoCheck (VAC) - Benchmark Validado Colab", 
    version="2.1.0", 
    description="API RESTful para inspeção automatizada com modelos validados e seguros para o Google Colab."
)

# Mapeamento técnico dos arquivos de modelo
MAPA_MODELOS = {
    #'vac': 'runs/detect/vision_auto_check_bbox/chave_combinada_v4-2/weights/best.pt',
    #'vac': 'runs/detect/vision_auto_check_bbox/completo_v1/weights/best.pt',    
    'vac': 'runs/detect/vision_auto_check_bbox/completo_v2/weights/best.pt',    
    '1': 'yolov8n.pt', '2': 'yolov8s.pt', '3': 'yolov8m.pt', '4': 'yolov8l.pt', '5': 'yolov8x.pt',
    '6': 'yolov8n-oiv7.pt', '7': 'yolov8s-oiv7.pt', '8': 'yolov8m-oiv7.pt', '9': 'yolov8l-oiv7.pt', '10': 'yolov8x-oiv7.pt',
    '11': 'yolo11n.pt', '12': 'yolo11s.pt', '13': 'yolo11m.pt', '14': 'yolo11l.pt', '15': 'yolo11x.pt',
    '16': 'yolov8s-world.pt', '17': 'yolov8m-world.pt', '18': 'yolov8l-world.pt', '19': 'yolov8x-world.pt'
}

# Define o Menu Suspenso (Dropdown) amigável para o Swagger
class OpcoesModeloEnum(str, Enum):
    vac = "vac (Modelo VisionAutoCheck Customizado)"
    # --- YOLOv8 Padrão (COCO) ---
    opt_1 = "1 - yolov8n.pt"
    opt_2 = "2 - yolov8s.pt"
    opt_3 = "3 - yolov8m.pt"
    opt_4 = "4 - yolov8l.pt"
    opt_5 = "5 - yolov8x.pt"
    # --- YOLOv8 OIV7 (Open Images - Ideal para peças e ferramentas complexas) ---
    opt_6 = "6 - yolov8n-oiv7.pt"
    opt_7 = "7 - yolov8s-oiv7.pt"
    opt_8 = "8 - yolov8m-oiv7.pt"
    opt_9 = "9 - yolov8l-oiv7.pt"
    opt_10 = "10 - yolov8x-oiv7.pt"
    # --- YOLO11 (Estado da Arte Ultralytics) ---
    opt_11 = "11 - yolo11n.pt"
    opt_12 = "12 - yolo11s.pt"
    opt_13 = "13 - yolo11m.pt"
    opt_14 = "14 - yolo11l.pt"
    opt_15 = "15 - yolo11x.pt"
    # --- YOLO-World (Vocabulário Aberto por Texto) ---
    opt_16 = "16 - yolov8s-world.pt (Vocabulário Aberto)"
    opt_17 = "17 - yolov8m-world.pt (Vocabulário Aberto)"
    opt_18 = "18 - yolov8l-world.pt (Vocabulário Aberto)"
    opt_19 = "19 - yolov8x-world.pt (Vocabulário Aberto)"

# Ajuste na função de carregar dinamicamente para lidar com o texto limpo do Enum
def carregar_modelo_dinamico(escolha_enum: str):
    # Extrai apenas a chave ou o código inicial (ex: "11 - yolo11n.pt" vira "11" ou usa "vac")
    if "vac" in escolha_enum:
        escolha = "vac"
    else:
        escolha = escolha_enum.split(" - ")[0].strip()
        
    nome_modelo = MAPA_MODELOS.get(str(escolha), escolha)
    
    if escolha == 'vac' or 'best.pt' in nome_modelo:
        if not os.path.exists(nome_modelo):
            print(f"✗ [AVISO] Peso customizado não encontrado. Usando yolo11n.pt como fallback.")
            nome_modelo = 'yolo11n.pt'

    print(f"[SISTEMA] Carregando modelo para inferência: {nome_modelo}")
    
    if 'world' in nome_modelo.lower():
        return YOLOWorld(nome_modelo), nome_modelo, True
    else:
        return YOLO(nome_modelo), nome_modelo, False

# ==========================================
# 3. ROTA DE PROCESSAMENTO DE IMAGEM
# ==========================================
@app.post("/detectar-imagem/", response_model=RespostaAPI)
async def analisar_imagem(
    arquivo: UploadFile = File(...),
    modelo_opcao: OpcoesModeloEnum = Form(
        OpcoesModeloEnum.vac, 
        alias="Selecione o modelo para a detecção"
    ),
    classes_world: Optional[str] = Form(
        "combination wrench, wrench, pliers, hammer, alternator, battery, oil filter, radiator, suspension spring, clutch disc, brake pad",
        alias="Classes para o YOLO-World (Apenas se escolher modelo World)"
    )
):
    if not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem válida.")
    
    imagem_bytes = await arquivo.read()
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Carrega o modelo escolhido no dropdown
    modelo, nome_modelo_ativo, usa_world = carregar_modelo_dinamico(modelo_opcao)

    args_predicao = {
        "source": img_cv2,
        "conf": 0.30,
        "iou": 0.45,
        "verbose": False
    }

    # SEGURANÇA: Só aplica set_classes se o modelo selecionado for da família YOLO-World!
    if usa_world:
        lista_alvos = [c.strip() for c in classes_world.split(",") if c.strip()]
        print(f"[SYSTEM] Configurando classes de vocabulário aberto para YOLO-World: {lista_alvos}")
        modelo.set_classes(lista_alvos)

    resultados = modelo.predict(**args_predicao)
    resultado = resultados[0]

    deteccoes = []
    if len(resultado.boxes) > 0:
        for box in resultado.boxes:
            cls_id = int(box.cls[0].item())
            
            if cls_id in modelo.names:
                nome_classe = modelo.names[cls_id]
            else:
                nome_classe = f"classe_{cls_id}"
                
            confianca = round(float(box.conf[0].item()), 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            deteccoes.append(
                ObjetoDetectado(
                    class_name=nome_classe,
                    confidence=confianca,
                    box=BoxCoordenadas(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2))
                )
            )

    return RespostaAPI(
        status="success",
        modelo_utilizado=nome_modelo_ativo,
        total_objetos=len(deteccoes),
        detections=deteccoes
    )

# ==========================================
# 4. ROTA DE PROCESSAMENTO DE VÍDEO
# ==========================================
@app.post("/detectar-video/")
async def analisar_video(arquivo: UploadFile = File(...), modelo_opcao: Optional[str] = Form("vac")):
    try:
        diretorio_base = os.path.abspath("temp_video")
        os.makedirs(diretorio_base, exist_ok=True)
        caminho_original = os.path.join(diretorio_base, arquivo.filename)
        
        with open(caminho_original, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
        
        modelo, _, _ = carregar_modelo_dinamico(modelo_opcao)
        
        resultados = modelo.predict(
            source=caminho_original, 
            save=True, 
            project=diretorio_base, 
            name="saida_processada", 
            exist_ok=True,
            conf=0.30,
            verbose=False
        )
        
        nome_arquivo = os.path.basename(caminho_original)
        caminho_processado = os.path.join(diretorio_base, "saida_processada", nome_arquivo)
        
        if not os.path.exists(caminho_processado):
            nome_sem_ext, _ = os.path.splitext(nome_arquivo)
            caminho_processado_avi = os.path.join(diretorio_base, "saida_processada", f"{nome_sem_ext}.avi")
            if os.path.exists(caminho_processado_avi):
                caminho_processado = caminho_processado_avi

        if os.path.exists(caminho_processado):
            return FileResponse(caminho_processado, media_type="video/mp4", filename=f"analisado_{arquivo.filename}")
        else:
            raise HTTPException(status_code=500, detail="O modelo não gerou o arquivo de saída.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))