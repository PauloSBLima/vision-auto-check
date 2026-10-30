# 🧠 Formação em Machine Learning e Ciência de Dados - SCTI.SC
## Projeto Final Avaliativo do Módulo 2 de VC(ML) - Semana 13
<br>
Bem-vindo(a) ao repositório do trabalho de construção de um pipeline de visão computacional.<br>
Projeto referente ao curso de Lógica de Programação e Machine Learning do SCTI/SENAI.<br>
Este espaço foi criado para centralizar os códigos, notebooks e materiais práticos desenvolvidos durante este projeto.<br>
<hr>

## 🚗 1. VisionAutoCheck (VAC) — Sistema Inteligente de Detecção para Oficina Automotiva

A Indústria 4.0 e a manutenção automotiva moderna exigem soluções inteligentes que vão muito além da simples captura de imagens. No mercado de trabalho atual, o Engenheiro de IA precisa saber construir fluxos completos de ponta a ponta: desde a curadoria de dados e o treinamento de modelos de estado-da-arte até o empacotamento da inteligência em uma API RESTful corporativa.
O VisionAutoCheck (VAC) foi desenvolvido como um microserviço backend de alta performance voltado para o ecossistema de oficinas mecânicas e gestão de peças. O objetivo do sistema é automatizar inicialmente a identificação visual de 10 classes distintas — abrangendo componentes de motor, suspensão, freios, fluidos e ferramentas de bancada — permitindo que outros sistemas da empresa consumam inferências em tempo real.

## 💰 2. Aplicabilidade Comercial e ROI
Do ponto de vista acadêmico, treinar um modelo com 10 classes de peças e ferramentas de oficina serve para dominar o pipeline de IA. No entanto, no mercado real (Indústria 4.0 e pós-venda automotivo), sistemas de visão computacional voltados para a identificação de componentes e ferramentas resolvem dores caríssimas para as empresas.

O `investimento comercial` em um sistema como o **VisionAutoCheck (VAC)** se justifica principalmente por quatro grandes frentes de valor:

**1. Automação de Estoque e Prevenção de Perdas (Smart Inventory)**
- `O Problema:` Oficinas mecânicas lidam com centenas de peças de alto valor (alternadores, baterias, bicos injetores) e ferramentas caras (multímetros, pistolas pneumáticas, chaves específicas). O controle manual de estoque sofre com furtos (shrinkage), peças extraviadas na bancada ou erros de contagem.

- `A Solução Comercial:` Câmeras instaladas nos escaninhos de peças ou na bancada de trabalho conseguem registrar automaticamente o que entra e o que sai. Se um mecânico pega uma bateria ou uma pastilha de freio, o sistema computa o consumo em tempo real, integrando-se diretamente ao **ERP** da oficina para reposição automática.

**2. Auditoria de Montagem e Segurança (Poka-Yoke Digital)**
- `O Problema:` Esquecer ferramentas dentro do compartimento do motor ou instalar uma peça errada (ex: um filtro de óleo incompatível) pode causar falhas catastróficas, prejuízos financeiros e processos judiciais graves.

- `A Solução Comercial:` Antes de fechar o capô do veículo, uma varredura visual automatizada por IA pode verificar se todas as ferramentas utilizadas (alicate, martelo, chave inglesa) foram recolhidas da bancada/motor e se o componente correto foi aplicado. Isso atua como um sistema de Poka-Yoke (à prova de falhas humanas).

**3. Agilidade no Faturamento e Ordens de Serviço (OS)**
- `O Problema:` Muitas vezes, o mecânico precisa anotar manualmente na ordem de serviço quais peças utilizou durante o conserto. Erros de digitação ou o esquecimento de itens menores geram prejuízo para a oficina (peças usadas que não foram cobradas do cliente).

- `A Solução Comercial:` Ao registrar as peças separadas para o reparo em uma "mesa inteligente" monitorada por visão computacional, o sistema gera automaticamente o orçamento e os itens consumidos para a aprovação do cliente, reduzindo o tempo administrativo e zerando perdas de faturamento.

**4. Apoio Operacional e Treinamento (Assistência a Mecânicos Júnior)**
- `O Problema:` Com a alta rotatividade de mão de obra e a complexidade dos carros modernos, mecânicos iniciantes gastam muito tempo procurando ferramentas específicas ou identificando visualmente peças muito parecidas (como diferentes tipos de filtros ou correias).

- `A Solução Comercial:` Uma interface corporativa que ajude a identificar rapidamente a peça correta no catálogo através de uma foto tirada pelo celular ou tablet acelera o diagnóstico e o atendimento ao cliente.

> **Por que o investimento se paga (ROI)❓**
>
> O retorno sobre o investimento (ROI) nesse tipo de tecnologia vem:
> * da redução drástica de erros humanos, 
> * do fim do tempo perdido procurando ferramentas,
> * da eliminação de furtos/desperdícios de estoque e
> * da velocidade no fechamento de ordens de serviço.

## 🎯 3. Objetivos
   * **Do Repositório:** Fornecer um ambiente de consulta prática de um pipeline completo de Visão Computacional e **MLOps** (Machine Learning Operations => Operações de Aprendizado de Máquina). Aqui você encontrará os *scripts* de curadoria, o treinamento de IA baseado em **YOLO**, a API construída em FastAPI e a documentação associada.

   * **Dos Módulos:** O fluxo é dividido em etapas integradas:
      - 1º) **Aquisição e Preparação (Dataset Multiclasse)**: Seleção e estruturação de um dataset contendo 20 classes distintas de uma oficina automotiva/mecânica (peças de motor, suspensão, freios, fluidos e ferramentas de bancada) no padrão compatível com o YOLO.
      - 2º) **Fine-Tuning com IA (Ultralytics YOLO)**: Carregamento de modelos pré-treinados da família YOLO para realizar ajuste fino (*fine-tuning*) focado em detectar com alta precisão e velocidade os 20 tipos de objetos.
      - 3º) **Backend e Endpoint de Inferência (FastAPI)**: Disponibilização de uma API RESTful corporativa contendo um endpoint POST (`/detect`) que recebe uploads de imagens, processa via modelo treinado e retorna um JSON estruturado com as detecções (classes, confianças e *bounding boxes*).

## 👁‍🗨 4. Visão Geral dos Dados (Data Overview)
Origem dos Dados (Input): Dataset multiclasse customizado estruturado para o domínio de oficina automotiva.<br>
   * **Conjunto de Dados / Fontes:** Roboflow Universe, Kaggle e variadas da internet.<br><br>
   * **Diretório de Dados:** https://drive.google.com/drive/folders/14BmA6Q8lnALGOQoqE8rn7uKQm-Mi-nnf?usp=drive_link<br>

Identificação e Escopo da Base de Dados:<br>
- **Classes Mapeadas (10):** 
   * Ferramentas e Equipamentos de Bancada:
	   - 00\. chave_combinada
	   - 01\. alicate
	   - 02\. martelo

   * Componentes do Motor e Elétrica:
	   - 03\. alternador
	   - 04\. bateria
	   - 05\. filtro_oleo
      - 06\. radiador

   * Suspensão, Freios e Rodas: 
	   - 07\. mola_suspensao
	   - 08\. disco_embreagem
	   - 09\. pastilha_freio

- **Formato:** Imagens em resolução variadas estruturadas nos diretórios de treino (`train`) e validação (`val`) com anotações padronizadas.

## ✅ 5. A escolha da versão do YOLO
Elencamos os principais motivos para a seleção de escolha da **versão 11 nano do modelo YOLO (yolo11n)**:<br>
   - <font color="orange">**Compatibilidade com o Hardware (Recursos de VRAM):**</font> A variante nano (n) é a menor, mais leve e otimizada da família YOLO11. Ela consome muito menos memória de vídeo (VRAM) e poder computacional, rodando de forma fluida nas GPUs disponibilizadas no Colab (como a T4) sem estourar limites de memória.
   - <font color="orange">**Prototipagem Rápida e Validação do Pipeline:**</font> Como as épocas de treinamento fecham muito mais rápido em um modelo nano, conseguimos testar se todo o pipeline de dados — desde a estrutura do arquivo data.yaml até as anotações das 10 classes da oficina automotiva — está funcionando perfeitamente sem precisar gastar horas esperando um modelo pesado treinar.   
   - <font color="orange">**Criação de uma Linha de Base (Baseline):**</font> O modelo yolo11n serve como um ponto de partida rápido para estabelecer métricas iniciais de desempenho (como mAP). Com essa base, validamos se o dataset tem qualidade suficiente antes de decidir investir tempo e poder computacional em modelos maiores (como s, m, l ou x).
   - <font color="orange">**Viabilidade de Inferência Rápida:**</font> Para sistemas aplicados em oficinas automotivas, modelos mais leves garantem menor latência na detecção de peças e ferramentas, o que é fundamental caso o sistema precise rodar de forma ágil ou em dispositivos com restrição de hardware no futuro.

   Em suma, começar com o <font color="orange">**yolo11n**</font> garante <u>velocidade, economia de recursos e validação ágil do projeto</u> na fase de laboratório, abrindo margem para escalar a complexidade do modelo apenas quando a estrutura de dados estiver 100% validada na branch de finetuning! 

## 🔄 6. Fases Técnicas Aplicadas (Sprints)

- **Fase 1: Aquisição e Preparação do Dataset (`feature/dataset-prep`):**
   Criação do quadro Kanban/Scrum no Trello, mapeamento das 20 classes, download, limpeza e estruturação do dataset em formato padrão YOLO com o respectivo arquivo de configuração `data.yaml`.

- **Fase 2: Fine-Tuning do Modelo YOLO (`feature/yolo-finetuning`):**
   Configuração e execução do treinamento da arquitetura YOLO (v8/v11) sobre as 20 classes, avaliando métricas de acurácia (mAP) e capacidade de generalização com imagens variadas.

- **Fase 3: Construção do Backend FastAPI (`feature/fastapi-setup`):**
   Implementação do microserviço em FastAPI, estruturando o endpoint POST `/detect` para receber requisições de arquivos de imagem (`UploadFile`) e retornar a resposta estruturada em JSON.

- **Fase 4: Documentação e Versionamento (`feature/readme-docs`):**
   Consolidação da documentação técnica no `README.md`, ajustes no Trello e validação final do fluxo com GitFlow (`main` / `develop` / `feature/`).

- **Fase 5: Auditoria e Gravação (`feature/video-recording`):**
   Gravação do vídeo técnico de apresentação (máximo de 7 minutos) abordando o dataset, demonstração da API via Swagger/Postman, o processo do YOLO, o projeto no Trello e o versionamento do GitHub.

## 🛠️ 7. Tecnologias e Ferramentas Utilizadas
- **Linguagem:** Python 3.10+
- **Framework API:** FastAPI & Uvicorn
- **Visão Computacional e IA:** Ultralytics YOLO (YOLOv8/v11), OpenCV
- **Ambiente:** Google Colab / Visual Studio Code (VS Code)
- **Metodologia Ágil:** Trello (Kanban)
- **Controle de Versão:** Git e GitHub (GitFlow)
- **Bibliotecas Principais:**
   * **- Core e Processamento Numérico**
      - numpy >= 1.23.0
   * **- Visualização Gráfica**
      - matplotlib >= 3.5.0
      - seaborn >= 0.12.0
   * **- IA e Visão Computacional**   
      - ultralytics >= 8.0.0
      - opencv-python >= 4.8.0
   * **- Backend / API**
      - fastapi >= 0.100.0
      - uvicorn >= 0.22.0
      - python-multipart >= 0.0.6
   * **- Ambiente de Execução Interativo (Jupyter/IPython)**
      - ipython >=7.34.0
   * **- Arquivos**
      - gdown >= 5.2.2
   * **- Integração**
      - google >= 3.0.0

## 🧱 8. Estrutura do Projeto
```text
├── VisionAutoCheck/
│   ├── dataset/        	 # Diretório utilizado inicialmente com bounding boxes e segmentação por polígonos
│   ├── dataset_bbox/            # Diretório secundário utilizado somente com bounding boxes
│   ├── dataset_completo/        # Diretório Principal de armazenamento das imagens (M)
│   │   ├── train/     		 # Subdiretório de treino
│   │   │   ├── images/       	 # Subdiretório das imagens a serem treinadas
│   │   │   └── labels/        	 # Subdiretório dos labels das imagens treinadas
│   │   ├── val/     		 # Subdiretório de validação
│   │   │   ├── images/       	 # Subdiretório das imagens a serem validadas
│   │   │   └── labels/        	 # Subdiretório dos labels das imagens validadas
│   │   └── test/     		 # Subdiretório de teste
│   │       ├── images/       	 # Subdiretório das imagens a serem testadas
│   │       └── labels/        	 # Subdiretório dos labels das imagens testadas
│   ├── runs/                    # Treinamentos do modelo (internamente ficam as versões "best.pt")
│   └── weights/        		   # Pesos treinados do YOLO (.pt)
├── .gitignore                   # Relação de arqs que não devem ser levados p/ o  GitHub
├── ARQUITETURA.jpg              # Arquitetura/Diagrama da solução
├── api.ipynb                    # Código principal da API FastAPI
├── api_vac.py                   # Módulo de funções relacionadas com a API
├── dataset-prep_bbox.ipynb      # Módulo de detecção das imagens utilizando apenas bounding boxes
├── dataset-prep_mix.ipynb       # Módulo de detecção das imagens utilizando bounding boxes e segmentação por polígonos
├── funcoes_vac.py               # Módulo de funções deste projeto
├── LICENSE.txt                  # Licença do código
├── README.md                    # Leia-me do projeto
├── requirements.txt             # Relação de bibliotecas e dependências
├── setup_inicial_vac.ipynb      # Módulo de integração com o GitHub - Criação das branches e atualizações
├── train_bbox.ipynb             # Módulo de treinamento inicial e utilizando apenas bounding boxes
└── train_iterativo.ipynb        # Módulo de treinamento principal do modelo
```

O dataset de imagens e os pesos pesados do modelo não estão incluídos diretamente no repositório por boas práticas de versionamento. O treinamento pode ser reproduzido via train_bbox.ipynb e a API executada via api.py.

## ▶ 9. Passo a Passo (Git Flow & Google Colab)
Por ser um projeto solo estruturado com metodologia ágil (Kanban/Scrum no Trello) e GitFlow, o desenvolvimento é realizado **`utilizando o Google Colab (com suporte a GPU)`** para os treinamentos e manipulação de IA, e o versionamento via comandos Git integrados (onde o fluxo de desenvolvimento separa as entregas em branches de features ramificadas a partir da develop e consolidadas na main.<br>
- Estrutura de Branches (5):<br>
```text
├── VisionAutoCheck/
│   └── main/        		          # Branch principal
│       └── develop/     		  # Branch de desenvolvimento
│           ├── feature/dataset-prep      # Branch de preparação do dataset
│           ├── feature/fastapi-setup  	  # Branch da API
│           └── feature/yolo-finetuning	  # Branch do YOLO

Esta estrutura já deve estar preparada lá no GitHub*.

Branchs locais e temporárias poderão aparecer durante o desenvolvimento.

* O nome do serviço é uma palavra composta (CamelCase) que junta dois termos:
    - Git (o sistema de controle de versão criado por Linus Torvalds)
    - Hub (termo em inglês para um centro de conexões/concentrador)
```

- Passo Inicial de Configuração e Setup (Google Colab):<br>
Antes de iniciar os treinamentos ou scripts de desenvolvimento, utilize o notebook de inicialização automatizada `setup_inicial_vac.ipynb` fornecido na raiz do projeto. Ele é responsável por:
   - Configurar as credenciais do Git no ambiente Colab;
   - Inicializar o repositório local e estruturar as ramificações (main, develop, feature/dataset-prep) conectadas ao GitHub via Token (PAT);
   - Gerar e versionar automaticamente os arquivos essenciais (requirements.txt, .gitignore e data.yaml configurado para as 20 classes do projeto).

- Instalação de Dependências (via Notebook ou Terminal):

```bash
!pip install -r requirements.txt
```

- Passos de Execução do Pipeline de IA no *Google Colab*:<br>

   1\. Configuração e Sincronização Inicial:<br>
   Abra o notebook `setup_inicial_vac.ipynb` no Google Colab e execute as células para inicializar o ambiente, configurar as credenciais do Git, versionar os arquivos essenciais (`data.yaml`, `.gitignore`) e sincronizar as branches. O módulo central de utilitários (`funcoes_vac.py`) e os demais scripts do projeto serão carregados diretamente do repositório versionado.

   2\. Execução do Pipeline e Treinamento (Fine-Tuning):<br>
   Com o ambiente configurado e o dataset estruturado, o treinamento do modelo YOLO utilizando as 20 classes de autopeças e ferramentas da oficina pode ser disparado por meio do script de treino principal:

```bash
!python train_bbox.ipynb
```

- Execução da API FastAPI:<br>
Nota: Para expor a API FastAPI a partir do ambiente do Colab para testes externos ou via Swagger, recomenda-se o uso de ferramentas de túnel como o ngrok ou a execução local controlada:
```bash
!uvicorn main:app --reload
```
Acesse a documentação interativa em: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

- Exemplo de Payload JSON de Retorno esperado (/detect):<br>
<BR>JSON
```bash
{
  "status": "success",
  "detections": [
    {
      "class_name": "bateria",
      "confidence": 0.94,
      "box": {"x_min": 120, "y_min": 180, "x_max": 280, "y_max": 320}
    },
    {
      "class_name": "filtro_oleo",
      "confidence": 0.89,
      "box": {"x_min": 310, "y_min": 90, "x_max": 390, "y_max": 170}
    }
  ]
}
```
O `setup_inicial_vac.ipynb` seria a ferramenta oficial de bootstrap do projeto.

## 💡 10. Contribuição
Contribuições, melhorias de código ou sugestões de novos modelos são sempre bem-vindas!<br>
Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## 📜 11. Licença
Este projeto é distribuído sob a licença MIT. Sinta-se livre para utilizá-lo em pesquisas, projetos acadêmicos ou soluções comerciais voltadas ao setor automotivo.

## 👨‍🏫 Desenvolvido por 
Paulo Sérgio Barreiros Lima<br>
Bacharel em Ciências da Computação / Aperfeiçoamento em ML e Visão Computacional<br>
✉ paulosergiobarreiroslima@gmail.com
👔 www.linkedin.com/in/paulosergiobarreiroslima
