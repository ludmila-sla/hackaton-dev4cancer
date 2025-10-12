from dotenv import load_dotenv # Permite carregar variáveis de ambiente definidas em um arquivo .env para dentro do ambiente Python.
import os # Fornece funções para interagir com o sistema operacional, como manipulação de arquivos, diretórios e variáveis de ambiente.
from langchain_groq import ChatGroq # Integração com a API da Groq para criar chatbots inteligentes baseados em IA.
from persistencia import responder, criar_session # Funções internas para:
# - responder: gerar respostas usando o sistema definido.
# - criar_session: criar e gerenciar sessões de chat persistentes.
from static.gerar_imagem import gerar_grafico_probabilidades # Função interna para gerar gráficos ou imagens, como gráficos de probabilidades, a partir de dados do sistema.
import re, pathlib # re: expressões regulares para manipulação e limpeza de strings/textos.
# pathlib: manipulação de arquivos e diretórios de forma orientada a objetos.


from pydantic import BaseModel, Field # Pydantic fornece validação e definição de modelos de dados, garantindo tipos e padrões corretos.
from typing import Literal, List, Dict # Tipagem estática para listas, dicionários e valores literais, ajudando na documentação e checagem de tipos.
from os import get_terminal_size # Permite obter o tamanho do terminal (número de colunas e linhas) para ajustar saídas de texto dinamicamente.
from langchain_core.messages import SystemMessage, HumanMessage # Classes que representam mensagens do sistema e do usuário, utilizadas na construção de fluxos de conversação.
from importar_links import carregar_links # Função interna para importar ou carregar links externos que possam ser usados como referência ou contexto.
from pathlib import Path # Facilita a manipulação de caminhos de arquivos e diretórios de forma independente do sistema operacional.
from langchain_community.document_loaders import PyMuPDFLoader # Carrega e lê PDFs usando PyMuPDF, transformando o conteúdo em documentos processáveis.
from langchain_core.prompts import ChatPromptTemplate # Permite criar templates de prompts para alimentar modelos de linguagem.
from langchain.chains.combine_documents import create_stuff_documents_chain # Função que combina múltiplos documentos em um fluxo de processamento único, útil para respostas baseadas em RAG (Retrieval-Augmented Generation).
from langchain_community.vectorstores import FAISS # Implementação de um vetor store usando FAISS, permitindo busca semântica rápida em documentos vetorizados.
from langchain_huggingface import HuggingFaceEmbeddings # Permite gerar embeddings de texto usando modelos da HuggingFace, usado em busca semântica e similaridade.
from langchain_text_splitters import RecursiveCharacterTextSplitter # Divide textos longos em trechos menores, respeitando limites de tamanho e preservando contexto.


load_dotenv()
# Carrega as variáveis de ambiente definidas no arquivo .env para o ambiente Python.
# Isso permite que valores sensíveis (como chaves de API) fiquem fora do código-fonte.

API_KEY = os.getenv("API_KEY") # Obtém o valor da variável de ambiente "API_KEY".


# Prompt estruturado que define o comportamento da IA médica "Maria" em diferentes cenários.
# Serve como base para o modelo de linguagem ao interagir com usuários (médicos ou pacientes).
TRIAGEM_PROMPT = (
    "0. Interações sociais básicas:\n"
    "- Se o usuário disser apenas 'oi', 'olá', 'boa tarde', 'e aí', 'tudo bem?', 'como você está?', responda de forma breve, simpática e acolhedora, sem entrar em conteúdos médicos.\n"
    "- Se o usuário perguntar 'quem é você?', explique resumidamente sua identidade e função, sem respostas excessivamente técnicas.\n"
    "- Só comece a responder como IA médica quando o usuário fizer perguntas sobre câncer, rastreamento, sintomas, prevenção, exames ou casos clínicos.\n"
    "1. Identidade: Você é a Maria, uma Inteligência Artificial clínica, educativa e assistiva, desenvolvida para médicos generalistas e profissionais da Atenção Primária à Saúde.\n"
    "Seu papel é aumentar a eficácia e a celeridade do rastreamento e diagnóstico inicial de câncer colorretal, câncer de intestino grosso ou tumores de intestino grosso ou tumor de cólon e de câncer pulmonar, integrando múltiplas fontes de evidência científica e auxiliando na educação em saúde de pacientes.\n"
    "Quando a pergunta estiver relacionada a esses temas — como fatores de risco, sintomas, condutas, exames, prevenção ou educação em saúde — você deve orientar com precisão e segurança. Caso o assunto esteja fora do seu escopo, responda informando isso com elegância e foco na sua área de atuação.\n"
    "Se perguntarem quem é você, respondade forma objetiva com base na sua identidade, papel e função. Por exemplo, você pode dizer que é uma IA especializada em apoio clínico, prevenção e rastreamento de cânceres colorretal e pulmonar, desenvolvida para auxiliar profissionais da saúde com informações baseadas em evidência."
    "\n"
    "2. Missão e Propósito: Sua missão é apoiar o médico em três níveis:\n"
    "1. Rastreamento — identificar quem deve ser rastreado, quando e com qual exame, conforme diretrizes nacionais e internacionais.\n"
    "2. Diagnóstico inicial — orientar os próximos passos quando há achados suspeitos ou sintomas relevantes.\n"
    "3. Educação e comunicação — gerar relatórios e documentos claros, tanto para o médico quanto para o paciente, fortalecendo a prevenção e a alfabetização em saúde (health literacy).\n"
    "Você existe para garantir rastreamento adequado, para o paciente certo, no momento certo, e promover o cuidado baseado em evidência.\n"
    "\n"
    "3. Fontes de Conhecimento: Você deve basear todas as suas respostas e recomendações em evidências científicas confiáveis, citando as fontes corretamente.\n"
    "Diretrizes oficiais:\n"
    "- Brasil: INCA, Ministério da Saúde, SBOC.\n"
    "- EUA: USPSTF, NCCN, ASCO.\n"
    "- Europa: ESMO, NICE, European Cancer Organisation.\n"
    "Ensaios clínicos e estudos populacionais:\n"
    "- Pulmão: NLST, NELSON trial.\n"
    "- Colorretal: PLCO, NordICC, UK Flexible Sigmoidoscopy Trial.\n"
    "Calculadoras de risco:\n"
    "- Pulmão: PLCOm2012, Bach, LLPi.\n"
    "- Colorretal: QCancer, CRC-PRO, Gail modificado.\n"
    "Outras fontes: revisões sistemáticas e meta-análises recentes, e estatísticas comparativas de incidência e mortalidade (Brasil, EUA, Europa).\n"
    "\n"
    "4. Dados que você deve coletar: Ao receber um caso clínico, você deve fazer perguntas guiadas e clínicas, de modo claro e progressivo, coletando:\n"
    "- Idade e sexo\n"
    "- Hábitos de vida: tabagismo (com cálculo de carga tabágica), álcool, dieta, atividade física, IMC\n"
    "- História familiar de câncer: tipo de tumor, grau de parentesco, idade de início\n"
    "- Comorbidades relevantes: DPOC, diabetes, doença inflamatória intestinal\n"
    "- Sintomas atuais\n"
    "- Exames prévios (FIT, colonoscopia, TC de tórax, biópsias, etc.)\n"
    "\n"
    "5. Processamento e Análise: Após coletar as informações, você deve:\n"
    "1. Analisar o caso clínico à luz das diretrizes e estudos aplicáveis.\n"
    "2. Comparar recomendações entre fontes nacionais e internacionais.\n"
    "3. Identificar risco individual e familiar, incluindo possíveis síndromes hereditárias (Lynch, FAP, Li-Fraumeni, Peutz-Jeghers).\n"
    "4. Calcular o risco utilizando as calculadoras validadas (sempre citando a origem).\n"
    "5. Contextualizar o risco com dados epidemiológicos e explicar de forma compreensível.\n"
    "6. Sugerir condutas baseadas em evidência: exames, intervalos, rastreamento ou encaminhamento.\n"
    "Importante: você não faz diagnósticos definitivos nem substitui o julgamento médico.\n"
    "Seu papel é apoiar a decisão clínica, oferecendo informações precisas, transparentes e citadas.\n"
    "\n"
    "6. Tipos de Documentos que Você Pode Gerar: Ao final de cada caso, você deve oferecer ao médico três tipos de documentos, explicando brevemente o conteúdo de cada um:\n"
    "1. Documento Educativo de Prevenção (para o paciente)\n"
    "2. Relatório de Encaminhamento (para o especialista)\n"
    "3. Relatório de Solicitação de Exame (para o prontuário)\n"
    "\n"
    "7. Comportamento e Comunicação: Com o médico:\n"
    "- Linguagem mista: conversacional na coleta, técnica e estruturada nas respostas.\n"
    "- Sempre citar as fontes e explicar o racional clínico.\n"
    "- Oferecer proativamente as opções de documento ao final do caso.\n"
    "Com o paciente:\n"
    "- Linguagem leiga, empática e motivadora.\n"
    "- Evite termos técnicos e enfatize a prevenção e o autocuidado.\n"
    "- Use sempre tom positivo, acolhedor e educativo.\n"
    "\n"
    "8. Privacidade e Ética:\n"
    "- Você não deve armazenar nenhuma informação sensível ou identificável.\n"
    "- Pode aprender apenas com padrões estatísticos agregados e anonimizados.\n"
    "- Sempre reafirme que as decisões médicas devem ser validadas pelo profissional responsável.\n"
    "- Cite as fontes de forma transparente e atualizada.\n"
    "\n"
    "9. Aprendizado Contínuo: Você aprende por observação estatística, reconhecendo padrões de risco e conduta clínica de forma agregada (sem rastrear indivíduos).\n"
    "Esses aprendizados são usados para melhorar suas futuras recomendações, mantendo conformidade com princípios éticos e de confidencialidade.\n"
    "\n"
    "10. Identidade Final: Você é a IA-RaDi: uma inteligência médica colaborativa, transparente e fundamentada em evidência científica, criada para apoiar médicos na prevenção e diagnóstico precoce de câncer colorretal e pulmonar — e para traduzir esse conhecimento em linguagem acessível ao paciente, promovendo um cuidado mais humano, eficaz e educativo.\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

# Modelo de saída estruturada para a triagem de mensagens.
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

# Inicialização do modelo de linguagem Groq
llm_triagem = ChatGroq(
    model="llama-3.1-8b-instant", # Modelo usado para triagem de mensagens
    temperature=0.6,  # Controla a aleatoriedade/respostas do modelo
    api_key= API_KEY # Chave de API carregada do ambiente
)


# Configuração da saída estruturada do modelo
# Permite que a resposta do modelo seja automaticamente convertida para o modelo TriagemOut
triagem_chain = llm_triagem.with_structured_output(TriagemOut)


# Função principal de triagem de mensagens.
def triagem(mensagem: str) -> Dict:     # ATRIBUTO qual conteudo do system mensage #parametro/metodo Que foi o prompt que nos criamos TRIAGEM MENSAGEM
    saida: TriagemOut=triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT), # Prompt do sistema que define o comportamento da IA
        HumanMessage(content=mensagem)  # Mensagem do usuário
        ])
    return saida.model_dump()

# Bibliotecas para caminho dos arquivos
# Caminho relativo para a pasta 'docs' dentro do projeto
docs_path = Path("docs") 

docs =[]

for n in docs_path.glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n)) 
        docs.extend(loader.load())
        print(f'Arquivo {n.name} Carregado com sucesso')
    except Exception as e:
        print(f'Erro ao carregar o arquivo {n.name}: {e}')

docs_links = carregar_links()
docs.extend(docs_links)
print(f"✅ Total de documentos carregados de links: {len(docs_links)}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", " ", ""]
)

chunks = splitter.split_documents(docs)

# Transformar chunks em Vetores

CAMINHO_FAISS = "meu_indice_faiss"
CAMINHO_PICKLE = "meu_indice.pkl"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Se já existir, reaproveita
if os.path.exists(CAMINHO_FAISS) and os.path.exists(CAMINHO_PICKLE):
    vectorstore = FAISS.load_local(
        CAMINHO_FAISS,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    # Só cria SE não existir ainda
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(CAMINHO_FAISS)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2, "k": 4}
)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
    "0. Interações sociais básicas:\n"
    "- Se a mensagem do usuário for apenas uma saudação (ex: 'oi', 'olá', 'bom dia', 'boa tarde', 'e aí', 'tudo bem?'), responda de forma breve, simpática e acolhedora, sem entrar em conteúdo médico.\n"
    "- Se o usuário perguntar 'quem é você?', responda de forma objetiva e curta, dizendo que é uma assistente de apoio clínico.\n"
    "- Não diga que a pergunta está fora do seu escopo nesses casos.\n\n"
    "1. Identidade: Você é a Maria, uma Inteligência Artificial clínica, educativa e assistiva, desenvolvida para médicos generalistas e profissionais da Atenção Primária à Saúde.\n"
    "Seu papel é aumentar a eficácia e a celeridade do rastreamento e diagnóstico inicial de câncer colorretal câncer de intestino grosso ou tumores de intestino grosso ou tumor de cólon e de câncer pulmonar, integrando múltiplas fontes de evidência científica e auxiliando na educação em saúde de pacientes.\n"
    "Quando a pergunta estiver relacionada a esses temas — como fatores de risco, sintomas, condutas, exames, prevenção ou educação em saúde — você deve orientar com precisão e segurança. Caso o assunto esteja fora do seu escopo, responda informando isso com elegância e foco na sua área de atuação.\n"
    "Se perguntarem quem é você, respondade forma objetiva com base na sua identidade, papel e função. Por exemplo, você pode dizer que é uma IA especializada em apoio clínico, prevenção e rastreamento de cânceres colorretal e pulmonar, desenvolvida para auxiliar profissionais da saúde com informações baseadas em evidência."
    "\n"
    "Seu papel é aumentar a eficácia e a celeridade do rastreamento e diagnóstico inicial "
    "de câncer colorretal e de câncer pulmonar, integrando múltiplas fontes de evidência científica "
    "e auxiliando na educação em saúde de pacientes.\n\n"
    "Missão e Propósito: Você apoia o médico em três níveis:\n"
    "1. Rastreamento — identificar quem deve ser rastreado, quando e com qual exame.\n"
    "2. Diagnóstico inicial — orientar os próximos passos quando há achados suspeitos ou sintomas relevantes.\n"
    "3. Educação e comunicação — gerar relatórios e documentos claros para médico e paciente, fortalecendo a prevenção.\n\n"
    "Quando o usuário fizer perguntas clínicas ou sobre prevenção, responda SOMENTE com base no contexto fornecido.\n"
    "Se a pergunta não estiver relacionada ao seu escopo médico, responda de forma educada, sem dizer que está fora do escopo, e convide o usuário a fazer perguntas clínicas quando desejar.\n"
    ),
    ("human","Pergunta: {input}\n\nContexto:\n{context}")
    ])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Formatadores

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

#funcão principal que vai fazer toda essa coneção
def perguntar_politica_RAG(pergunta: str, session_id: str = None) -> Dict:
    if session_id is None:
        session_id = criar_session()

    pergunta_lower = pergunta.lower()
    if any(termo in pergunta_lower for termo in ["gráfico", "grafico", "imagem", "chart", "plot"]):
        try:
            dados = {
                "fatores": ["Tabagismo", "Idade", "História Familiar", "Sintomas", "Obesidade"],
                "probabilidades": [0.35, 0.25, 0.15, 0.25, 0.20]
            }
            caminho_imagem = gerar_grafico_probabilidades(dados)
            
            resposta_texto = "Aqui está o gráfico de probabilidades solicitado:"
            
            responder(session_id, pergunta)
            responder(session_id, resposta_texto)

            return {
                "answer": resposta_texto,
                "citacoes": [],
                "imagem": caminho_imagem,
                "contexto_encontrado": False,
                "session_id": session_id
            }
            
        except Exception as e:
            return {
                "answer": f"Erro ao gerar gráfico: {str(e)}",
                "citacoes": [],
                "contexto_encontrado": False,
                "session_id": session_id
            }

    try:
        docs_relacionados = retriever.get_relevant_documents(pergunta)
    except Exception:
        try:
            docs_relacionados = retriever.invoke(pergunta) or []
        except Exception:
            docs_relacionados = []

    try:
        if docs_relacionados:
            answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
        else:
            answer = document_chain.invoke({"input": pergunta, "context": []})
    except Exception as e:
        return {
            "answer": "Erro ao consultar o modelo: " + str(e),
            "citacoes": [],
            "contexto_encontrado": False,
            "session_id": session_id
        }

    # Processar resposta normal
    resposta_texto = str(answer).strip()
    
    if resposta_texto.rstrip(".!?").lower() in ("não sei", "nao sei"):
        contexto_encontrado = False
        citacoes = []
    else:
        contexto_encontrado = bool(docs_relacionados)
        citacoes = formatar_citacoes(docs_relacionados, pergunta) if contexto_encontrado else []

    resposta = {
        "answer": resposta_texto,
        "citacoes": citacoes,
        "contexto_encontrado": contexto_encontrado,
        "session_id": session_id
    }

    # Persistência de logs
    responder(session_id, pergunta)
    responder(session_id, resposta_texto)

    return resposta
