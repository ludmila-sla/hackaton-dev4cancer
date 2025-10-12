import uuid

conversas = {}

def criar_session():
    return str(uuid.uuid4())

def responder(session_id, user_msg):
    if session_id not in conversas:
        conversas[session_id] = []
    conversas[session_id].append({"role": "user", "content": user_msg})

def obter_historico_conversa(session_id: str):
    """
    Recupera o histórico completo de uma sessão da memória
    """
    if session_id not in conversas:
        return []
    
    historico = []
    for mensagem in conversas[session_id]:
        historico.append({
            "tipo": mensagem["role"],
            "mensagem": mensagem["content"],
            "timestamp": "Não disponível"
        })
    
    return historico
