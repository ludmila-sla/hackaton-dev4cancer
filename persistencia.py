import uuid

def criar_session():
    return str(uuid.uuid4())

conversas = {}

def responder(session_id, user_msg):
    if session_id not in conversas:
        conversas[session_id] = []

    conversas[session_id].append({"role": "user", "content": user_msg})

    resposta = gerar_resposta(conversas[session_id])

    conversas[session_id].append({"role": "assistant", "content": resposta})

    return resposta
