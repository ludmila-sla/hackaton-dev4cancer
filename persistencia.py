historico = []

def responder(usuario_msg):
    historico.append({"role": "user", "content": usuario_msg})
    
    resposta = gerar_resposta(historico)
    
    historico.append({"role": "assistant", "content": resposta})
    return resposta