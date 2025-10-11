from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["chat_app"]
collection = db["conversas"]

def salvar_mensagem(chat_id, role, content):
    collection.insert_one({
        "chat_id": chat_id,
        "role": role,
        "content": content
    })

def carregar_historico(chat_id):
    return list(collection.find({"chat_id": chat_id}))

def responder(chat_id, usuario_msg):
    salvar_mensagem(chat_id, "user", usuario_msg)
    
    historico = carregar_historico(chat_id)
    resposta = gerar_resposta(historico)
    
    salvar_mensagem(chat_id, "assistant", resposta)
    return resposta
