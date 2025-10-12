from flask import Flask, render_template, request, jsonify
from AgenteIA import perguntar_politica_RAG 
from AgenteIA import processar_envio_relatorio
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/pergunta", methods=["POST"])
def pergunta():
    data = request.json
    pergunta_usuario = data.get("mensagem")
    resposta = perguntar_politica_RAG(pergunta_usuario)
    return jsonify(resposta)

@app.route('/enviar_relatorio', methods=['POST'])
def enviar_relatorio():
    try:
        data = request.get_json()
        email_destino = data.get('email_destino')
        session_id = data.get('session_id')
        
        if not email_destino or not session_id:
            return jsonify({
                "success": False,
                "message": "Email e session_id são obrigatórios"
            })
        
        resultado = processar_envio_relatorio(session_id, email_destino)
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": f"Erro interno: {str(e)}"
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)