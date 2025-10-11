from flask import Flask, render_template, request, jsonify
from AgenteIA import perguntar_politica_RAG 
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

if __name__ == "__main__":
    app.run(debug=True)
