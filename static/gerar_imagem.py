# Exemplo de como deveria ser a função
def gerar_grafico_probabilidades(dados: dict) -> str:
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    # Criar gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(dados["fatores"], dados["probabilidades"], color='skyblue')
    plt.title('Probabilidades de Fatores de Risco')
    plt.ylabel('Probabilidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Garantir que o diretório existe
    os.makedirs('static/images', exist_ok=True)
    
    # Salvar com nome único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = f'static/images/grafico_{timestamp}.png'
    plt.savefig(caminho)
    plt.close()
    
    return caminho