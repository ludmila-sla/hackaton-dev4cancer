from langchain_community.document_loaders import UnstructuredURLLoader
from bs4 import BeautifulSoup

def carregar_links():
    urls = [
        "https://www.cancer.org/health-care-professionals/american-cancer-society-prevention-early-detection-guidelines/lung-cancer-screening-guidelines.html",
        "https://www.nejm.org/doi/full/10.1056/NEJMoa1911793",
        "https://www.nejm.org/doi/full/10.1056/NEJMoa1102873",
        "https://www.gov.br/inca/pt-br/assuntos/cancer/tipos/intestino/versao-para-profissionais-de-saude",
        "https://www.gov.br/inca/pt-br/assuntos/cancer/tipos/pulmao"
    ]

    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    docs = loader.load()

    for doc in docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        doc.page_content = soup.get_text(separator="\n", strip=True)

    return docs  

if __name__ == "__main__":
    docs = carregar_links()
    print(f"Total de documentos carregados: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:")
        print(f"Fonte: {doc.metadata.get('source', 'sem fonte')}")
        print(f"Primeiros 400 caracteres do conteúdo:\n{doc.page_content[:400]}")
