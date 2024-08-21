from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
# from langchain_text_splitters import TextSplitter
from chromadb import Client
# import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from langchain_openai import OpenAIEmbeddings
# from langchain.chains.llm import LLMChain
# from langchain_community.chat_models import ChatOllama
from sentence_transformers import SentenceTransformer
import numpy as np

QUERY = "ransomeware OR malware"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Bu program, bir DuckDuckGo araması yaparak ilgili web sayfalarının içeriğini çeker,
# özet ve Türkçe çeviri üretir, bu içerikleri vektör veritabanında depolar ve
# yerel dosyalara kaydeder.

# Web sayfası içeriğini çıkarır
def extract_web_page_content(url):
    print(f"Extracting web page from {url}")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        page_content = BeautifulSoup(response.content, 'html.parser')
        content_list = []

        # Tüm başlıkları buluyoruz (h1, h2, h3, h4, h5, h6)
        headings = page_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for heading in headings:
            heading_text = heading.get_text(strip=True)
            # Başlığın altındaki paragrafları topla
            paragraphs = heading.find_all_next('p')
            paragraph_texts = [p.get_text(strip=True) for p in paragraphs]
            content_list.append(f"{heading_text}: {' '.join(paragraph_texts)}")

        return '\n'.join(content_list)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# İçeriği bir dosyaya kaydeder
def save_content_to_file(file_name, page):
    print(f"Saving content to {file_name}")
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(page)

# Web sayfası içeriğini özetler
def summarization(chunks):
    ollama_prompt = {
        "model": "llama3",
        "prompt": f"Write a long summary of the following document. "
                  f"Only include information that is part of the document. "
                  f"Do not include your own opinion or analysis. Document: {chunks}",
        "stream": False,
        "keep_alive": "1m",
    }

    response = requests.post(OLLAMA_ENDPOINT, json=ollama_prompt).json()["response"]
    return response


def translate(text):
    ollama_prompt_translate = {
        "model": "llama3",
        "prompt": f"Bu teknik dökümanı, ilgili tüm terimleri doğru bir şekilde kullanarak, "
                  f"anlaşılır bir Türkçeye çevirmeni istiyorum. Döküman: {text}",
        "stream": False,
        "keep_alive": "1m",
    }

    response_tr = requests.post(OLLAMA_ENDPOINT, json=ollama_prompt_translate).json()["response"]
    return response_tr


def add_vector_db(collection, chunks, name, embedding_model):
    embeddings = embedding_model.encode(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"doc_{name}_{i}"],
            documents=[chunk],
            embeddings=embeddings.tolist()
        )


def main():
    chroma_client = Client()
    collection = chroma_client.create_collection(name="llm-summarize-collection")

    # Duckduckgo Search
    results = DDGS().text(QUERY, max_results=3, timelimit="3d")

    # Metni küçük chunk'lara ayırma modeli
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    for result in results:
        content = extract_web_page_content(result['href'])
        if content is None:
            continue
        chunks = splitter.split_text(content)
        add_vector_db(collection, chunks, f"cont_{result['href'].split('//')[-1].split('.com')[0]}", embedding_model)

        summary = summarization(chunks)
        add_vector_db(collection, summary, f"sum_{result['href'].split('//')[-1].split('.com')[0]}", embedding_model)

        summary_tr = translate(summary)
        add_vector_db(collection, summary_tr, f"sum_tr_{result['href'].split('//')[-1].split('.com')[0]}", embedding_model)

        filename = f"{result['href'].split('//')[1][:50].replace('/', '_')}"
        save_content_to_file(filename + ".txt", content)
        save_content_to_file(filename + "_summary_en.txt", summary)
        save_content_to_file(filename + "_summary_tr.txt", summary_tr)


if __name__ == '__main__':
    main()

