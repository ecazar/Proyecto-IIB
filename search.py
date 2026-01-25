from pathlib import Path

from kagglehub import dataset_download
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
from tqdm.notebook import tqdm # progress bar
import chromadb
from chromadb.config import Settings
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import CrossEncoder
import copy

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = SentenceTransformer('clip-ViT-B-32')

def check_data():
    path_emb_img = Path("models/emb_img.npy")
    path_emb_txt = Path("models/emb_txt.npy")
    chroma_db = Path("models/chroma_db/")

    if not path_emb_img.is_file() or not path_emb_txt.is_file() or not any(chroma_db.iterdir()):
        df = read_csv()
        df, df_text, df_review, df_image = preprocesar_dataset(df)
        save_embeddings_text(df_text=df_text)
        save_embeddings_image(df_image=df_image)

        embeddings_text, embeddings_images = load_embeddings()
        chroma_client = connect_chromadb()
        create_chromadb(chroma_client, df, df_text, df_image, embeddings_text, embeddings_images)



def download_dataset():
    path = dataset_download("datafiniti/consumer-reviews-of-amazon-products")
    df = pd.read_csv(path + "/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
                usecols=['name','categories','imageURLs','reviews.doRecommend','reviews.text','reviews.title'])
    return df

def read_csv():
    df = pd.read_csv("models/df_limpio.csv")
    return df

def preprocesar_dataset(df):
    df_text = df['name'] + " " + df['categories']
    df_image = df['imageURLs']
    df_review = df['reviews.title'] + " " + df['reviews.text']
    df_review = df_review.fillna("")
    df_recommend = df['reviews.doRecommend']

    df_text = (df_text.astype(str).str.lower().str.replace("&", " and ").str.replace("/", " ").str.replace(r"\s+", " ", regex=True).str.strip())
    df_review = (df_review.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())

    df['imageURLs'] = df_image.apply(check_first_valid_image)

    return df, df_text, df_review, df_image

def check_first_valid_image(urls, timeout=5):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in map(str.strip, str(urls).split(",")):
        if not url.startswith("http"):
            continue
        try:
            response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    return url
        except Exception:
            continue
    return None

def download_first_valid_image(urls, timeout=5):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in map(str.strip, str(urls).split(",")):
        if not url.startswith("http"):
            continue
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            return img
        except Exception:
            continue
    return None

def save_embeddings_text(df_text):
    ls_text = df_text.tolist()
    embeddings_text = model.encode(ls_text, batch_size=128, show_progress_bar=True)  # uso de gpu
    np.save('models/emb_txt.npy', embeddings_text)
    return None

def save_embeddings_image(df_image):
    chunk_size = 250
    total_embeddings = []
    indices_validos = []
    reporte_fallos = []

    pbar = tqdm(total=len(df_image), desc="Buscando imagen válida por producto")

    for start in range(0, len(df_image), chunk_size):
        batch_urls = df_image[start: start + chunk_size]
        batch_images = []
        exitos_en_este_batch = 0

        for offset, url_block in enumerate(batch_urls):
            idx = start + offset
            img = download_first_valid_image(url_block)

            if img is not None:
                batch_images.append(img)
                indices_validos.append(idx)
                exitos_en_este_batch += 1
            else:
                reporte_fallos.append({
                    "index": idx,
                    "error": "Ninguna URL funcionó"
                })
            pbar.update(1)

        if batch_images:
            embeddings = model.encode(
                batch_images,
                batch_size=64,
                show_progress_bar=False
            )
            total_embeddings.append(embeddings)
        # LIMPIEZA RAM
        del batch_images
        gc.collect()

    pbar.close()

    embeddings_images = np.vstack(total_embeddings)
    np.save('models/emb_img.npy', embeddings_images)
    return None

def load_embeddings():
    embeddings_text = np.load('models/emb_txt.npy')
    embeddings_images = np.load('models/emb_img.npy')
    return embeddings_text, embeddings_images

def connect_chromadb():
    chroma_client = chromadb.PersistentClient(path="models/chroma_db")
    return chroma_client

def create_chromadb(chroma_client, df, df_text, df_image, embeddings_text, embeddings_images):
    collection_text = chroma_client.create_collection(
        name="products_text"
    )
    collection_image = chroma_client.create_collection(
        name="products_image"
    )
    ids = df.index.astype(str).tolist()
    metadatas = [
        {
            "name": df.loc[i, "name"],
            "categories": df.loc[i, "categories"],
            "imageURL": df.loc[i, "imageURLs"],
            "reviews": str(df.loc[i, "reviews.title"] or "") + " " + str(df.loc[i, "reviews.text"] or ""),
            "doRecommend": str(df.loc[i, "reviews.doRecommend"])
        }
        for i in df.index
    ]
    collection_text.add(
        ids=ids,
        embeddings=embeddings_text,
        documents=df_text.tolist(),
        metadatas=metadatas
    )
    collection_image.add(
        ids=ids,
        embeddings=embeddings_images,
        documents=df_image.tolist(),
        metadatas=metadatas
    )

def load_chromadb(chroma_client):
    collection_text = chroma_client.get_collection(name="products_text")
    collection_image = chroma_client.get_collection(name="products_image")
    return collection_text, collection_image

def search_products(texto_query=None, imagen_query=None, collection=None, top_k=10):
    queries_combinated = []
    if texto_query:
        query_vector = model.encode(texto_query, convert_to_numpy=True)
        queries_combinated.append(query_vector)
    if imagen_query:
        img = Image.open(imagen_query) if isinstance(imagen_query, str) else imagen_query
        query_vector = model.encode(img, convert_to_numpy=True)
        queries_combinated.append(query_vector)
    if len(queries_combinated) == 2:
        query_vector = (0.7*queries_combinated[0] + 0.3*queries_combinated[1]) / 2
    else:
        query_vector = queries_combinated[0]
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=top_k
    )
    candidates = []
    for i in range(len(results['ids'][0])):
        candidates.append({
            "id": results['ids'][0][i],
            "score": results['distances'][0][i],
            "metadata": results['metadatas'][0][i]
        })
    return candidates

def mostrar_resultados(candidates):
    print("\n" + "="*50)
    print(f"🔍 RESULTADOS (Top {len(candidates)})")
    print("="*50 + "\n")
    if not candidates:
        print("❌ No se encontraron productos.")
        return
    for i, c in enumerate(candidates, 1):
        meta = c['metadata']
        id_prod = c.get('id', 'N/A')
        titulo = meta.get('name', 'Sin título')
        categoria = meta.get('categories', 'N/A')
        score = c.get('score', 0.0)
        fuente_img = meta.get('imageURL') or "No disponible"
        print(f"{i}. [ID: {id_prod}] - DISTANCIA: {score:.4f}")
        print(f"   📦 TÍTULO:    {titulo}")
        print(f"   🏷️  CATEGORÍA: {categoria}")
        print(f"   🖼️  IMAGEN:    {fuente_img}")
        print("-" * 50)

def rerank_results(texto_query, candidates):
    pairs = []
    for c in candidates:
        meta = c["metadata"]
        doc_text = (
            meta.get("name", "") + " " +
            meta.get("categories", "") + " " +
            meta.get("reviews.text", "")
        )
        pairs.append((texto_query, doc_text))
    scores = cross_encoder.predict(pairs)

    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return candidates

def search_with_rerank(texto_query=None,imagen_query=None,collection=None, top_k=10):
    # retrival basic
    retrieval_results = search_products(texto_query=texto_query, imagen_query=imagen_query, collection=collection, top_k=top_k)
    if texto_query is None:
        # 2️. Si NO hay texto → no hay re-ranking
        return {
            "retrieval": retrieval_results,
            "reranked": retrieval_results,
            "reranking_applied": False
        }
    else:
        # 2. Re-ranking textual
        reranked_results = rerank_results(texto_query, copy.deepcopy(retrieval_results))
        return {
            "retrieval": retrieval_results,
            "reranked": reranked_results,
            "reranking_applied": True
        }
