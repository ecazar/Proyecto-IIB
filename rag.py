import os
from google import genai
from google.genai import types

def get_gemini_cliente():
    client = genai.Client()
    return client

def call_llm(prompt, client):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt]
    )
    return response.text

def build_rag_context(reranked_results, top_k=3):
    context = ""
    for i, c in enumerate(reranked_results[:top_k]):
        meta = c["metadata"]
        context += f"""
Producto {i+1}:
Nombre: {meta.get("name")}
Categorías: {meta.get("categories")}
Reseña: {meta.get("reviews")}
"""
    return context

def generate_answer(context, query, client):
    prompt = """
    Eres un asistente de recomendación de productos para una tienda asi que actua como tal al momento de responder.
    Debes responder ÚNICAMENTE usando la información proporcionada en el contexto.
    Justifica la recomendación usando evidencias del contexto.
    """
    prompt += f"""
        CONTEXTO:
        {context}
        CONSULTA DEL USUARIO:
        {query}
        TAREA:
        - Recomienda uno o más productos del contexto
        - Justifica la recomendación usando la información disponible"""
    return call_llm(prompt, client)