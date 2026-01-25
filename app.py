# app.py
import streamlit as st
from PIL import Image
from search import search_with_rerank, check_data, connect_chromadb, load_chromadb
from rag import build_rag_context, generate_answer, get_gemini_cliente
from session import init_session, update_session
from rewrite import rewrite_query, build_conversational_context

st.set_page_config(page_title="🛍️ Multimodal Product Chatbot", layout="wide")

st.title("🛍️ Multimodal Product Chatbot")
st.write("Ask about products with text or images, get recommendations and explanations.")

check_data()
chroma_client= connect_chromadb()
collection_text, collection_image = load_chromadb(chroma_client)
client = get_gemini_cliente()
rewrite_cache = {}
# -----------------------------
# Session state for chatbot
# -----------------------------
if "session" not in st.session_state:
    st.session_state.session = init_session()
    rewrite_cache = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # store messages for UI

# -----------------------------
# Chat input
# -----------------------------
st.markdown("### 💬 Your Message")
col1, col2 = st.columns([2,1])
with col1:
    text_query = st.text_input("Enter your query:", placeholder="e.g., pink tablet")

with col2:
    image_file = st.file_uploader("Upload image (optional):", type=["jpg", "jpeg", "png"])

if st.button("Send"):

    # Load image if uploaded
    image = Image.open(image_file) if image_file else None

    # Rewrite query (LLM or heuristic)
    rewritten_query = rewrite_query(text_query, st.session_state.session, rewrite_cache)

    # Multimodal search + rerank
    results = search_with_rerank(
        texto_query=str(rewritten_query),
        imagen_query=image,
        collection=collection_image,
        top_k=10
    )

    # Update session (store simplified results)
    st.session_state.session = update_session(
        st.session_state.session,
        text_query,
        results["reranked"][:3]
    )

    # -----------------------------
    # Build RAG context and generate answer
    # -----------------------------
    rag_context = build_rag_context(results["reranked"], top_k=3)
    conv_context = build_conversational_context(st.session_state.session)
    full_context = rag_context + "\n\n" + conv_context

    answer = generate_answer(full_context, text_query, client)

    # -----------------------------
    # Update chat history for UI
    # -----------------------------
    # User message
    st.session_state.chat_history.append({"sender": "user", "message": text_query, "image": image})

    # Bot response
    st.session_state.chat_history.append({"sender": "bot", "message": answer, "results": results})

# -----------------------------
# Display chat history
# -----------------------------
st.markdown("### 🗨️ Chat History")
for chat in st.session_state.chat_history[::-1]:  # show latest on top
    if chat["sender"] == "user":
        if chat["image"]:
            st.image(chat["image"], width=150)
        st.markdown(f"**You:** {chat['message']}")
    else:
        st.markdown(f"**Bot:** {chat['message']}")
        st.markdown("**Top results:**")
        for i, r in enumerate(chat["results"]["reranked"][:3], 1):
            meta = r["metadata"]
            st.markdown(f"{i}. {meta.get('name','')} | {meta.get('categories','')}")
            st.markdown(f"Review: {meta.get('reviews','')}")
            if meta.get("imageURL"):
                st.image(meta["imageURL"], width=120)
    st.markdown("---")
