from rag import call_llm

def build_conversational_context(session):
    context = "\nHistorial reciente:\n"

    for h in session["history"]:
        context += f"\nTurn {h['turn']}\n"
        context += f"Query: {h['query']}\n"

        for i, r in enumerate(h["results"], 1):
            context += f"Result {i}:\n"
            context += f"- Name: {r['name']}\n"
            context += f"- Categories: {r['categories']}\n"
            context += f"- Review: {r['review']}\n"

    return context

def rewrite_query_llm(user_query, session, client):
    history_text = ""
    for h in session["history"]:
        history_text += f"- {h['query']}\n"
    prompt = f"""
    You rewrite search queries for a multimodal CLIP-based retrieval system.

    Previous queries:
    {history_text}

    Current query:
    {user_query}

    Rules:
    - If the current query refines the previous intent, merge attributes into ONE concise query.
    - Keep the original product type from the previous query.
    - Use simple attribute + object phrasing (CLIP-friendly).
    - Do NOT repeat attributes unnecessarily.
    - Do NOT explain anything.

    Return ONLY the final rewritten query.
    """
    return call_llm(prompt, client)

def rewrite_query(user_query, session, rewrite_cache):
    history_queries = tuple(h["query"] for h in session["history"])
    cache_key = (history_queries, user_query)
    if cache_key in rewrite_cache:
        return rewrite_cache[cache_key]
    # no hay historia → no refinamiento
    if not session["history"]:
        rewritten = user_query
    # no parece refinamiento → usar tal cual
    elif not is_refinement(user_query):
        rewritten = user_query
    # refinamiento → LLM
    else:
        rewritten = rewrite_query_llm(user_query, session)
    rewrite_cache[cache_key] = rewritten
    return rewritten, rewrite_cache

def is_refinement(user_query):
    refinement_keywords = [
    "same", "similar", "like", "that", "those", "these", "one", "ones",
    "better", "best", "worse", "instead", "but", "rather",
    "more", "less", "cheaper", "expensive", "costly", "premium",
    "bigger", "smaller", "larger", "lighter", "heavier",
    "prefer", "preferred", "want", "need",
    "without", "no", "not",
    "only", "just"
]

    q = user_query.lower()
    return any(k in q for k in refinement_keywords)