def init_session():
    return {
        "turn": 0,
        "history": []  # últimos N turnos
    }

def update_session(session, user_query, results):
    session["turn"] += 1

    filtered_results = simplify_results(results)

    session["history"].append({
        "turn": session["turn"],
        "query": user_query,
        "results": filtered_results
    })

    session["history"] = session["history"][-2:]
    return session

def simplify_results(results, top_k=3):
    simplified = []

    for r in results[:top_k]:
        meta = r.get("metadata", {})
        simplified.append({
            "name": meta.get("name", ""),
            "categories": meta.get("categories", ""),
            "review": meta.get("reviews", "")
        })

    return simplified