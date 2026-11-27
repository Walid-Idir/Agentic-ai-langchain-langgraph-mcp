"""
Serveur MCP SAV — Outil distant `rag_multimodal` (streamable-http)
Auteur : Walid IDIR

Derrière l'outil `rag_multimodal`, un agent RAG recherche des passages textuels
et des figures issues des manuels techniques (ex: imprimante PX-200) et produit
une réponse sourcée.

Lancement :
    python sav_mcp_server.py
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from mcp.server.mcpserver import MCPServer

from langgraph_agentic_rag import LocalDeterministicEmbeddings, _has_real_openai_key, get_llm

load_dotenv(override=True)

mcp = MCPServer(name="sav-multimodal-mcp-server")

# Base documentaire multimodale (passages textuels + figures indexées des manuels SAV)
MANUELS_CORPUS = [
    Document(
        page_content=(
            "[Manuel PX-200 | Section 4.2 — Voyants et pictogrammes d'état | Page 18 | Figure FIG-4.2]\n"
            "Voyant orange clignotant accompagné du pictogramme d'une feuille avec flèche courbée "
            "(référence visuelle : img_px200_01) : indique un bourrage papier dans le bac d'alimentation arrière "
            "ou un capteur de passage papier obstrué. Action recommandée : éteindre l'appareil, ouvrir le capot "
            "arrière, retirer délicatement la feuille coincée sans la déchirer, puis appuyer 2 secondes sur Reset."
        ),
        metadata={"modele": "PX-200", "source": "Manuel_PX200_v2.pdf", "page": 18, "figure": "FIG-4.2", "image_ref": "img_px200_01"},
    ),
    Document(
        page_content=(
            "[Manuel PX-200 | Section 4.3 — Erreurs de cartouche | Page 21 | Figure FIG-4.3]\n"
            "Voyant rouge fixe avec icône de goutte barrée (référence visuelle : img_px200_02) : "
            "niveau d'encre critique ou cartouche non reconnue. Vérifier le loquet de verrouillage et nettoyer "
            "les contacts dorés avec un chiffon sec non pelucheux."
        ),
        metadata={"modele": "PX-200", "source": "Manuel_PX200_v2.pdf", "page": 21, "figure": "FIG-4.3", "image_ref": "img_px200_02"},
    ),
    Document(
        page_content=(
            "[Manuel PX-200 | Section 5.1 — Connectivité réseau | Page 27 | Figure FIG-5.1]\n"
            "Icône Wi-Fi clignotante bleue (référence visuelle : img_px200_03) : appareil en mode appairage WPS "
            "ou perte de connexion avec le routeur. Maintenir le bouton Wi-Fi pendant 5 secondes pour relancer la configuration."
        ),
        metadata={"modele": "PX-200", "source": "Manuel_PX200_v2.pdf", "page": 27, "figure": "FIG-5.1", "image_ref": "img_px200_03"},
    ),
    Document(
        page_content=(
            "[Manuel RX-500 | Section 3.1 — Surchauffe thermique | Page 12 | Figure FIG-3.1]\n"
            "Symbole thermomètre clignotant : température interne excessive du module de fusion. "
            "Laisser refroidir l'appareil 15 minutes et dégager les grilles de ventilation latérales."
        ),
        metadata={"modele": "RX-500", "source": "Manuel_RX500_v1.pdf", "page": 12, "figure": "FIG-3.1", "image_ref": "img_rx500_01"},
    ),
]

# Index des descriptions visuelles des photos d'écran envoyées par les clients
IMAGE_CATALOGUE = {
    "img_px200_01": "Photo d'écran PX-200 montrant un voyant orange clignotant et le pictogramme feuille avec flèche courbée (FIG-4.2).",
    "img_px200_02": "Photo d'écran PX-200 montrant un voyant rouge fixe et une icône goutte barrée (FIG-4.3).",
    "img_px200_03": "Photo d'écran PX-200 montrant une icône Wi-Fi bleue clignotante (FIG-5.1).",
}

_embeddings = (
    OpenAIEmbeddings(model="text-embedding-3-small")
    if _has_real_openai_key()
    else LocalDeterministicEmbeddings()
)
_manual_vectorstore = FAISS.from_documents(MANUELS_CORPUS, _embeddings)


def _run_rag_multimodal_agent(question: str, image_ref: str | None = None) -> dict:
    """Agent RAG interne qui récupère passages et figures dans les manuels et produit une réponse sourcée."""
    visual_desc = IMAGE_CATALOGUE.get(image_ref or "", "")
    search_query = f"{question} {image_ref or ''} {visual_desc}".strip()

    # Prioriser une correspondance exacte sur image_ref si présente, complétée par FAISS
    exact_docs = [d for d in MANUELS_CORPUS if image_ref and d.metadata.get("image_ref") == image_ref]
    semantic_docs = _manual_vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(search_query)

    seen = set()
    retrieved: list[Document] = []
    for doc in exact_docs + semantic_docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            retrieved.append(doc)

    sources = [
        f"{d.metadata.get('source')} (p. {d.metadata.get('page')}, {d.metadata.get('figure')})"
        for d in retrieved
    ]
    context = "\n\n".join(d.page_content for d in retrieved)

    llm = get_llm(temperature=0.0)
    if llm is not None:
        prompt = (
            "Tu es un agent RAG multimodal du service après-vente. "
            "Réponds en français à partir EXCLUSIVEMENT des extraits et figures du manuel ci-dessous, "
            "et cite explicitement les sources (manuel, page, figure).\n\n"
            f"Question client : {question}\n"
            f"Référence image : {image_ref or 'Aucune'}\n"
            f"Analyse visuelle : {visual_desc or 'Non fournie'}\n\n"
            f"Extraits récupérés :\n{context}"
        )
        answer_text = llm.invoke(prompt).content
    else:
        primary = retrieved[0]
        answer_text = (
            f"D'après le manuel technique **{primary.metadata.get('source')}** "
            f"(page {primary.metadata.get('page')}, figure **{primary.metadata.get('figure')}**) :\n\n"
            f"{primary.page_content}\n\n"
            f"**Sources consultées :** {', '.join(sources)}"
        )

    return {
        "answer": answer_text,
        "image_ref": image_ref,
        "visual_analysis": visual_desc or None,
        "sources": sources,
    }


@mcp.tool()
def rag_multimodal(question: str, image_ref: str | None = None) -> dict:
    """Outil RAG multimodal du SAV : recherche des passages et des figures dans les manuels
    techniques à partir d'une question et éventuellement d'une référence photo (`image_ref`),
    puis retourne une réponse sourcée.
    """
    print(f"[MCP Server SAV - rag_multimodal] question='{question}' | image_ref='{image_ref}'")
    return _run_rag_multimodal_agent(question=question, image_ref=image_ref)


if __name__ == "__main__":
    port = int(os.getenv("SAV_MCP_PORT", "24001"))
    print(f"🚀 Serveur MCP SAV (`rag_multimodal`) démarré sur http://0.0.0.0:{port}/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
