"""
C - Partie 3 : Serveur MCP exposant deux tools via le protocole streamable-http
Master 5IADM — EMSI / ENSET Mohammedia
Auteur : Walid IDIR

Lancement :
    python mcp_server.py
Test avec MCP Inspector :
    npx @modelcontextprotocol/inspector
    (Transport: Streamable HTTP | URL: http://localhost:24000/mcp)
"""

import os
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from langgraph_agentic_rag import _VECTOR_STORE

load_dotenv(override=True)

# Initialisation du serveur MCP (MCP v2.x)
mcp = MCPServer(name="tp-lang-mcp-server")


@mcp.tool()
def get_employee_infos(name: str) -> dict:
    """Retourne les informations professionnelles d'un employé ou enseignant (nom, salaire, ancienneté, poste)."""
    directory = {
        "mohamed youssfi": {
            "name": "Mohamed Youssfi",
            "role": "Professeur en Informatique & IA",
            "institution": "ENSET Mohammedia - UH2C",
            "salary": 23000,
            "seniority": 31,
        },
        "walid idir": {
            "name": "Walid IDIR",
            "role": "Ingénieur IA / Big Data (5IADM)",
            "institution": "EMSI",
            "salary": 18000,
            "seniority": 2,
        },
        "hassan": {
            "name": "Hassan",
            "role": "Développeur Senior IA",
            "institution": "TechLab Casablanca",
            "salary": 23000,
            "seniority": 12,
        },
    }
    key = name.strip().lower()
    if key in directory:
        return directory[key]
    return {
        "name": name,
        "role": "Consultant IA",
        "salary": 15000,
        "seniority": 5,
    }


@mcp.tool()
def rag_search(query: str) -> str:
    """Recherche sémantique dans la base documentaire RAG (CV Mohamed Youssfi, Transformer, LangGraph, MCP, Master 5IADM)."""
    print(f"[MCP Server - rag_search] Requête reçue : {query}")
    retriever = _VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if not docs:
        return "Aucune information trouvée dans la base documentaire."
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "24000"))
    print(f"🚀 Démarrage du serveur MCP (streamable-http) sur http://0.0.0.0:{port}/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
