"""
Système Multi-Agent SAV piloté par un Orchestrateur LangGraph + Client MCP (`streamable-http`)
Auteur : Walid IDIR

Architecture :
    START -> orchestrateur
      ├── categorie == "documentation" -> agent_documentaire (outil distant MCP `rag_multimodal`) -> orchestrateur -> END
      └── categorie == "garantie"      -> agent_garantie (outil local `consulter_garantie`)          -> orchestrateur -> END
"""

import asyncio
import json
import os
import re
from types import MappingProxyType
from typing import Annotated, Any, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import Client

from langgraph_agentic_rag import get_llm

load_dotenv(override=True)

SAV_MCP_URL = os.getenv("SAV_MCP_URL", "http://127.0.0.1:24001/mcp")


def _has_real_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.startswith("mock")


# ==============================================================================
# 1. REGISTRE EN LECTURE SEULE & OUTIL LOCAL `consulter_garantie`
# ==============================================================================
_REGISTRE_GARANTIE_DATA = {
    "PX200-0042": MappingProxyType({
        "numero_serie": "PX200-0042",
        "modele": "Imprimante PX-200",
        "date_achat": "2025-06-15",
        "fin_garantie": "2027-06-15",
        "statut": "Sous garantie",
        "type_couverture": "Garantie constructeur 2 ans (pièces et main-d'œuvre)",
    }),
    "PX200-0011": MappingProxyType({
        "numero_serie": "PX200-0011",
        "modele": "Imprimante PX-200",
        "date_achat": "2022-03-10",
        "fin_garantie": "2024-03-10",
        "statut": "Expirée",
        "type_couverture": "Garantie expirée",
    }),
    "RX500-0108": MappingProxyType({
        "numero_serie": "RX500-0108",
        "modele": "Imprimante RX-500",
        "date_achat": "2026-01-20",
        "fin_garantie": "2028-01-20",
        "statut": "Sous garantie",
        "type_couverture": "Garantie constructeur étendue 2 ans",
    }),
}
REGISTRE_GARANTIE = MappingProxyType(_REGISTRE_GARANTIE_DATA)


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Interroge en lecture seule le registre des garanties à partir d'un numéro de série."""
    cle = numero_serie.strip().upper()
    print(f"🛠️ [Tool Local] consulter_garantie(numero_serie='{cle}')")
    fiche = REGISTRE_GARANTIE.get(cle)
    if fiche is None:
        return {
            "trouve": False,
            "numero_serie": cle,
            "message": f"Aucun appareil enregistré avec le numéro de série '{cle}'.",
        }
    return {"trouve": True, **dict(fiche)}


# ==============================================================================
# 2. CLIENT MCP POUR L'OUTIL DISTANT `rag_multimodal`
# ==============================================================================
async def appeler_mcp_rag_multimodal(question: str, image_ref: str | None = None) -> dict:
    """Appelle l'outil distant `rag_multimodal` publié par le serveur MCP via streamable-http."""
    args: dict[str, Any] = {"question": question}
    if image_ref:
        args["image_ref"] = image_ref

    print(f"🌐 [Client MCP -> {SAV_MCP_URL}] Appel de `rag_multimodal` avec {args}")
    async with Client(SAV_MCP_URL) as client:
        resultat = await client.call_tool("rag_multimodal", arguments=args)
        if resultat.is_error:
            raise RuntimeError(f"Erreur MCP rag_multimodal : {resultat.content}")
        if resultat.structured_content is not None:
            return resultat.structured_content
        # Fallback si le contenu est retourné dans content[0].text
        if resultat.content:
            txt = getattr(resultat.content[0], "text", "")
            try:
                return json.loads(txt)
            except Exception:
                return {"answer": txt, "sources": []}
        return {"answer": "Aucune réponse retournée par le serveur MCP.", "sources": []}


# ==============================================================================
# 3. DÉFINITION DE L'ÉTAT ET DES NŒUDS LANGGRAPH
# ==============================================================================
class SAVState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    image_ref: str | None
    categorie: Literal["documentation", "garantie"]
    agent_selectionne: str
    reponse_finale: str
    sources: list[str]


def _extraire_question_et_image(state: SAVState) -> tuple[str, str | None]:
    question = state.get("question", "")
    image_ref = state.get("image_ref")
    if not question and state.get("messages"):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break
    if not image_ref and question:
        m = re.search(r"\b(img_[a-zA-Z0-9_]+)\b", question)
        if m:
            image_ref = m.group(1)
    return question, image_ref


def _classifier_demande(question: str, image_ref: str | None) -> Literal["documentation", "garantie"]:
    # Si une photo d'écran est jointe sans demande explicite de garantie, c'est du dépannage documentaire
    q_low = question.lower()
    if image_ref and "garantie" not in q_low:
        return "documentation"

    llm = get_llm(temperature=0.0)
    if llm is not None:
        prompt = (
            "Tu es un classificateur SAV. Choisis UNE SEULE catégorie : documentation OU garantie.\n"
            "Exemples :\n"
            "- 'Mon imprimante clignote, que signifie ce voyant ?' -> documentation\n"
            "- 'Est-ce que mon appareil PX200-0042 est sous garantie ?' -> garantie\n"
            "Réponds UNIQUEMENT par un mot (documentation ou garantie), sans phrase.\n\n"
            f"Demande : {question}\nImage jointe : {image_ref or 'Aucune'}\nRéponse :"
        )
        cat = llm.invoke(prompt).content.strip().lower()
        matches = re.findall(r"\b(documentation|garantie)\b", cat)
        if matches:
            return matches[0]  # type: ignore[return-value]

    if any(k in q_low for k in ["garantie", "numéro de série", "numero de serie", "série"]) or re.search(
        r"\b[A-Z]{2}\d{3}-\d{4}\b", question
    ):
        return "garantie"
    return "documentation"


async def orchestrateur_node(state: SAVState) -> dict[str, Any]:
    """Orchestrateur LangGraph :
    1) Si une réponse finale est déjà produite par un agent spécialisé, termine l'exécution.
    2) Sinon, analyse la demande, détermine la catégorie et sélectionne l'agent spécialisé.
    """
    if state.get("reponse_finale"):
        print("🏁 [Orchestrateur] Réponse finale disponible -> fin du workflow (END).")
        return {}

    question, image_ref = _extraire_question_et_image(state)
    categorie = _classifier_demande(question, image_ref)
    agent_cible = "agent_documentaire" if categorie == "documentation" else "agent_garantie"
    print(f"🧭 [Orchestrateur] Catégorie='{categorie}' | Agent sélectionné='{agent_cible}'")

    return {
        "question": question,
        "image_ref": image_ref,
        "categorie": categorie,
        "agent_selectionne": agent_cible,
    }


async def agent_documentaire_node(state: SAVState) -> dict[str, Any]:
    """Agent spécialisé documentation : appelle l'outil distant MCP `rag_multimodal`."""
    question = state.get("question", "")
    image_ref = state.get("image_ref")
    print(f"📘 [agent_documentaire] Traitement de la demande documentaire (image={image_ref})...")

    mcp_res = await appeler_mcp_rag_multimodal(question=question, image_ref=image_ref)
    answer = mcp_res.get("answer", "")
    sources = mcp_res.get("sources", [])

    return {
        "reponse_finale": answer,
        "sources": sources,
        "messages": [AIMessage(content=answer)],
    }


async def agent_garantie_node(state: SAVState) -> dict[str, Any]:
    """Agent spécialisé garantie : extrait le numéro de série et interroge `consulter_garantie`."""
    question = state.get("question", "")
    print("🛡️ [agent_garantie] Traitement de la demande de vérification de garantie...")

    match = re.search(r"\b([A-Z]{2,4}\d{2,4}-\d{3,6})\b", question.upper())
    numero_serie = match.group(1) if match else ""

    if not numero_serie:
        answer = "Veuillez fournir un numéro de série valide (ex: `PX200-0042`) afin de vérifier votre garantie."
        return {"reponse_finale": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    info = consulter_garantie.invoke({"numero_serie": numero_serie})
    if not info.get("trouve"):
        answer = info["message"]
    else:
        answer = (
            f"Vérification de garantie pour le numéro de série **{info['numero_serie']}** ({info['modele']}) :\n"
            f"- **Statut :** {info['statut']}\n"
            f"- **Date d'achat :** {info['date_achat']}\n"
            f"- **Fin de garantie :** {info['fin_garantie']}\n"
            f"- **Couverture :** {info['type_couverture']}"
        )

    return {
        "reponse_finale": answer,
        "sources": [f"Registre garantie (lecture seule) : {numero_serie}"],
        "messages": [AIMessage(content=answer)],
    }


def router_orchestrateur(state: SAVState) -> Literal["agent_documentaire", "agent_garantie", "__end__"]:
    """Route vers l'agent spécialisé choisi, ou vers END dès qu'une réponse finale existe."""
    if state.get("reponse_finale"):
        return "__end__"
    if state.get("categorie") == "garantie":
        return "agent_garantie"
    return "agent_documentaire"


# ==============================================================================
# 4. CONSTRUCTION ET COMPILATION DU GRAPHE LANGGRAPH
# ==============================================================================
workflow = StateGraph(SAVState)
workflow.add_node("orchestrateur", orchestrateur_node)
workflow.add_node("agent_documentaire", agent_documentaire_node)
workflow.add_node("agent_garantie", agent_garantie_node)

workflow.add_edge(START, "orchestrateur")
workflow.add_conditional_edges(
    "orchestrateur",
    router_orchestrateur,
    {
        "agent_documentaire": "agent_documentaire",
        "agent_garantie": "agent_garantie",
        "__end__": END,
    },
)
# Après traitement, chaque agent spécialisé rend la main à l'orchestrateur
workflow.add_edge("agent_documentaire", "orchestrateur")
workflow.add_edge("agent_garantie", "orchestrateur")

# Pour LangGraph Studio
graph = workflow.compile()
# Pour exécution locale avec persistance
graph_with_memory = workflow.compile(checkpointer=InMemorySaver())


async def executer_exemples() -> None:
    print("=== Topologie du graphe SAV LangGraph ===")
    try:
        print(graph.get_graph().draw_ascii())
    except Exception:
        print(graph.get_graph().draw_mermaid())

    # Exemple 1 — Assistance documentaire
    print("\n" + "=" * 70)
    print("EXEMPLE 1 — Assistance documentaire")
    print("=" * 70)
    ex1 = await graph.ainvoke({
        "messages": [
            HumanMessage(
                content="Mon imprimante PX-200 affiche un voyant clignotant. "
                "Que signifie le pictogramme sur la photo jointe ?"
            )
        ],
        "image_ref": "img_px200_01",
    })
    print(f"Catégorie détectée : {ex1['categorie']} | Agent : {ex1['agent_selectionne']}")
    print(f"Réponse finale :\n{ex1['reponse_finale']}")

    # Exemple 2 — Vérification de garantie
    print("\n" + "=" * 70)
    print("EXEMPLE 2 — Vérification de garantie")
    print("=" * 70)
    ex2 = await graph.ainvoke({
        "messages": [
            HumanMessage(
                content="Est-ce que mon imprimante est encore sous garantie ? "
                "Son numéro de série est PX200-0042."
            )
        ]
    })
    print(f"Catégorie détectée : {ex2['categorie']} | Agent : {ex2['agent_selectionne']}")
    print(f"Réponse finale :\n{ex2['reponse_finale']}")


if __name__ == "__main__":
    asyncio.run(executer_exemples())
