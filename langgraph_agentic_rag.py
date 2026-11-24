"""
B - Partie LangGraph : Chatbot Agentic RAG avec LangGraph
Master 5IADM — EMSI / ENSET Mohammedia
Auteur : Walid IDIR

Ce module définit le graphe LangGraph complet (StateGraph) pour un Chatbot Agentic RAG :
- `graph` : graphe compilé sans checkpointer externe (compatible avec LangGraph Studio)
- `graph_with_memory` : graphe compilé avec InMemorySaver (pour Streamlit et usage autonome)
"""

import logging
import os
import urllib.request
from typing import Annotated, Any, Literal, Sequence, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("langgraph_agentic_rag")


def _has_real_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.startswith("mock") and key != "ollama"


def _is_ollama_available() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    tags_url = base_url[:-3] + "/api/tags" if base_url.endswith("/v1") else base_url + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_active_backend_label() -> str:
    if _has_real_openai_key():
        return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
    if _is_ollama_available():
        return f"Ollama Local ({os.getenv('OLLAMA_MODEL', 'qwen2.5:1.5b')})"
    return "Mode autonome local (simulation)"


def get_llm(temperature: float = 0.0) -> ChatOpenAI | None:
    """Retourne un client ChatOpenAI (OpenAI distant ou Ollama local `qwen2.5:1.5b`)."""
    if _has_real_openai_key():
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    if _is_ollama_available():
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="ollama",
            temperature=temperature,
        )
    return None


class LocalDeterministicEmbeddings(Embeddings):
    """Embeddings déterministes locaux utilisés automatiquement en l'absence de clé OpenAI."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        # Combinaison de tokens et préfixes (stemming léger) pour une meilleure similarité locale
        vec = np.zeros(256, dtype=np.float32)
        tokens = [tok.strip(".,;:!?()'\"«»").lower() for tok in text.split() if len(tok) > 2]
        for tok in tokens:
            vec[abs(hash(tok)) % 256] += 1.0
            if len(tok) >= 5:
                vec[abs(hash(tok[:5])) % 256] += 0.8
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).tolist()


def get_embeddings_model() -> Embeddings:
    if _has_real_openai_key():
        return OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info("Mode autonome : utilisation des embeddings lexicaux locaux.")
    return LocalDeterministicEmbeddings()


DEFAULT_CORPUS = [
    (
        "Je m'appelle Mohamed Youssfi, je suis Professeur en Informatique et Intelligence Artificielle "
        "à l'ENSET Mohammedia, Université Hassan II de Casablanca."
    ),
    (
        "Parcours académique de Mohamed Youssfi : Doctorat d'État en 2015 dans le domaine des systèmes "
        "informatiques parallèles et distribués, Doctorat de troisième cycle en 1996, et Diplôme de "
        "professeur de second cycle en 1993 à la Faculté des Sciences de Rabat."
    ),
    (
        "L'architecture Transformer a été présentée par Vaswani et al. en 2017 dans l'article fondateur "
        "'Attention Is All You Need'. Elle remplace les couches récurrentes (RNN) par le mécanisme "
        "d'auto-attention (Self-Attention)."
    ),
    (
        "La formule du Scaled Dot-Product Attention est : "
        "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. "
        "La division par sqrt(d_k) évite la saturation de la fonction softmax lorsque d_k est grand."
    ),
    (
        "LangGraph permet de modéliser des agents IA sous forme de graphes d'états cycliques (StateGraph) "
        "avec des nœuds (nodes), des arêtes conditionnelles (conditional edges), une gestion de mémoire "
        "persistante (Checkpointer) et du Human-in-the-Loop."
    ),
    (
        "Le Master 5IADM de l'EMSI forme des ingénieurs spécialisés en Intelligence Artificielle, "
        "Big Data, RAG Agentique, LangChain, LangGraph et protocole MCP (Model Context Protocol)."
    ),
]


def build_vectorstore(extra_texts: list[str] | None = None, pdf_dir: str = "./pdfs") -> FAISS:
    """Construit la base vectorielle FAISS à partir du corpus par défaut, de PDFs éventuels et de textes ajoutés."""
    texts = list(DEFAULT_CORPUS)
    if extra_texts:
        texts.extend(extra_texts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documents: list[Document] = splitter.create_documents(texts)

    if os.path.isdir(pdf_dir):
        try:
            from langchain_community.document_loaders import PyPDFDirectoryLoader

            loader = PyPDFDirectoryLoader(path=pdf_dir)
            pdf_docs = loader.load_and_split(text_splitter=splitter)
            if pdf_docs:
                documents.extend(pdf_docs)
                logger.info("Chargé %d fragments depuis %s", len(pdf_docs), pdf_dir)
        except Exception as exc:
            logger.warning("Impossible de charger les PDFs depuis %s : %s", pdf_dir, exc)

    return FAISS.from_documents(documents, get_embeddings_model())


_VECTOR_STORE = build_vectorstore()


def update_vectorstore(new_texts: list[str]) -> int:
    """Ajoute dynamiquement de nouveaux documents à l'index vectoriel actif."""
    global _VECTOR_STORE
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs = splitter.create_documents(new_texts)
    _VECTOR_STORE.add_documents(docs)
    return len(docs)


@tool
def retriever_tool(query: str) -> str:
    """Recherche des informations pertinentes dans la base de connaissances (CV Prof. Youssfi,
    architecture Transformer, LangGraph, MCP, Master 5IADM et documents importés).

    Args:
        query: Question ou mots-clés de recherche.
    Returns:
        Les extraits les plus pertinents de la base documentaire.
    """
    logger.info("TOOL [retriever_tool] -> Requête : %s", query)
    retriever = _VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if not docs:
        return "Aucun document pertinent trouvé dans la base de connaissances."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class AgentState(TypedDict):
    """État du graphe LangGraph contenant l'historique des messages."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


SYSTEM_PROMPT = (
    "Tu es un assistant IA RAG (LangGraph). "
    "Tu DOIS appeler l'outil `retriever_tool` pour toute question sur Mohamed Youssfi, "
    "l'architecture Transformer, LangGraph, MCP, le Master 5IADM ou des documents importés. "
    "Réponds directement sans outil uniquement pour les salutations simples (bonjour, salut, merci). "
    "Base ta réponse finale uniquement sur les extraits retournés par `retriever_tool`."
)


def _offline_llm_decision(messages: Sequence[BaseMessage]) -> AIMessage:
    """Moteur ReAct local de démonstration lorsqu'aucune clé OPENAI_API_KEY n'est fournie."""
    last = messages[-1]
    if isinstance(last, ToolMessage):
        # Synthèse à partir des documents retournés par retriever_tool
        context = str(last.content)
        # Retrouver la dernière question humaine
        last_human = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = str(msg.content)
                break
        return AIMessage(
            content=(
                f"Voici la réponse synthétisée à partir de la base documentaire pour *« {last_human} »* :\n\n"
                f"{context}\n\n"
                "*(Réponse générée par le pipeline LangGraph Agentic RAG en mode autonome)*"
            )
        )

    user_text = str(getattr(last, "content", "")).strip()
    lower = user_text.lower()
    greetings = ["bonjour", "salut", "hello", "hi", "ça va", "comment vas-tu", "merci"]
    if any(g in lower for g in greetings) and len(lower.split()) <= 6:
        return AIMessage(
            content=(
                "Bonjour ! Je suis votre Chatbot **Agentic RAG sous LangGraph**. "
                "Posez-moi des questions sur les Transformers, LangGraph, MCP, le Master 5IADM "
                "ou vos documents importés."
            )
        )

    if "mon nom" in lower or "comment je m'appelle" in lower:
        for msg in messages:
            if isinstance(msg, HumanMessage) and "m'appelle" in str(msg.content).lower():
                return AIMessage(content=f"D'après notre conversation précédente : {msg.content}")

    # Décider d'appeler retriever_tool
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retriever_tool",
                "args": {"query": user_text},
                "id": f"call_rag_{abs(hash(user_text)) % 100000}",
                "type": "tool_call",
            }
        ],
    )


def agent_node(state: AgentState) -> dict[str, Any]:
    """Nœud principal (Mind) : raisonne et décide s'il faut appeler retriever_tool ou répondre."""
    logger.info("AGENT NODE: Traitement des messages (%d message(s))", len(state["messages"]))
    llm = get_llm(temperature=0.0)
    if llm is not None:
        llm_with_tools = llm.bind_tools([retriever_tool])
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)

        # Pour les petits modèles locaux (ex: qwen2.5:1.5b), garantir l'appel de retriever_tool
        # sur les questions factuelles si aucun ToolMessage n'a encore été produit pour ce tour.
        last_in = state["messages"][-1]
        if isinstance(last_in, HumanMessage) and not getattr(response, "tool_calls", None):
            user_txt = str(last_in.content).strip()
            lower = user_txt.lower()
            greetings = ["bonjour", "salut", "hello", "hi", "ça va", "comment vas-tu", "merci"]
            is_greeting = any(g in lower for g in greetings) and len(lower.split()) <= 6
            if not is_greeting and not _has_real_openai_key():
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retriever_tool",
                            "args": {"query": user_txt},
                            "id": f"call_rag_{abs(hash(user_txt)) % 100000}",
                            "type": "tool_call",
                        }
                    ],
                )

        # Normalise les arguments si le modèle utilise un autre nom de clé
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                if tc.get("name") == "retriever_tool" and "query" not in tc.get("args", {}):
                    args = tc.get("args", {})
                    first_val = next(iter(args.values()), "") if args else ""
                    tc["args"] = {"query": str(first_val)}
    else:
        response = _offline_llm_decision(state["messages"])

    if getattr(response, "tool_calls", None):
        logger.info(" -> Décision : appel de %d outil(s)", len(response.tool_calls))
    else:
        logger.info(" -> Décision : réponse finale directe")
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Arête conditionnelle : route vers `tools` si le LLM a émis des tool_calls, sinon termine."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# Construction du graphe LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([retriever_tool]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

# Graphe sans checkpointer pour LangGraph Studio (la plateforme fournit son propre checkpointer)
graph = workflow.compile()

# Graphe avec mémoire InMemorySaver pour Streamlit et tests CLI locaux
memory = InMemorySaver()
graph_with_memory = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    print("=== Test rapide du graphe LangGraph Agentic RAG ===")
    cfg = {"configurable": {"thread_id": "test-cli-1"}}
    for q in [
        "Bonjour !",
        "Quels sont les diplômes et l'affiliation de Mohamed Youssfi ?",
        "Quelle est la formule du Scaled Dot-Product Attention ?",
    ]:
        print(f"\n👤 Question : {q}")
        res = graph_with_memory.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)
        print(f"🤖 Réponse :\n{res['messages'][-1].content}")
