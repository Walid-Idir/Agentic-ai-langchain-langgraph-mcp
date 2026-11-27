"""
Chatbot RAG Agentique avec Mémoire et Outils
Auteur : Walid IDIR
"""

import os
import sys
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv(override=True)

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent, wrap_tool_call
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver


# 1. Corpus de connaissances documentaire
KNOWLEDGE_BASE = [
    """L'architecture Transformer a été introduite par Vaswani et al. en 2017 dans l'article 'Attention Is All You Need'.
    Elle repose sur des mécanismes d'attention sans aucune couche récurrente (RNN) ni convolutive (CNN).""",
    
    """Le Scaled Dot-Product Attention se calcule par la formule :
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V.
    Le facteur d'échelle 1/sqrt(d_k) empêche les gradients d'exploser pour des grandes dimensions d_k.""",
    
    """Le Multi-Head Attention projette linéairement Q, K et V h fois avec des matrices différentes,
    permettant au modèle de prêter attention simultanément à différentes positions et contextes.""",
    
    """Les Positional Encodings (sinusoïdaux ou appris) sont ajoutés aux embeddings d'entrée afin
    d'injecter l'ordre des tokens, puisque le mécanisme d'attention est invariant par permutation.""",
    
    """Le Master 5IADM de l'EMSI prépare des spécialistes en Intelligence Artificielle,
    Deep Learning, Systèmes Multi-Agents et architectures Big Data."""
]


def build_rag_tool():
    """Initialise le vector store FAISS et crée l'outil retriever."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(KNOWLEDGE_BASE)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="recherche_documentation",
        description="Consulter la base documentaire sur les Transformers et la formation 5IADM."
    )
    return retriever_tool


# Guardrail déterministe
@before_agent(can_jump_to=["end"])
def security_guardrail(state: AgentState, runtime):
    """Bloque les requêtes comportant des termes malveillants."""
    if not state.get("messages"):
        return None
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "").lower()
    for bad in ["hack", "exploit", "malware", "rootkit"]:
        if bad in content:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "⚠️ [Sécurité] Requête refusée en raison de la présence de mots-clés interdits."
                }],
                "jump_to": "end"
            }
    return None


# Middleware de gestion des erreurs d'outils
@wrap_tool_call
def tool_error_middleware(request, handler):
    """Empêche les plantages en cas d'erreur dans l'exécution des outils."""
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Erreur d'outil : {exc}. Veuillez reformuler ou vérifier les données.",
            tool_call_id=request.tool_call["id"]
        )


def main():
    print("=" * 65)
    print("🤖 CHATBOT RAG AGENTIQUE (LangChain & LangGraph)")
    print("   Mémoire de conversation (InMemorySaver) & Garde-fous actifs")
    print("   Tapez 'exit' ou 'quit' pour quitter.")
    print("=" * 65)
    
    # Clé d'API OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ Avertissement : OPENAI_API_KEY non détectée dans l'environnement.")
        print("Veuillez configurer votre clé dans un fichier .env ou variable système.\n")
        
    try:
        retriever_tool = build_rag_tool()
    except Exception as e:
        print(f"Erreur lors de la création du retriever : {e}")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=(
            "Vous êtes un assistant RAG Agentique bienveillant et expert. "
            "Interrogez 'recherche_documentation' dès qu'une question requiert des connaissances techniques "
            "sur les Transformers ou la formation. Pour les salutations, répondez directement."
        ),
        middleware=[security_guardrail, tool_error_middleware],
        checkpointer=memory
    )

    thread_config = {"configurable": {"thread_id": "cli_session_1"}}

    while True:
        try:
            user_input = input("\n👤 Vous : ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nAu revoir !")
                break

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=thread_config
            )
            bot_reply = result["messages"][-1].content
            print(f"\n🤖 Assistant : {bot_reply}")

        except KeyboardInterrupt:
            print("\nArrêt demandé.")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")


if __name__ == "__main__":
    main()
