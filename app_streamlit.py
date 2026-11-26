"""
Interface Web Streamlit pour le Chatbot Agentic RAG & le Système Multi-Agent SAV (LangGraph + MCP)
Master 5IADM — EMSI / ENSET Mohammedia
Auteur : Walid IDIR

Lancement :
    streamlit run app_streamlit.py
"""

import asyncio
import os
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_agentic_rag import (
    _has_real_openai_key,
    get_active_backend_label,
    graph_with_memory,
    update_vectorstore,
)
from sav_orchestrator import graph_with_memory as sav_graph_with_memory

st.set_page_config(
    page_title="TP LangGraph & MCP — Walid IDIR",
    page_icon="🤖",
    layout="wide",
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    app_mode = st.radio(
        "Application à tester",
        [
            "🤖 Chatbot Agentic RAG (Partie B)",
            "🛠️ SAV Multi-Agent + MCP (Orchestrateur)",
        ],
    )

    ollama_model = st.selectbox(
        "Modèle local Ollama",
        ["qwen2.5:1.5b", "tinyllama:latest"],
        index=0,
        help="Utilisé automatiquement lorsque Ollama tourne en local (sans clé OpenAI).",
    )
    os.environ["OLLAMA_MODEL"] = ollama_model

    api_key_input = st.text_input(
        "Clé API OpenAI (optionnelle)",
        value="",
        type="password",
        help="Laissez vide pour utiliser votre modèle local Ollama (qwen2.5:1.5b), ou renseignez une clé sk-...",
    )
    if api_key_input.strip():
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()

    backend_label = get_active_backend_label()
    if _has_real_openai_key() or "Ollama" in backend_label:
        st.success(f"Modèle actif : {backend_label}")
    else:
        st.info(f"Modèle actif : {backend_label}")

    st.divider()
    st.subheader("💬 Gestion de session (Mémoire)")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session-{str(uuid.uuid4())[:6]}"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sav_history" not in st.session_state:
        st.session_state.sav_history = []

    st.text_input("Thread ID actif", key="thread_id")
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.thread_id = f"session-{str(uuid.uuid4())[:6]}"
        st.session_state.chat_history = []
        st.session_state.sav_history = []
        st.rerun()

    if app_mode.startswith("🤖"):
        st.divider()
        st.subheader("📄 Enrichir la base RAG")
        uploaded_files = st.file_uploader(
            "Importer des fichiers (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Indexer les fichiers", use_container_width=True):
            extracted_texts = []
            for up_file in uploaded_files:
                if up_file.name.lower().endswith(".pdf"):
                    try:
                        import fitz  # PyMuPDF

                        doc = fitz.open(stream=up_file.read(), filetype="pdf")
                        pdf_text = "\n".join(page.get_text() for page in doc)
                        extracted_texts.append(pdf_text)
                    except Exception as exc:
                        st.error(f"Erreur lecture PDF {up_file.name} : {exc}")
                else:
                    extracted_texts.append(up_file.read().decode("utf-8", errors="ignore"))

            if extracted_texts:
                count = update_vectorstore(extracted_texts)
                st.success(f"{count} fragment(s) ajouté(s) à l'index FAISS !")

        st.divider()
        st.subheader("🔀 Graphe LangGraph")
        try:
            st.code(graph_with_memory.get_graph().draw_ascii(), language="text")
        except Exception:
            st.code(graph_with_memory.get_graph().draw_mermaid(), language="mermaid")
    else:
        st.divider()
        st.subheader("🖼️ Photo jointe (Optionnelle)")
        sav_image_ref = st.selectbox(
            "Référence image écran",
            ["Aucune", "img_px200_01", "img_px200_02", "img_px200_03"],
            help="Simule une photo jointe par le client pour l'outil MCP `rag_multimodal`.",
        )
        st.caption("Numéros de série en registre : `PX200-0042`, `PX200-0011`, `RX500-0108`")
        st.divider()
        st.subheader("🔀 Graphe SAV")
        st.code(sav_graph_with_memory.get_graph().draw_mermaid(), language="mermaid")


# ==============================================================================
# MODE 1 : CHATBOT AGENTIC RAG (PARTIE B)
# ==============================================================================
if app_mode.startswith("🤖"):
    st.title("🤖 Chatbot Agentic RAG avec LangGraph")
    st.caption("Architecture ReAct cyclique (Mind / Body) avec mémoire par session (`thread_id`) et outil RAG FAISS.")

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("tool_traces"):
                with st.expander("🔍 Détails du raisonnement Agentic RAG (Tool Calls)"):
                    for trace in item["tool_traces"]:
                        st.markdown(f"**Outil appelé :** `{trace['tool']}` | **Arguments :** `{trace['args']}`")
                        st.text(trace["output"])

    if prompt := st.chat_input("Posez votre question (ex: Diplômes de Mohamed Youssfi, formule Attention, LangGraph...)"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        prev_len = len(graph_with_memory.get_state(config).values.get("messages", []))

        with st.chat_message("assistant"):
            with st.spinner("Raisonnement LangGraph en cours..."):
                result = graph_with_memory.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )
                new_messages = result["messages"][prev_len:]

                tool_traces = []
                pending_calls = {}
                for msg in new_messages:
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            pending_calls[tc["id"]] = {"tool": tc["name"], "args": tc["args"]}
                    elif isinstance(msg, ToolMessage):
                        meta = pending_calls.get(msg.tool_call_id, {"tool": msg.name, "args": {}})
                        tool_traces.append({
                            "tool": meta["tool"],
                            "args": meta["args"],
                            "output": str(msg.content),
                        })

                answer = result["messages"][-1].content
                st.markdown(answer)
                if tool_traces:
                    with st.expander("🔍 Détails du raisonnement Agentic RAG (Tool Calls)"):
                        for trace in tool_traces:
                            st.markdown(f"**Outil appelé :** `{trace['tool']}` | **Arguments :** `{trace['args']}`")
                            st.text(trace["output"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "tool_traces": tool_traces,
        })

# ==============================================================================
# MODE 2 : SYSTÈME MULTI-AGENT SAV + MCP
# ==============================================================================
else:
    st.title("🛠️ Assistant SAV Multi-Agent (LangGraph + MCP)")
    st.caption(
        "Orchestrateur LangGraph routant vers `agent_documentaire` (outil MCP `rag_multimodal` sur le port 24001) "
        "ou `agent_garantie` (outil local `consulter_garantie`)."
    )

    col1, col2 = st.columns(2)
    quick_prompt = None
    quick_img = None
    with col1:
        if st.button("📘 Tester Exemple 1 (Documentation + img_px200_01)", use_container_width=True):
            quick_prompt = "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?"
            quick_img = "img_px200_01"
    with col2:
        if st.button("🛡️ Tester Exemple 2 (Garantie PX200-0042)", use_container_width=True):
            quick_prompt = "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042."
            quick_img = None

    for item in st.session_state.sav_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("meta"):
                st.caption(item["meta"])

    user_input = st.chat_input("Décrivez votre panne ou indiquez votre numéro de série (ex: PX200-0042)...")
    active_prompt = quick_prompt or user_input
    active_img = quick_img if quick_prompt else (None if sav_image_ref == "Aucune" else sav_image_ref)

    if active_prompt:
        display_user = active_prompt + (f"\n\n*(Photo jointe : `{active_img}`)*" if active_img else "")
        st.session_state.sav_history.append({"role": "user", "content": display_user})
        with st.chat_message("user"):
            st.markdown(display_user)

        config = {"configurable": {"thread_id": f"sav-{st.session_state.thread_id}"}}
        with st.chat_message("assistant"):
            with st.spinner("Orchestration LangGraph + appel agent spécialisé en cours..."):
                try:
                    res = asyncio.run(
                        sav_graph_with_memory.ainvoke(
                            {
                                "messages": [HumanMessage(content=active_prompt)],
                                "question": active_prompt,
                                "image_ref": active_img,
                                "reponse_finale": "",
                            },
                            config=config,
                        )
                    )
                    answer = res.get("reponse_finale") or res["messages"][-1].content
                    meta_info = (
                        f"🧭 Catégorie : `{res.get('categorie')}` | "
                        f"Agent : `{res.get('agent_selectionne')}` | "
                        f"Sources : {', '.join(res.get('sources', [])) or 'Aucune'}"
                    )
                except Exception as exc:
                    answer = f"❌ Erreur (vérifiez que `sav_mcp_server.py` tourne sur le port 24001) : {exc}"
                    meta_info = ""

            st.markdown(answer)
            if meta_info:
                st.caption(meta_info)

        st.session_state.sav_history.append({
            "role": "assistant",
            "content": answer,
            "meta": meta_info,
        })
