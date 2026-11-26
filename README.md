# 🤖 Agentic AI : LangChain, LangGraph, MCP & Orchestration Multi-Agent

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-teal.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/Protocol-MCP_HTTP-purple.svg)](https://modelcontextprotocol.io/)
[![Ollama](https://img.shields.io/badge/Local_LLM-Ollama_Qwen2.5-green.svg)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![n8n](https://img.shields.io/badge/Automation-n8n-brightgreen.svg)](https://n8n.io/)

Projet complet de conception et de développement de systèmes d'**IA Agentique** intégrant **LangChain** (agents ReAct, mémoire, outils et middlewares), **LangGraph** (graphes d'états cycliques, LangGraph Studio et interface Streamlit), le protocole **MCP (Model Context Protocol via `streamable-http`)**, l'intégration **n8n**, ainsi qu'un **système multi-agent SAV** complet.

Compatible avec **Ollama en local (`qwen2.5:1.5b`)**, **OpenAI (`gpt-4o-mini`)** et un mode de démonstration autonome.

---

## 📐 Architecture des Systèmes

### 1. Architecture Multi-Agent SAV (Orchestrateur LangGraph + Serveur MCP)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATEUR LANGGRAPH (SAV)                          │
│                                                                              │
│   Demande Client ──► [ orchestrateur ] : Analyse & classification unique     │
│                             │                                                │
│             ┌───────────────┴───────────────┐                                │
│             │ categorie == "documentation"  │ categorie == "garantie"        │
│             ▼                               ▼                                │
│  ┌───────────────────────────┐   ┌───────────────────────────────┐           │
│  │    agent_documentaire     │   │        agent_garantie         │           │
│  │  (Client MCP streamable)  │   │  (Outil local lecture seule)  │           │
│  └─────────────┬─────────────┘   └───────────────┬───────────────┘           │
│                │                                 │                           │
│                ▼                                 ▼                           │
│  ┌───────────────────────────┐   ┌───────────────────────────────┐           │
│  │   Serveur MCP (:24001)    │   │   consulter_garantie(serial)  │           │
│  │  Tool: rag_multimodal     │   │   Registre MappingProxyType   │           │
│  │  (Manuels + Figures FAISS)│   │   (ex: PX200-0042)            │           │
│  └─────────────┬─────────────┘   └───────────────┬───────────────┘           │
│                │                                 │                           │
│                └──────────────► ◄────────────────┘                           │
│                               │                                              │
│                               ▼                                              │
│             [ orchestrateur ] : Réponse finale sourcée ──► END               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. Architecture Agentic RAG & Écosystème MCP

```
┌───────────────────────────┐      streamable-http       ┌──────────────────────────┐
│  Clients & Interfaces     │ ─────────────────────────► │  Serveurs MCP (v2.3)     │
│  - Streamlit (:8501)      │   http://localhost:24000   │  - mcp_server.py (:24000)│
│  - LangGraph Studio(:2024)│   http://localhost:24001   │    • get_employee_infos  │
│  - Agent LangGraph MCP    │ ◄───────────────────────── │    • rag_search          │
│  - Workflow n8n           │    JSON / Tool Results     │  - sav_mcp_server (:24001)│
│  - MCP Inspector (:6274)  │                            │    • rag_multimodal      │
└─────────────┬─────────────┘                            └────────────┬─────────────┘
              │                                                       │
              └───────────────────────► ┌─────────────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  LLM & Index Vectoriel FAISS  │
                        │  - Ollama Local : qwen2.5:1.5b│
                        │  - OpenAI : gpt-4o-mini       │
                        │  - Base FAISS + PyMuPDF       │
                        └───────────────────────────────┘
```

---

## 🚀 Les 4 Parties Implémentées

### A — Partie LangChain ([TP_Langchain_Agentic_Walid_IDIR.ipynb](TP_Langchain_Agentic_Walid_IDIR.ipynb))
1. **Simple Agent (`create_agent`) :** Création et invocation d'un agent basé sur le pattern ReAct.
2. **Mémoire conversationnelle :** Persistance par session via `InMemorySaver` et `thread_id`.
3. **Outils personnalisés (`@tool`) :** Météo, calculatrice sécurisée, annuaire étudiant.
4. **Outils prédéfinis :** Intégration de **DuckDuckGo** (`duckduckgo-search`), **Tavily** (`tavily-python`) et **PythonREPLTool**.
5. **Middlewares avancés :**
   - `dynamic_model` (`@wrap_model_call`) : sélection dynamique du modèle selon le contexte.
   - `dynamic_prompt` (`@dynamic_prompt`) : adaptation du *System Prompt* selon le rôle utilisateur.
   - `tool_error_handling` (`@wrap_tool_call`) : interception gracieuse des exceptions d'outils.
   - `GuardRails` : filtrage déterministe (`@before_agent`), contrôle post-agent (`@after_agent`) et `PIIMiddleware`.
   - `Human In The Loop` (`HumanInTheLoopMiddleware`) : interruption et reprise après approbation via `Command(resume=...)`.
6. **Chatbot RAG Agentique :** Notebook complet et script CLI ([chatbot_rag_agentique.py](chatbot_rag_agentique.py)).

### B — Partie LangGraph, LangGraph Studio & Streamlit
1. **Graphe Agentic RAG ([langgraph_agentic_rag.py](langgraph_agentic_rag.py)) :** Construction d'un `StateGraph` (`agent_node` + `ToolNode([retriever_tool])` + arêtes conditionnelles `should_continue`) avec index **FAISS**.
2. **LangGraph Studio ([langgraph.json](langgraph.json)) :** Exposition des graphes `agentic_rag` et `sav_orchestrator` via `langgraph dev`.
3. **Interface Web Streamlit ([app_streamlit.py](app_streamlit.py)) :** Double interface interactive permettant de tester le **Chatbot Agentic RAG** (avec import de PDF/TXT et inspection des *Tool Calls*) et le **SAV Multi-Agent**.

### C — Partie MCP (`streamable-http`) & n8n
1. **Serveur MCP ([mcp_server.py](mcp_server.py)) :** Serveur `MCPServer` exposant deux tools (`get_employee_infos` et `rag_search`) sur `http://0.0.0.0:24000/mcp` via `transport="streamable-http"`.
2. **Inspection MCP :** Validé avec `@modelcontextprotocol/inspector` (`tools/list` et `tools/call`).
3. **Agent Client LangGraph ([mcp_langgraph_agent.py](mcp_langgraph_agent.py)) :** Découverte dynamique des outils MCP via `mcp.Client` + `StructuredTool.from_function`.
4. **Agent n8n ([n8n_mcp_agent_workflow.json](n8n_mcp_agent_workflow.json)) :** Workflow prêt à importer (`When chat message received` → `AI Agent` → `Simple Memory` + `MCP Client Tool` en transport `httpStreamable`).

### D — Cas Pratique : Service Après-Vente Multi-Agent ([TP_SAV_LangGraph_MCP_Walid_IDIR.ipynb](TP_SAV_LangGraph_MCP_Walid_IDIR.ipynb))
1. **Exemple 1 — Assistance documentaire :** Requête + référence photo `img_px200_01` → `orchestrateur` → `agent_documentaire` → appel MCP distant `rag_multimodal` ([sav_mcp_server.py](sav_mcp_server.py)) → réponse sourcée (`Manuel_PX200_v2.pdf`, page 18, `FIG-4.2`).
2. **Exemple 2 — Vérification de garantie :** Numéro de série `PX200-0042` → `orchestrateur` → `agent_garantie` → outil local `consulter_garantie` (registre en lecture seule) → statut détaillé.

---

## 📦 Installation et Prérequis

Cloner le dépôt :
```bash
git clone https://github.com/Walid-Idir/Agentic-ai-langchain-langgraph-mcp.git
cd Agentic-ai-langchain-langgraph-mcp
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Configurer les variables d'environnement (optionnel) :
```bash
cp .env.example .env
```
- **En local avec Ollama (par défaut si actif) :** assurez-vous d'avoir `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b` puis `ollama serve`).
- **Avec OpenAI :** renseignez `OPENAI_API_KEY` dans `.env` ou directement dans la barre latérale Streamlit.

---

## 💻 Utilisation

### 1. Lancer les serveurs MCP (`streamable-http`)
```bash
# Terminal 1 : Serveur MCP Partie C (port 24000)
python mcp_server.py

# Terminal 2 : Serveur MCP SAV rag_multimodal (port 24001)
python sav_mcp_server.py
```

### 2. Lancer l'Interface Web Streamlit
```bash
streamlit run app_streamlit.py
```
Ouvrir **`http://localhost:8501`** pour tester le **Chatbot Agentic RAG** et le **SAV Multi-Agent + MCP**.

### 3. Tester dans LangGraph Studio
```bash
langgraph dev
```
Ouvrir **`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`** et sélectionner `agentic_rag` ou `sav_orchestrator`.

### 4. Tester le serveur MCP avec MCP Inspector
```bash
npx @modelcontextprotocol/inspector
```
Choisir **Transport Type : `Streamable HTTP`**, **URL : `http://localhost:24000/mcp`** (ou `:24001/mcp`), puis cliquer sur **Connect**.

### 5. Exécuter l'Agent Client MCP ou l'Orchestrateur SAV en console
```bash
python mcp_langgraph_agent.py
python sav_orchestrator.py
```

### 6. Importer l'Agent dans n8n
```bash
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n
```
Ouvrir **`http://localhost:5678`**, importer `n8n_mcp_agent_workflow.json` et utiliser `http://host.docker.internal:24000/mcp` dans le nœud **MCP Client Tool**.

---

## 📂 Structure du Répertoire

```
├── TP_Langchain_Agentic_Walid_IDIR.ipynb   # Partie A : Notebook LangChain complet et exécuté
├── TP_SAV_LangGraph_MCP_Walid_IDIR.ipynb   # Partie D : Notebook SAV Multi-Agent + MCP exécuté
├── agentic_ai_with_langchain_V3.ipynb      # Notebook de cours LangChain
├── demo-langgraph.ipynb                    # Notebook de cours LangGraph
├── chatbot_rag_agentique.py                # Chatbot RAG Agentique en CLI (Partie A)
├── langgraph_agentic_rag.py                # Graphe StateGraph Agentic RAG + Ollama/OpenAI (Partie B)
├── langgraph.json                          # Configuration LangGraph Studio (agentic_rag & sav_orchestrator)
├── app_streamlit.py                        # Interface Web Streamlit unifiée (RAG & SAV)
├── mcp_server.py                           # Serveur MCP streamable-http (:24000) avec 2 tools (Partie C)
├── mcp_langgraph_agent.py                  # Agent LangChain/LangGraph client MCP (Partie C)
├── n8n_mcp_agent_workflow.json             # Workflow n8n AI Agent + MCP Client Tool (Partie C)
├── sav_mcp_server.py                       # Serveur MCP SAV (:24001) exposant rag_multimodal
├── sav_orchestrator.py                     # Orchestrateur SAV LangGraph + outil consulter_garantie
├── requirements.txt                        # Dépendances Python
├── .env.example                            # Modèle de configuration (Ollama / OpenAI / MCP)
└── .gitignore                              # Fichiers exclus du suivi Git
```

---

## 👨‍💻 Auteur
- **Walid IDIR**
- Projet réalisé dans le cadre du module **Agentic AI : LangChain, LangGraph & MCP**.
