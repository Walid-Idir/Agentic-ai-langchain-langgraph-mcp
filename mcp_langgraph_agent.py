"""
C - Partie 3 : Agent LangChain / LangGraph consommant le serveur MCP via streamable-http
Auteur : Walid IDIR

Prérequis :
    1. Démarrer le serveur MCP dans un premier terminal : python mcp_server.py
    2. Exécuter ce script dans un second terminal : python mcp_langgraph_agent.py
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from mcp import Client

from langgraph_agentic_rag import get_llm

load_dotenv(override=True)


def _has_real_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.startswith("mock")


def convertir_outil_mcp(client: Client, outil) -> StructuredTool:
    """Convertit un outil MCP découvert via streamable-http en StructuredTool LangChain."""

    async def appeler(**arguments) -> str:
        resultat = await client.call_tool(outil.name, arguments=arguments)
        if resultat.is_error:
            raise RuntimeError(f"Échec de {outil.name} : {resultat.content}")

        contenu = resultat.structured_content
        if contenu is None:
            contenu = [bloc.model_dump(mode="json") for bloc in resultat.content]
        return json.dumps(contenu, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=appeler,
        name=outil.name,
        description=outil.description or outil.name,
        args_schema=outil.input_schema,
    )


async def run_autonomous_mcp_demo(tools: list[StructuredTool], queries: list[str]) -> None:
    """Démontre l'invocation réelle des outils MCP via streamable-http même sans clé OpenAI."""
    tool_map = {t.name: t for t in tools}
    print("\nℹ️ Mode autonome (sans OPENAI_API_KEY) : appel réel des outils MCP via streamable-http.")

    for q in queries:
        print(f"\n👤 Question : {q}")
        q_lower = q.lower()
        if any(k in q_lower for k in ["salaire", "employé", "hassan", "walid", "ancienneté"]):
            name = "Hassan" if "hassan" in q_lower else ("Walid IDIR" if "walid" in q_lower else "Mohamed Youssfi")
            tool_obj = tool_map["get_employee_infos"]
            print(f"🔧 Appel MCP Tool `{tool_obj.name}` avec args={{'name': '{name}'}}")
            result = await tool_obj.ainvoke({"name": name})
            print(f"🤖 Réponse Agent :\nInformations récupérées via MCP (`get_employee_infos`) : {result}")
        else:
            tool_obj = tool_map["rag_search"]
            print(f"🔧 Appel MCP Tool `{tool_obj.name}` avec args={{'query': '{q}'}}")
            result = await tool_obj.ainvoke({"query": q})
            print(f"🤖 Réponse Agent :\nRésultats récupérés via MCP (`rag_search`) :\n{json.loads(result)}")


async def main() -> None:
    mcp_url = os.getenv("MCP_URL", "http://localhost:24000/mcp")
    print(f"🔌 Connexion au serveur MCP via streamable-http : {mcp_url}")

    async with Client(mcp_url) as client:
        catalogue = await client.list_tools()
        tools = [convertir_outil_mcp(client, outil) for outil in catalogue.tools]
        print(f"✅ {len(tools)} outils récupérés depuis le serveur MCP :")
        for t in tools:
            print(f"   - {t.name} : {t.description}")

        default_queries = [
            "Quel est le salaire et l'ancienneté de Hassan ?",
            "Quels sont les diplômes et l'affiliation de Mohamed Youssfi ?",
        ]

        llm = get_llm(temperature=0.0)
        if llm is None:
            await run_autonomous_mcp_demo(tools, default_queries)
            return

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "Tu es un assistant intelligent connecté à un serveur MCP via streamable-http. "
                "Utilise les outils MCP disponibles pour répondre précisément en français au format Markdown."
            ),
        )

        if "--demo" in sys.argv:
            for q in default_queries:
                print(f"\n👤 Question : {q}")
                resp = await agent.ainvoke({"messages": [HumanMessage(content=q)]})
                print(f"🤖 Réponse Agent :\n{resp['messages'][-1].content}")
            return

        print("\n💬 Mode interactif démarré (tapez 'exit' pour quitter).")
        while True:
            user_query = input("\nQuestion : ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit", "q"}:
                break
            resp = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})
            print(f"\n🤖 Réponse :\n{resp['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
