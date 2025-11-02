from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model_name="gpt-4")  # , temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def main():
    result = chain.invoke(
        {
            "input": "Procure no Linkedin por 3 vagas de emprego recentes que mencionam conhecimento em Langchain como habilidade do candidato. Liste seus detalhes"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
