import os
import time
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# =======================
# LOAD ENV VARIABLES
# ======================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# =======================
# CUSTOM TOOL (API)
# ======================
@tool
def get_weather_data(city: str):
    """
    Fetch current weather information for a city
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"city: {city}\n"
        f"Temperature: {data['current']['temperature']}⁰C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity; {data['current']['humidity']}%"
    )


search_tool = TavilySearchResults(max_results=3)

# =======================
# LLM
# ======================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# =======================
# PROMPT
# ======================
prompt = hub.pull("hwchase17/react")

# =======================
# TOOLS
# ======================
tools = [search_tool, get_weather_data]

# =======================
# CREATE AGENT
# ======================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


def run_agent(question: str):
    response = agent_executor.invoke({"input": question})
    return response["output"]


def stream_agent(question: str):
    for chunk in agent_executor.stream({"input": question}):
        if "output" in chunk:
            yield chunk["output"]
        elif "actions" in chunk:
            for action in chunk["actions"]:
                yield f"Tool call: {action.tool}\n"
        elif "steps" in chunk:
            for step in chunk["steps"]:
                if hasattr(step, "observation"):
                    yield f"Observation: {step.observation}\n"


# =======================
# STREAMLIT UI
# ======================
def main():
    st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
    st.title("AI Agent with Search + Weather")

    with st.form("agent_form"):
        user_question = st.text_area(
            "Ask the agent",
            value="Find the capital of Burundi and then find its current weather",
            height=120,
        )
        submitted = st.form_submit_button("Run")

    if submitted and user_question.strip():
        with st.spinner("Agent is thinking..."):
            try:
                final_response = run_agent(user_question)
            except Exception as error:
                st.error(f"Unable to get a response: {error}")
            else:
                if final_response and str(final_response).strip():
                    st.subheader("Response")
                    st.markdown(str(final_response))
                    success_alert = st.empty()
                    success_alert.success("Response generated successfully.")
                    time.sleep(2)
                    success_alert.empty()
                else:
                    st.warning("No response returned from the agent.")


if __name__ == "__main__":
    main()
