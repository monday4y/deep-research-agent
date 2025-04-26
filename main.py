import os
from typing import Annotated, List, Tuple, TypedDict

# LangChain & LangGraph Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Import Google Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# Set your API keys as environment variables
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # <-- Changed to GOOGLE_API_KEY

if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY environment variable not set.")
# Check for the Google API Key
if not os.environ.get("GOOGLE_API_KEY"):
     raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the LLM (Using Google Gemini)
# Use a suitable Gemini model, e.g., "gemini-pro" or "gemini-1.5-pro-latest"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0) # <-- Changed LLM class and model

# Initialize the Tavily tool
tavily_tool = TavilySearchResults(max_results=5) # Get top 5 search results

# --- State Definition ---
# Define the state for the LangGraph
class ResearchState(TypedDict):
    """Represents the state of our research graph."""
    query: str # The original research query
    research_results: List[str] # List of research findings (e.g., snippets from Tavily)
    final_answer: str # The synthesized final answer


# --- Agent Definitions (as nodes in LangGraph) ---

# 1. Researcher Agent (Node)
def research_node(state: ResearchState) -> ResearchState:
    """
    Researches the web based on the query using Tavily.
    Updates the state with research results.
    """
    print("---RESEARCHING---")
    query = state["query"]
    print(f"Searching for: {query}")

    # Use the Tavily tool to perform the search
    # The tool returns a list of dictionaries, we'll just store the snippet
    results = tavily_tool.invoke({"query": query})
    research_results = [r['content'] for r in results]

    print(f"Found {len(research_results)} results.")

    # Update state with research results
    return {"research_results": research_results}

# 2. Drafter Agent (Node)
def draft_node(state: ResearchState) -> ResearchState:
    """
    Synthesizes the research results into a final answer.
    Updates the state with the final answer.
    """
    print("---DRAFTING ANSWER---")
    query = state["query"]
    research_results = state["research_results"]

    # Create a prompt for the drafter agent
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert answer synthesizer.
         Your goal is to provide a comprehensive and well-structured answer
         based on the research results provided.
         Carefully read the query and the research results.
         Synthesize the information into a clear, concise, and accurate answer.
         Cite sources or indicate where information came from if possible (e.g., based on search result 1, ...).
         If the research results don't contain enough information, state that."""),
        ("human", "Query: {query}\n\nResearch Results:\n{research_results}\n\nProvide the final answer:")
    ])

    # Create a chain for the drafter
    # The LLM here is now Google Gemini
    draft_chain = draft_prompt | llm

    # Invoke the chain with the state data
    final_answer = draft_chain.invoke({"query": query, "research_results": "\n---\n".join(research_results)})

    print("---ANSWER DRAFTED---")

    # Update state with the final answer
    return {"final_answer": final_answer.content}


# --- Build the LangGraph ---

# Define a new graph
workflow = StateGraph(ResearchState)

# Add nodes for each agent
workflow.add_node("researcher", research_node)
workflow.add_node("drafter", draft_node)

# Define the sequence of nodes: Researcher -> Drafter
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "drafter")

# The graph ends after the drafter node completes
workflow.add_edge("drafter", END)

# Compile the graph
app = workflow.compile()

# --- Run the System ---

if __name__ == "__main__":
    # Example Query
    research_query = "What are the key features of the latest Gemini models?" # Example query relevant to Gemini

    print(f"\n--- Starting Research for: {research_query} ---\n")

    # Run the graph with the initial state
    # Note: If you previously had an OpenAI quota issue, ensure your Google API key
    # has billing enabled and sufficient quota if using paid models/features.
    try:
        final_state = app.invoke({"query": research_query})

        print("\n--- Final Result ---")
        # The final state contains all the information gathered and synthesized
        print("Query:", final_state["query"])
        print("\nResearch Results (Snippets):")
        for i, result in enumerate(final_state["research_results"]):
            print(f"--- Snippet {i+1} ---")
            print(result)
            print("-" * 20)

        print("\nFinal Answer:")
        print(final_state["final_answer"])
        print("\n--- End ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your API keys, environment variables, and ensure you have sufficient quota/billing set up for both Tavily and Google API.")