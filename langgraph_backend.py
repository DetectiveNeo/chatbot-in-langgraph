from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

# Add the chat node to the graph
graph.add_node("chat_node", chat_node)

# Add edges to the graph
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile the graph into a callable function
chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Example usage
    user_message = "What is my name ?"

    config1 = {"configurable": {"thread_id": "1"}}

    response = chatbot.invoke({"messages": [user_message]}, config= config1)
    print(response['messages'][-1].content)



