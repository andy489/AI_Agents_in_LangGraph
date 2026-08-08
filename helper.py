import os
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from typing import TypedDict

_ = load_dotenv()

PLAN_PROMPT = (
    "You are an expert writer tasked with writing a high level outline of an essay. "
    "Write such an outline for the user provided topic. Give an outline of the essay "
    "along with any relevant notes or instructions for the sections."
)

WRITER_PROMPT = (
    "You are an essay assistant tasked with writing excellent 5-paragraph essays. "
    "Generate the best essay possible for the user's request and the initial outline. "
    "If the user provides critique, respond with a revised version of your previous attempts. "
    "Utilize all the information below as needed:\n\n------\n\n{content}"
)

REFLECTION_PROMPT = (
    "You are a teacher grading an essay submission. "
    "Generate critique and recommendations for the user's submission. "
    "Provide detailed recommendations, including requests for length, depth, style, etc."
)

RESEARCH_PLAN_PROMPT = (
    "You are a researcher charged with providing information that can be used when writing "
    "the following essay. Generate a list of search queries that will gather any relevant "
    "information. Only generate 3 queries max."
)

RESEARCH_CRITIQUE_PROMPT = (
    "You are a researcher charged with providing information that can be used when making "
    "any requested revisions (as outlined below). Generate a list of search queries that "
    "will gather any relevant information. Only generate 3 queries max."
)


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: Optional[List[str]]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


class ewriter:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def _research_plan_node(self, state: AgentState):
        queries = self.model.with_structured_output(
            Queries, method="function_calling"
        ).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ])
        content = state.get("content", [])
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {"content": content}

    def _generation_node(self, state: AgentState):
        content = "\n\n".join(state.get("content", []))
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        )
        messages = [
            SystemMessage(content=WRITER_PROMPT.format(content=content)),
            user_message,
        ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    def _reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state["draft"]),
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def _research_critique_node(self, state: AgentState):
        queries = self.model.with_structured_output(
            Queries, method="function_calling"
        ).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["critique"]),
        ])
        content = state.get("content", [])
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {"content": content}

    def _should_continue(self, state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self._plan_node)
        builder.add_node("generate", self._generation_node)
        builder.add_node("reflect", self._reflection_node)
        builder.add_node("research_plan", self._research_plan_node)
        builder.add_node("research_critique", self._research_critique_node)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_conditional_edges(
            "generate",
            self._should_continue,
            {END: END, "reflect": "reflect"},
        )
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        return builder.compile(checkpointer=self.memory)


def writer_gui(graph):
    import gradio as gr
    import uuid

    def run_essay(topic, max_revisions):
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_draft = ""
        for s in graph.stream(
            {"task": topic, "max_revisions": int(max_revisions), "revision_number": 1},
            thread,
        ):
            node = list(s.keys())[0]
            state = s[node]
            if "draft" in state:
                final_draft = state["draft"]
        return final_draft

    with gr.Blocks() as app:
        gr.Markdown("## Essay Writer")
        with gr.Row():
            topic = gr.Textbox(label="Essay Topic", placeholder="Enter a topic...")
            max_rev = gr.Slider(1, 5, value=2, step=1, label="Max Revisions")
        btn = gr.Button("Write Essay")
        output = gr.Textbox(label="Essay", lines=20)
        btn.click(run_essay, inputs=[topic, max_rev], outputs=output)

    return app
