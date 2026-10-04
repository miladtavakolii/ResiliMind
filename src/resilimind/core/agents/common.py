import logging
import networkx as nx

from ...graph.ingestion import load_resilience_graph
from ...llm.engine import LLMEngine

# Initialize module logger
logger = logging.getLogger(__name__)

# Initialize LLM Engine singleton and load graph into memory once
llm_engine: LLMEngine = LLMEngine()
resilience_graph: nx.DiGraph = load_resilience_graph()
