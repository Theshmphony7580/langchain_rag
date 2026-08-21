"""Logfire configuration - import FIRST in main.py and agent.py before any other imports."""
import os
from dotenv import load_dotenv
load_dotenv()

import logfire

# Enable LangSmith OTel export (routes LangChain/LangGraph traces to Logfire)
os.environ.setdefault('LANGSMITH_OTEL_ENABLED', 'true')
os.environ.setdefault('LANGSMITH_OTEL_ONLY', 'true')
os.environ.setdefault('LANGSMITH_TRACING', 'true')

# Configure Logfire (reads LOGFIRE_TOKEN from env)
logfire.configure(service_name='langchain-rag')

# System metrics (CPU, memory)
logfire.instrument_system_metrics()

# Standard logging -> Logfire
import logging
logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()], level=logging.INFO)

# HTTP client tracing (load_doc.py uses requests)
logfire.instrument_requests()