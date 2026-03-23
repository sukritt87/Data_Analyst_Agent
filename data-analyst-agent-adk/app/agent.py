from dotenv import load_dotenv
from app.tool_wrappers import (
    get_columns,
    get_summary_statistics,
    get_average_sales_by_region,
    get_correlation_matrix,
    plot_sales_by_region,
    run_sales_regression,
    run_churn_classification,
)

load_dotenv()

# Placeholder structure for your ADK-based agent wiring.
# We are keeping this file simple so you can adapt it to the exact ADK version installed.

SYSTEM_INSTRUCTION = """
You are an AI Data Analyst Agent.
You must answer user questions by selecting and using the available tools.
Do not invent dataset information.
Use tool outputs as the source of truth.
If a question is outside the supported tools, say so clearly.
"""

AVAILABLE_TOOLS = [
    get_columns,
    get_summary_statistics,
    get_average_sales_by_region,
    get_correlation_matrix,
    plot_sales_by_region,
    run_sales_regression,
    run_churn_classification,
]