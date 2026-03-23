from google.adk.agents.llm_agent import Agent

from app.tool_wrappers import (
    get_columns,
    get_summary_statistics,
    get_average_sales_by_region,
    get_correlation_matrix,
    plot_sales_by_region,
    run_sales_regression,
    run_churn_classification,
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="data_analyst_agent",
    description="A tool-using AI data analyst agent for small CSV datasets.",
    instruction=(
        "You are an AI Data Analyst Agent. "
        "Always use the available tools to answer dataset questions. "
        "Do not invent dataset facts. "
        "Use tool outputs as the source of truth. "
        "If the user asks for columns or schema, use the schema tool. "
        "If the user asks for summaries or averages, use the EDA tools. "
        "If the user asks for plots, use the visualization tool. "
        "If the user asks for prediction or churn analysis, use the ML tools. "
        "If the question is outside the supported tools, say so clearly."
    ),
    tools=[
        get_columns,
        get_summary_statistics,
        get_average_sales_by_region,
        get_correlation_matrix,
        plot_sales_by_region,
        run_sales_regression,
        run_churn_classification,
    ],
)