from google.adk.agents.llm_agent import Agent

from app.tool_wrappers import (
    list_uploaded_artifacts,
    inspect_dataset,
    get_columns,
    get_data_types,
    get_missing_values,
    get_summary_statistics,
    get_categorical_summary,
    get_correlation_matrix,
    groupby_aggregate,
    generate_bar_chart,
    generate_histogram,
    generate_scatter_plot,
    run_linear_regression,
    run_random_forest_classification,
)

root_agent = Agent(
    model="gemini-3.1-flash-lite-preview",
    name="data_analyst_agent",
    description="A strict tool-using AI data analyst agent for CSV datasets.",
    instruction=(
        "You are a STRICT AI Data Analyst Agent.\n\n"
        "MANDATORY RULES:\n"
        "1. You MUST ALWAYS use tools for any dataset-related question.\n"
        "2. You are NOT allowed to guess column names, schema, statistics, or insights.\n"
        "3. You are NOT allowed to answer dataset questions from memory.\n"
        "4. If the user asks about a dataset, first use a tool such as inspect_dataset, get_columns, or list_uploaded_artifacts.\n"
        "5. Use uploaded CSV files if available in the current session.\n"
        "6. If no uploaded CSV exists, use the fallback sample dataset.\n"
        "7. Only say that no dataset is available if both uploaded artifacts and the fallback dataset are unavailable.\n\n"
        "WORKFLOW:\n"
        "- First inspect the dataset or list uploaded artifacts.\n"
        "- Then choose the appropriate analysis tool.\n"
        "- Answer only from tool outputs.\n"
    ),
    tools=[
        list_uploaded_artifacts,
        inspect_dataset,
        get_columns,
        get_data_types,
        get_missing_values,
        get_summary_statistics,
        get_categorical_summary,
        get_correlation_matrix,
        groupby_aggregate,
        generate_bar_chart,
        generate_histogram,
        generate_scatter_plot,
        run_linear_regression,
        run_random_forest_classification,
    ],
)