import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig

# Get API key from environment variable
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key.strip(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=llm_model,
    model_provider=external_client,
    tracing_disabled=True,
)

tutor_agent = Agent(
    name="Physical AI Tutor",
    instructions=(
        "You are a tutor for the textbook 'Physical AI & Humanoid Robotics'. "
        "You MUST answer ONLY using the provided CONTEXT. "
        "If the context does not contain the answer, say: "
        "\"This information is not found in the textbook.\""
    ),
    model=llm_model,
)


async def run_agent(question: str, context: str) -> str:
    """
    Build a prompt with CONTEXT + QUESTION and run the agent.
    Return the final_output string.
    """
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER (short and clear, based only on the context):"
    )

    # IMPORTANT: this is awaited and returns a result object, not a coroutine
    result = await Runner.run(
        starting_agent=tutor_agent,
        input=prompt,
        run_config=config,
    )

    # Make sure this is a plain string
    return result.final_output
