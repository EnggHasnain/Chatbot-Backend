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
        "You are a friendly tutor for the textbook 'Physical AI & Humanoid Robotics'.\n\n"
        "BEHAVIOR:\n"
        "1. GREETINGS: Respond warmly to greetings (hi, hello, etc.) and introduce yourself briefly.\n"
        "2. WITH CONTEXT: Answer questions using the provided context. Be concise and helpful.\n"
        "3. NO CONTEXT: If context is empty or doesn't answer the question, explain what topics you CAN help with:\n"
        "   - Physical AI fundamentals\n"
        "   - Humanoid robotics\n"
        "   - ROS 2 and robot programming\n"
        "   - Sensors, actuators, and control systems\n"
        "   - AI/ML for robotics\n"
        "4. GENERAL QUERIES: For 'summarize everything' type questions, give a brief overview of the book's scope.\n\n"
        "Always be helpful and guide users toward questions you can answer from the textbook."
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
