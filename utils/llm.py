import ollama
from ollama import chat, web_fetch, web_search

available_tools = {'web_search': web_search, 'web_fetch': web_fetch}

def query_llm_stream(prompt, model="llama3:latest"):
    """
    Stream LLM output from Ollama token-by-token.
    Yields text chunks as they are generated.
    """
    # Use Ollama streaming
    for response in ollama.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        stream=True  # important for incremental output
    ):
        # response.message may be list or single object
        if isinstance(response.message, list):
            for msg in response.message:
                if msg.content:  # skip empty chunks
                    yield msg.content
        else:
            if response.message.content:
                yield response.message.content