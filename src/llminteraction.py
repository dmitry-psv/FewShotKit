import os
import openai
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def build_icl_prompt(examples, new_dialogue, max_examples=5, preprocess_func=None):
    """
    Creates an optimal prompt for Instruct-type models (e.g., OpenAI text-davinci, Mistral).

    Parameters:
    ----------
    examples : List[Dict[str, str]]
        List of demonstration examples:
          {
            'dialog': '...',  # dialog text
            'summary': '...'  # final summary
          }
    new_dialogue : str
        New dialog for which summary needs to be generated.
    max_examples : int
        Maximum number of demonstration examples.
    preprocess_func : Callable[[str], str], optional
        Text preprocessing function (if not specified, uses `strip`).

    Returns:
    ----------
    str
        Optimized prompt for Instruct models.
    """
    if not isinstance(examples, list):
        raise ValueError("`examples` must be list of dictionaries.")
    if not isinstance(new_dialogue, str):
        raise ValueError("`new_dialogue` must be a string.")
    if preprocess_func is None:
        preprocess_func = lambda x: x.strip()

    # Crop examples list to max_examples
    used_examples = examples[:max_examples]

    # If list of examples is empty, use another prompt
    if not used_examples:
        new_task_text = f"{os.environ.get('INPUT_PREFIX')}{preprocess_func(new_dialogue)}\n{os.environ.get('OUTPUT_PREFIX')}"
        return os.environ.get("EMPTY_TASK_DESCRIPTION") + new_task_text

    # Create examples text
    examples_text = []
    for i, ex in enumerate(used_examples, 1):
        dialog = preprocess_func(ex.get('x_label', ''))
        summary = preprocess_func(ex.get('y_label', ''))

        if not dialog or not summary:
            continue  # Ignore empty examples

        examples_text.append(f"### Example {i} ###\n{os.environ.get('INPUT_PREFIX')}{dialog}\n{os.environ.get('OUTPUT_PREFIX')}{summary}\n")

    # Process the new dialogue
    new_dialogue_processed = preprocess_func(new_dialogue)
    new_task_text = f"{os.environ.get('INPUT_PREFIX')}{new_dialogue_processed}\n{os.environ.get('OUTPUT_PREFIX')}"

    # Final prompt
    prompt = os.environ.get("TASK_DESCRIPTION") + "\n\n".join(examples_text) + "\n\n" + "### Task ###\n" + new_task_text
    return prompt

def call_openrouter_llm(
    prompt: str,
    model: str,
    openrouter_api_key: str = None,
    referer: str = "https://my.site.example",
    site_name: str = "MySiteOnOpenRouter",
    temperature: float = 0.3
) -> str:
    """
    Requests a response from a model hosted on OpenRouter using the 'openai' library.
    """

    # 1. Если ключ явно не указан, берем из переменной окружения
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key not found. Please provide api_key or set OPENROUTER_API_KEY in env.")

    # 2. Устанавливаем API ключ и базовый URL для OpenRouter
    openai.api_key = api_key
    openai.api_base = "https://openrouter.ai/api/v1"

    # 3. Формируем запрос на chat.completions 
    headers = {
        "HTTP-Referer": referer,
        "X-Title": site_name
    }

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        headers=headers
    )

    # 4. Извлекаем контент ответа
    return completion['choices'][0]['message']['content']

def get_available_models(api_key: str = None):
    """
    Fetch the list of available models from OpenRouter API.

    Args:
        api_key (str): Your OpenRouter API key.

    Returns:
        list: A list of available models or an error message if the request fails.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key not found. Please provide api_key or set OPENROUTER_API_KEY in env.")

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP error responses
        models = response.json()  # Parse the JSON response
        return models
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

