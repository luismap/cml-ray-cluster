memory_prompt_template = """<s>[INST] <<SYS>> your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>
{history}
 {input} [/INST]
"""
from typing import List, Tuple
from langchain import PromptTemplate

memory_prompt = PromptTemplate.from_template(memory_prompt_template)

chat_memory = []


user_info = {"default": {"get_history": "false", "history": chat_memory}}

users = {"default"}


import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


from langchain_community.llms import VLLM

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

llm = VLLM(
    model=model_id,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.2,
    tensor_parallel_size=2,
    dtype="half",
)


def add_user(user_name: str):
    if user_name not in users:
        chat_rag_history = []
        user_info[user_name] = {"get_history": "false", "history": chat_rag_history}

        users.add(user_name)
        return True
    else:
        return False


def generate_history(history: List[Tuple[str, str]]) -> str:
    """
    given a list of tuples with values question and answer
    generate a str for use as history for llama model

    Args:
        history (List[Tuple[str, str]]): history tuple

    Returns:
        str: history formatted
    """
    combined = [q + " [/INST] " + a + "</s><s>[INST]" for q, a in history]
    return "\n".join(combined)


@timeit
def call_llm(payloads: List[str], user_id: str = "default"):
    if user_id not in users:
        add_user(user_id)

    user_hist = user_info[user_id]["history"]
    history_g = generate_history(user_hist)
    questions = []
    for q in payloads:
        question_formatted = memory_prompt.format(history=history_g, input=q)
        questions.append(question_formatted)

    answers = llm.batch(questions)

    record = zip(payloads, answers)

    for q, a in record:
        user_info[user_id]["history"].append((q, a))
        print(f"QUESTION: {q}")
        print(f"ANSWER: {a}\n")


def get_history(user_id: str = "default"):
    print("\n".join(map(lambda a: a[0] + " " + a[1], user_info[user_id]["history"])))


def clean_user_history(user_id: str) -> bool:
    if user_id in users:
        user_info[user_id]["history"] = []
        return True
    else:  # TODO add exception
        return False


def chat(user_id: str = "default"):
    q = input("ask me someting: (exit type :q)")
    while q != ":q":
        call_llm([q], user_id=user_id)
        q = input("ask me someting: (exit type :q)")
