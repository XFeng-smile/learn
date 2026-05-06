from langchain_openai import ChatOpenAI
# 用于测试
from dotenv import load_dotenv
import os

from sympy import true

load_dotenv()  # 默认加载当前目录下的 .env 文件


def get_llm():
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V4-Flash",
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("api_key"),
        temperature=0.1,
    )
    return llm


if __name__ == '__main__':
    model = get_llm()
    print(model.invoke("hello world"))
    # print(os.getenv("api_key"))

    res = [HumanMessage(content='现在是几点', additional_kwargs={}, response_metadata={},
                        id='dc177c44-116c-4b93-8649-40f28e141209'),
           AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={
               'token_usage': {'completion_tokens': 11, 'prompt_tokens': 56, 'total_tokens': 67,
                               'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None,
                                                             'reasoning_tokens': 0, 'rejected_prediction_tokens': None},
                               'prompt_tokens_details': None}, 'model_provider': 'openai',
               'model_name': 'gemma-4-e4b-it', 'system_fingerprint': 'gemma-4-e4b-it',
               'id': 'chatcmpl-wxe0gdg76j66k8evngkss', 'finish_reason': 'tool_calls', 'logprobs': None},
                     id='lc_run--019dfc7c-16e8-7780-b24a-d8e2f644597a-0',
                     tool_calls=[{'name': 'get_current_time', 'args': {}, 'id': '829326647', 'type': 'tool_call'}],
                     invalid_tool_calls=[], usage_metadata={'input_tokens': 56, 'output_tokens': 11, 'total_tokens': 67,
                                                            'input_token_details': {},
                                                            'output_token_details': {'reasoning': 0}}),
           ToolMessage(content='20260506165124', name='get_current_time', id='14669b1e-a379-40dd-8874-c245736f1c3d',
                       tool_call_id='829326647'),
           AIMessage(content='现在是 2026年5月6日，下午 4 点 51 分 24 秒。', additional_kwargs={'refusal': None},
                     response_metadata={
                         'token_usage': {'completion_tokens': 27, 'prompt_tokens': 93, 'total_tokens': 120,
                                         'completion_tokens_details': {'accepted_prediction_tokens': None,
                                                                       'audio_tokens': None, 'reasoning_tokens': 0,
                                                                       'rejected_prediction_tokens': None},
                                         'prompt_tokens_details': None}, 'model_provider': 'openai',
                         'model_name': 'gemma-4-e4b-it', 'system_fingerprint': 'gemma-4-e4b-it',
                         'id': 'chatcmpl-imum2qk7skl8dmbz3e3ygb', 'finish_reason': 'stop', 'logprobs': None},
                     id='lc_run--019dfc7c-1927-7163-a1b9-fe7cc0fa97bc-0', tool_calls=[], invalid_tool_calls=[],
                     usage_metadata={'input_tokens': 93, 'output_tokens': 27, 'total_tokens': 120,
                                     'input_token_details': {}, 'output_token_details': {'reasoning': 0}})]
