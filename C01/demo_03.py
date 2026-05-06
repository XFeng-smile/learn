from langchain.agents import create_agent
from langchain_core.tools import tool
from model import get_model
import datetime

# 第一个工具：两个数字相加
@tool
def add_two_numbers(a, b):
    """
    将两个数字相加
    :param a:
    :param b:
    :return:
    """
    return a + b

# 第二个工具：获取当前时间
@tool
def get_current_time():
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent = create_agent(
    model=get_model(),
    tools=[get_current_time, add_two_numbers],
    system_prompt="你是一个有用的助手。"
)

result = agent.invoke(input={"messages":[{"role":"user", "content":"1+2等于?"}]})
print(result["messages"][-1].content)