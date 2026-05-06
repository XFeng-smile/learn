from langchain.agents import create_agent
import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from model import get_model


# 使用pydantic定义参数schema
class Args(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")


@tool
def get_current_time():
    """
    获取当前时间
    :return: 格式化的时间
    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


@tool
def add_two_numbers(a: int, b: int):
    """
    计算两个数字相加之和
    :param a:
    :param b:
    :return:
    """
    return a + b


agent = create_agent(
    model=get_model(),
    tools=[get_current_time, add_two_numbers],
    system_prompt="你是一个智能助手"
)

res = agent.invoke(input={"messages": [{"role": "user", "content": "1+1的值是多少？"}]})

print(res['messages'][-1].content)
# print(res)
