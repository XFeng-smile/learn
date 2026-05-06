from langchain.agents import create_agent
import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pymupdf import message
from sympy import content


@tool
def get_current_time():
    """
    获取当前时间
    :return: 格式化的时间
    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


from model import get_model

agent = create_agent(
    model=get_model(),
    tools=[get_current_time],
    system_prompt="你是一个时间助手"
)
res = agent.invoke({
    "messages": [HumanMessage(content="现在是几点")]
})
print(res['messages'][-1].content)
print(res)
