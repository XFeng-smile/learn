from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import requests
import re

# 加载环境变量
load_dotenv()


# ======================
# 1. 本地 LM Studio 模型
# ======================
def get_llm():
    llm = ChatOpenAI(
        model="local-model",
        base_url="http://localhost:1234/v1",
        # api_key="none",
        temperature=0.1,
    )
    return llm


# ======================
# 2. 天气查询工具
# ======================
def get_current_weather(city: str) -> str:
    """
    查询指定城市的实时天气。
    参数 city：城市名，如 南京、北京、上海
    """
    try:
        url = f"https://wttr.in/{city}?format=3"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            return f"【天气结果】{res.text.strip()}"
        else:
            return f"无法获取{city}天气"
    except Exception as e:
        return f"查询{city}天气失败: {str(e)}"


# ======================
# 3. 系统提示词：教模型如何使用工具
# ======================
SYSTEM_PROMPT = """
你是一个智能助手，可以帮助用户回答问题。
如果用户询问天气相关问题，你需要输出：`【调用天气工具】城市名`
例如：用户问“南京今天天气怎么样？”，你要输出：`【调用天气工具】南京`
其他问题请直接回答，不要输出额外格式。
"""

# ======================
# 4. 主程序
# ======================
if __name__ == '__main__':
    llm = get_llm()

    while True:
        user_input = input("\n请输入你的问题（输入exit退出）：")
        if user_input.lower() == "exit":
            break

        # 构造对话
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        # 调用模型
        response = llm.invoke(messages)
        model_reply = response.content.strip()
        print("\n模型回复：", model_reply)

        # 解析工具调用指令
        match = re.match(r"【调用天气工具】(.+)", model_reply)
        if match:
            city = match.group(1).strip()
            print(f"\n正在调用天气工具查询：{city}...")
            weather_result = get_current_weather(city)
            print(weather_result)

            # 把天气结果再传给模型，生成最终回答
            follow_up = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": model_reply},
                {"role": "user", "content": f"工具返回结果：{weather_result}，请根据结果回答用户问题"}
            ]
            final_response = llm.invoke(follow_up)
            print("\n最终回答：", final_response.content.strip())
