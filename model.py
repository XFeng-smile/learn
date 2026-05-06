from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


# ======================
# 1. 本地 LM Studio 模型
# ======================
def get_model():
    model = ChatOpenAI(
        model="local-model",
        base_url="http://localhost:1234/v1",
        # api_key="none",
        temperature=0.1,
    )
    return model

