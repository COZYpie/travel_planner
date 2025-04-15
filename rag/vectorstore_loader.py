from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Azure OpenAI 嵌入函数配置
embedding = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_key="6mVs9WqyFfLs72iUitCKF7IfncDRQ0utAQtdovHwtxlac6HsB1CWJQQJ99BDACfhMk5XJ3w3AAAAACOGHoZR",
    azure_endpoint="https://13067-m9egtpio-swedencentral.cognitiveservices.azure.com/",
    openai_api_version="2023-05-15",
    openai_api_type="azure"
)

# 调试嵌入函数
print("嵌入函数配置:", embedding)
print("测试嵌入:", embedding.embed_query("测试一下向量生成")[:10])  # 打印前10个值以验证

def load_vectorstore():
    # 设置持久化目录（与 build_vectorstore.py 一致）
    PERSIST_DIRECTORY = "C:\\Users\\13067\\PycharmProjects\\travel_planner\\vectordb"
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding,
        collection_name="places"  # 显式指定集合名称
    )