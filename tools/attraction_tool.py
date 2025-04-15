from rag.vectorstore_loader import load_vectorstore
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from config import OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_VERSION, DEPLOYMENT_NAME

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def recommend_attractions(query: str) -> str:
    try:
        result = qa_chain({"query": query})
        attractions = []
        for doc in result["source_documents"]:
            name = doc.metadata.get("name", "")
            if name:
                description = doc.page_content.split(" - ")[-1] if " - " in doc.page_content else doc.page_content
                attractions.append(f"{name}：{description}")
        if not attractions:
            return "暂无景点推荐，请检查知识库或查询内容。"
        return "。".join(attractions)
    except Exception as e:
        return f"查询景点推荐失败: {str(e)}"