from rag.vectorstore_loader import load_vectorstore
import os

# 打印工作目录
print("当前工作目录:", os.getcwd())

# 加载向量数据库
vectorstore = load_vectorstore()

# 检查文档数量
print("数据库中文档数量:", vectorstore._collection.count())

# 直接相似性搜索
results = vectorstore.similarity_search("上海有哪些推荐的博物馆？", k=2)
print("直接搜索结果数量:", len(results))
for i, res in enumerate(results):
    print(f"结果 {i+1}:", res.page_content, res.metadata)

# 测试检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("上海有哪些推荐的博物馆？")
print("返回对象类型:", type(docs))
print("检索器结果数量:", len(docs))
for i, doc in enumerate(docs, 1):
    print(f"--- 第 {i} 条 ---")
    print("内容:", doc.page_content)
    print("元数据:", doc.metadata)