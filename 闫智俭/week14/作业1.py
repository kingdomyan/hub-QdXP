import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 加载环境变量（存放API KEY）
load_dotenv()

# ===================== 1. 配置路径和模型 =====================
# 本地知识库文件路径（支持 .txt / .md / .pdf）
LOCAL_FILE_PATH = "local_knowledge.txt"  # 你可以改成自己的文件

# LLM模型配置（gpt-3.5-turbo 轻量快速）
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,  # 0=答案更精准、不发散
)

# 向量嵌入模型（用于把文本转成向量）
embeddings = OpenAIEmbeddings()

# ===================== 2. 加载本地文档 =====================
def load_local_document(file_path):
    """加载本地知识库文件"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        # 文本文件：txt/md 等
        loader = TextLoader(file_path, encoding="utf-8")
    
    documents = loader.load()
    print(f"✅ 文档加载完成，共 {len(documents)} 页/段")
    return documents

# ===================== 3. 文档分块（关键步骤） =====================
def split_documents(documents):
    """把长文档切成小块，提升检索精度"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 每块大小
        chunk_overlap=100, # 块之间重叠内容
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文档分块完成，共 {len(chunks)} 个文本块")
    return chunks

# ===================== 4. 构建本地向量库 =====================
def build_vector_db(chunks, embeddings):
    """基于FAISS构建本地轻量向量库"""
    db = FAISS.from_documents(chunks, embeddings)
    print("✅ 向量库构建完成")
    return db

# ===================== 5. 构建检索问答链 =====================
def build_qa_chain(llm, db):
    """构建 检索 + LLM回答 核心链"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 把检索内容直接塞进prompt
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # 检索最相关的3条
        return_source_documents=True  # 返回来源文档（方便查看）
    )
    return qa_chain

# ===================== 主程序：知识库问答 =====================
if __name__ == "__main__":
    # 1. 加载本地知识库
    docs = load_local_document(LOCAL_FILE_PATH)
    
    # 2. 切分文档
    doc_chunks = split_documents(docs)
    
    # 3. 构建向量库
    vector_db = build_vector_db(doc_chunks, embeddings)
    
    # 4. 构建问答链
    qa_chain = build_qa_chain(llm, vector_db)
    
    # 5. 开始问答
    print("\n===== 本地知识库问答系统（输入 exit 退出）=====")
    while True:
        query = input("\n请输入你的问题：")
        if query.lower() == "exit":
            print("程序退出！")
            break
        
        # 执行问答
        result = qa_chain.invoke({"query": query})
        
        # 输出答案
        print("\n【AI 回答】：")
        print(result["result"])
        
        # 可选：输出参考的原文片段
        print("\n【参考来源片段】：")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\n--- 片段 {i+1} ---")
            print(doc.page_content[:200] + "...")
