import os
import hashlib
import math
import uuid
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

# 读取 .env 文件中的环境变量
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

st.set_page_config(
    page_title="智能知识库问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 面向企业文档的智能知识库问答系统")

st.write("这是一个基于 RAG 的文档问答系统，支持上传 PDF、TXT、Markdown 文档。")

st.sidebar.title("⚙️ 检索参数设置")

top_k = st.sidebar.slider(
    "检索参考文本块数量 Top K",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

st.sidebar.divider()

st.sidebar.title("📌 项目说明")

st.sidebar.markdown(
    """
    **面向企业文档的智能知识库问答系统**

    本项目基于 RAG 技术实现文档智能问答，支持用户上传 PDF、TXT、Markdown 文档，
    系统会自动完成文本提取、文本切分、向量化检索，并调用大语言模型生成回答。
    """
)

st.sidebar.title("🧩 技术栈")

st.sidebar.markdown(
    """
    - Python
    - Streamlit
    - LangChain
    - Chroma 向量数据库
    - DeepSeek API
    - RAG 检索增强生成
    """
)

st.sidebar.title("🔁 系统流程")

st.sidebar.markdown(
    """
    1. 上传知识库文档  
    2. 提取文档文本  
    3. 文本切分 Chunk  
    4. 构建向量数据库  
    5. 检索相关文本块  
    6. 调用大模型生成回答  
    7. 展示参考来源  
    """
)

st.sidebar.title("🎯 适用场景")

st.sidebar.markdown(
    """
    - 企业制度文档问答
    - 毕业论文内容问答
    - 产品说明书问答
    - 学习资料智能检索
    - 项目文档知识库助手
    """
)

st.sidebar.info(
    "明确问题建议选择 1-2；综合分析类问题建议选择 3-5。"
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


class SimpleHashEmbeddings(Embeddings):
    """
    轻量级本地向量化方法：
    不需要下载模型，也不消耗 API 额度。
    适合项目初期演示 RAG 向量检索流程。
    后期可替换为 BGE、M3E、OpenAI Embedding、通义 Embedding 等更专业的 Embedding 模型。
    """

    def __init__(self, dim=384):
        self.dim = dim

    def _tokenize(self, text):
        text = text.strip()
        chars = list(text)
        bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        return chars + bigrams

    def _text_to_vector(self, text):
        vector = [0.0] * self.dim
        tokens = self._tokenize(text)

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def embed_documents(self, texts):
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text):
        return self._text_to_vector(text)


def extract_text_from_pdf(file_path):
    """读取 PDF 文件中的文本内容"""
    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_txt(file_path):
    """读取 TXT / Markdown 文件中的文本内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path):
    """根据文件类型选择不同的读取方式"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        return extract_text_from_txt(file_path)
    else:
        return ""


def split_text(text, file_name):
    """将长文本切分成适合检索的小块，并保留来源文件名"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    documents = []

    for index, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "file_name": file_name,
                "chunk_index": index + 1
            }
        )
        documents.append(doc)

    return documents


def build_vector_store(documents):
    """构建 Chroma 向量数据库"""
    embeddings = SimpleHashEmbeddings()

    # 每次运行都创建新的临时 collection，避免 Streamlit 刷新后重复写入相同文本块
    collection_name = f"knowledge_base_{uuid.uuid4().hex}"

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vector_store


def retrieve_relevant_docs(question, vector_store, top_k=3):
    """
    使用 Chroma 向量检索 + 关键词加权检索，并去除重复结果。
    这样可以提升中文文档问答中对“导师、专业、摘要、创新点”等明确问题的命中率。
    """
    # 先多取一些候选文本块，方便后面重新排序
    results = vector_store.similarity_search_with_score(question, k=top_k * 5)

    reranked_results = []

    # 常见论文/文档问答关键词
    important_keywords = [
        "导师", "指导教师", "教授", "致谢",
        "专业", "学校", "学院", "姓名",
        "摘要", "关键词", "创新", "研究背景",
        "系统", "功能", "测试", "结论"
    ]

    for doc, distance in results:
        content = doc.page_content

        # Chroma 的 score 通常是距离，距离越小越相关
        # 这里先把距离转成基础得分
        score = 1 / (1 + distance)

        # 如果问题中的字或词出现在文本块中，加分
        for char in set(question):
            if char in content:
                score += 0.05

        # 如果重要关键词同时出现在问题和文本块中，额外加分
        for keyword in important_keywords:
            if keyword in question and keyword in content:
                score += 2

        # 如果问题包含“导师”，文本块包含“余雷”或“指导教师”，重点加分
        if "导师" in question or "指导教授" in question or "指导教师" in question:
            if "余雷" in content or "指导教师" in content or "教授" in content:
                score += 5

        reranked_results.append((doc, distance, score))

    # 按综合得分从高到低排序
    reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)

    unique_results = []
    seen = set()

    for doc, distance, score in reranked_results:
        file_name = doc.metadata.get("file_name", "")
        chunk_index = doc.metadata.get("chunk_index", "")
        content_fingerprint = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

        key = f"{file_name}_{chunk_index}_{content_fingerprint}"

        if key not in seen:
            seen.add(key)
            unique_results.append((doc, distance))

        if len(unique_results) >= top_k:
            break

    return unique_results

def generate_answer(question, relevant_docs):
    """调用 DeepSeek，根据检索到的文本块生成回答"""

    if not API_KEY:
        return "未检测到 API Key，请先在 .env 文件中配置 OPENAI_API_KEY。"

    context = ""

    for doc, score in relevant_docs:
        file_name = doc.metadata.get("file_name", "未知文件")
        chunk_index = doc.metadata.get("chunk_index", "未知文本块")

        context += (
            f"【来源文件：{file_name}，文本块：{chunk_index}】\n"
            f"{doc.page_content}\n\n"
        )

    prompt = f"""
你是一个严谨的企业文档知识库问答助手。

你的任务是：只根据【参考资料】回答【用户问题】。

请严格遵守以下规则：
1. 必须优先根据参考资料作答，不要凭空编造。
2. 如果参考资料中能直接回答问题，请先给出明确结论。
3. 如果用户问“几个、多少、有哪些、创新点、功能、原因”等问题，请尽量用分点形式回答。
4. 如果可以判断数量，请明确写出“共 X 个”。
5. 如果参考资料中没有相关内容，请只回答：“根据当前文档资料无法确定。”
6. 回答要简洁、准确，避免无关扩展。

【参考资料】
{context}

【用户问题】
{question}

请用中文回答。
"""

    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content



uploaded_files = st.file_uploader(
    "请上传知识库文档",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

all_text = ""
all_documents = []

if uploaded_files:
    st.success(f"已上传 {len(uploaded_files)} 个文件")

    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        st.write(f"文件名：{file.name}")
        st.write(f"文件类型：{file.type}")
        st.write(f"文件大小：{round(file.size / 1024, 2)} KB")
        st.info(f"文件已保存到：{file_path}")

        text = extract_text(file_path)
        all_text += text + "\n"

        documents = split_text(text, file.name)
        all_documents.extend(documents)

    st.subheader("📄 文档内容预览")

    if all_text.strip():
        st.text_area(
            "以下是从文档中提取到的部分文本：",
            all_text[:1000],
            height=300
        )

        st.success(f"文本提取成功，共提取 {len(all_text)} 个字符")

        st.subheader("✂️ 文本切分结果")
        st.success(f"文本切分成功，共切分为 {len(all_documents)} 个文本块")

        with st.expander("查看前 3 个文本块"):
            for i, doc in enumerate(all_documents[:3]):
                st.markdown(
                    f"**文本块 {i + 1}：来源文件《{doc.metadata['file_name']}》，文件内文本块 {doc.metadata['chunk_index']}**"
                )
                st.write(doc.page_content)
                st.divider()

        st.subheader("🧠 向量数据库构建")

        with st.spinner("正在构建 Chroma 向量数据库..."):
            vector_store = build_vector_store(all_documents)

        st.success("Chroma 向量数据库构建成功")

        question = st.text_input("请输入你的问题：")

        if question:
            st.subheader("🔍 向量检索结果")

            relevant_docs = retrieve_relevant_docs(question, vector_store, top_k=top_k)

            if relevant_docs:
                st.success(f"已检索到 {len(relevant_docs)} 个相关文本块")

                st.subheader("🤖 AI 回答")

                with st.spinner("DeepSeek 正在根据文档生成回答..."):
                    try:
                        answer = generate_answer(question, relevant_docs)
                        st.write(answer)
                    except Exception as e:
                        st.error("调用大模型时出错，请检查 API Key、网络或 base_url 配置。")
                        st.code(str(e))

                st.subheader("📌 参考来源")

                for i, (doc, score) in enumerate(relevant_docs):
                    file_name = doc.metadata.get("file_name", "未知文件")
                    chunk_index = doc.metadata.get("chunk_index", "未知文本块")

                    with st.expander(
                        f"参考来源 {i + 1}：{file_name} - 文本块 {chunk_index}，向量距离：{round(score, 4)}"
                    ):
                        st.write(doc.page_content)

            else:
                st.warning("没有检索到相关内容，请换一个问题试试。")

    else:
        st.warning("没有提取到文本。这个 PDF 可能是扫描版图片，需要 OCR 才能识别。")
else:
    st.info("请先上传文档，然后开始构建知识库。")