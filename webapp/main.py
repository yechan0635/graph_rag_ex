import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ✅ webapp/.env 로드
load_dotenv(override=True)

app = FastAPI(title="Movie GraphRAG API")

# -----------------------------
# 1) 요청/응답 스키마 (Swagger 예쁘게)
# -----------------------------
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

# -----------------------------
# 2) 전역 초기화(앱 시작 시 1회)
# -----------------------------
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    enhanced_schema=True
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Neo4jVector.from_existing_index(
    embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="movie_content_embeddings",
    text_node_property="overview",
)

# -----------------------------
# 3) GraphRAG 유틸 함수들
# -----------------------------
def get_movie_details_and_actors(movie_titles):
    query = """
    MATCH (m:Movie)
    WHERE ANY(t IN $titles WHERE m.title CONTAINS t)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    RETURN 
        m.title as title, 
        m.released as released, 
        m.rating as rating, 
        m.overview as overview,
        collect(a.name) as actor_names,
        collect(elementId(a)) as actor_ids
    """
    return graph.query(query, params={"titles": movie_titles})

def get_actor_filmography(actor_ids, exclude_titles):
    if not actor_ids:
        return []

    query = """
    MATCH (a:Person)
    WHERE elementId(a) IN $actor_ids
    MATCH (a)-[:ACTED_IN]->(m:Movie)
    WHERE NOT m.title IN $exclude_titles
    RETURN 
        a.name as actor_name, 
        collect({title: m.title, released: m.released}) as other_movies
    """
    return graph.query(query, params={"actor_ids": actor_ids, "exclude_titles": exclude_titles})

def format_context_for_llm(movies, filmographies):
    context_parts = ["## 검색된 영화 정보"]

    for m in movies:
        actors = ", ".join(m["actor_names"]) if m["actor_names"] else "정보 없음"
        overview = m["overview"] or ""
        context_parts.append(
            f"- 영화: {m['title']} ({m['released']})\n"
            f"  평점: {m['rating']}\n"
            f"  배우: {actors}\n"
            f"  줄거리: {overview[:100]}..."
        )

    if filmographies:
        context_parts.append("\n## 출연 배우의 다른 작품들")
        for f in filmographies:
            titles = [f"{x['title']}({x['released']})" for x in f["other_movies"][:3]]
            context_parts.append(f"- {f['actor_name']}: {', '.join(titles)}")

    return "\n".join(context_parts)

def movie_graph_search_orchestrator(user_query: str) -> str:
    docs = vector_store.similarity_search(user_query, k=3)
    found_titles = [d.metadata.get("title") for d in docs if d.metadata.get("title")]

    if not found_titles:
        return "관련 정보를 찾을 수 없습니다."

    movie_data = get_movie_details_and_actors(found_titles)

    all_actor_ids = []
    for m in movie_data:
        all_actor_ids.extend(m["actor_ids"])
    all_actor_ids = list(set(all_actor_ids))

    film_data = get_actor_filmography(all_actor_ids, found_titles)
    return format_context_for_llm(movie_data, film_data)

# -----------------------------
# 4) LangChain RAG 체인 1회 구성
# -----------------------------
template = """당신은 영화 추천 전문가로서 오직 주어진 정보에 기반하여 객관적이고 정확한 답변을 제공합니다.

[주어진 영화 정보]
{context}

[질문]
{question}

# 답변 작성 지침:
1. 제공된 영화 정보에 명시된 사실만 사용하세요.
2. 간결하고 정확하게 답변하세요.
3. 제공된 정보에 없는 내용은 "제공된 정보에서 해당 내용을 찾을 수 없습니다"라고 답하세요.
4. 영화의 제목, 평점 등 주요 정보를 포함해서 답변하세요.
5. 한국어로 자연스럽고 이해하기 쉽게 답변하세요.
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

rag_input = {
    "context": RunnableLambda(movie_graph_search_orchestrator),
    "question": RunnablePassthrough(),
}

graph_rag_chain = rag_input | prompt | llm | StrOutputParser()

# -----------------------------
# 5) API 엔드포인트
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QueryRequest):
    answer = graph_rag_chain.invoke(req.query)
    return {"answer": answer}