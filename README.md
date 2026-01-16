# graph DB(neo4j) 활용 영화 추천
## graph RAG 소개
- Graph 데이터베이스를 기반으로 한 질의 응답 시스템(GraphRAG)은 전통적인 벡터 기반 RAG 시스템보다 더 정확하고 연관성 있는 답변을 제공함. 
- 자연어 질문을 Neo4j Cypher 쿼리로 변환하여 지식 그래프를 효과적으로 탐색함.

### 특장점:
- 정확한 관계 검색: 그래프 데이터베이스의 관계 중심 구조를 활용해 복잡한 연결 패턴을 찾을 수 있음.
- 컨텍스트 유지: 엔티티 간의 관계를 유지하여 더 풍부한 컨텍스트를 제공함.
구조화된 정보 검색: 단순 텍스트 검색이 아닌 구조화된 방식으로 정보를 검색함.


# 설치 라이브러리
```
pip install langchain, langchain-neo4j, langchain-openai
pip install pandas
```
# 실습 설명
## 실습1
- 파일 : 1.neo4j_movie_graphdb구축_csv.ipynb
### 내용
- 목적 : CSV 영화 데이터를 **지식 그래프(Movie/Person/Genre + 관계)**로 변환해 Neo4j에 적재
- 입력/출력
    - 입력 : datas/movies_tmdb_v1.csv
    - 출력 : Neo4j에 (:Movie)-[:ACTED_IN|DIRECTED|IN_GENRE]->(:Person|:Genre) 그래프 생성

- 핵심 로직
    1. .env 로드 후 Neo4jGraph 연결
    2. DB 초기화(노드/관계 삭제 + 제약/인덱스 정리)
    3. 제약조건/인덱스 생성 (중복 방지, 검색 성능)
    4. CSV 로드 → Row 단위로 Node/Relationship 생성(중복 방지 dict 사용)
    5. GraphDocument(nodes, relationships) 생성
    6. graph.add_graph_documents()로 일괄 적재

- 핵심 포인트
    - movie_id_unique, person_name_unique, genre_name_unique 제약으로 데이터 무결성 확보
    - 대용량 대비 batch 처리로 변환 수행

- 실행 방법
    1. NEO4J_URI/USERNAME/PASSWORD/DATABASE 설정
    2. 노트북 위에서부터 실행

## 실습2
- 파일 : 2.neo4j_movie_basic_search.ipynb
### 내용
- 목적: Cypher로 기본 통계 + 관계 기반 분석/추천 연습

- 입력/출력
    - 입력 : Neo4j에 적재된 영화 그래프
    - 출력 : 평점 Top-N, 장르별 개수, 배우 출연 수, 유사 영화 추천 등

- 핵심 로직
    1. Neo4jGraph 연결
    2. Cypher로 단순 집계(영화 수, 평점 Top10 등)
    3. 관계 탐색(배우-영화-공동출연, 배우-감독 협업)
    4. 추천(같은 장르/평점 기준 필터 등)

- 대표 쿼리 예시(개념)
    - 평점 상위 영화: MATCH (m:Movie) WHERE m.rating IS NOT NULL RETURN ... ORDER BY m.rating DESC LIMIT 10
    - 공동 출연자: (actor)-[:ACTED_IN]->(m)<-[:ACTED_IN]-(co_actor)

- 개선 포인트
    - 추천은 단순 규칙 기반이라 개인화/의도 반영 한계 → 이후 Vector/RAG로 확장

## 실습3
- 파일 : 3.neo4j_movie_full-text_search.ipynb
### 내용
- 목적: Neo4j FULLTEXT INDEX로 키워드 검색 후 그래프 탐색으로 확장

- 입력/출력
    - 입력 : 검색어(예: love, star, "love story", 퍼지검색 등)
    - 출력 : 검색된 영화/인물 + 연관 배우/작품 목록

- 핵심 로직
    1. FULLTEXT INDEX 생성 (movie_title_fulltext, movie_title_tagline_fulltext, person_name_fulltext)
    2. CALL db.index.fulltext.queryNodes()로 검색
    3. 퍼지/와일드카드/구문/불리언/가중치 검색 실험
    4. 검색 결과 노드를 시작점으로 ACTED_IN 등 그래프 탐색 결합

- 대표 쿼리 예시(개념)
    - CALL db.index.fulltext.queryNodes("movie_title_fulltext",$term) YIELD node, score RETURN ...

- 개선 포인트
    - 텍스트 매칭 중심 → 의미 기반(semantic) 질의는 Vector Search로 보완

## 실습4
- 파일 : 4.neo4j_movie_vector_search.ipynb
### 내용
- 목적 : OpenAI Embedding으로 영화 설명을 벡터화하고 Neo4j VECTOR INDEX 구성

- 입력/출력
    - 입력 : Movie.title/tagline/overview
    - 출력 : m.content_embedding 벡터 속성 + movie_content_embeddings 인덱스

- 핵심 로직
    1. OpenAIEmbeddings(text-embedding-3-small) 초기화
    2. CREATE VECTOR INDEX ... FOR (m:Movie) ON m.content_embedding
    3. 영화 텍스트(title+tagline+overview) 결합
    4. 배치로 embeddings.embed_documents() 생성
    5. db.create.setNodeVectorProperty()로 노드에 벡터 저장

- 주의 포인트
    - vector.dimensions는 임베딩 모델 차원(예: 1536)과 반드시 일치해야 함
    - 배치 크기(BATCH_SIZE) 크게 잡으면 API/메모리 문제 가능

## 실습5
- 파일 : 5.neo4j_movie_text2cypher.ipynb
### 내용
- 목적 : LangChain GraphCypherQAChain으로 자연어 질의를 Cypher로 변환해 결과를 얻기

- 입력/출력
    - 입력: 사용자 자연어 질문
    - 출력: (1) 생성된 Cypher (2) 실행 결과 (3) 최종 답변(옵션)

- 핵심 로직
    1. Neo4jGraph(enhanced_schema=True)로 스키마를 LLM에 제공
    2. GraphCypherQAChain.from_llm() 구성
    3. .invoke({"query": ...})로 질의 → Cypher 생성/실행
    4. 옵션: top_k, return_intermediate_steps, return_direct

- 확장
    - Cypher 생성 LLM(OpenAI) + 답변 LLM(Gemini) 분리 가능
    - 커스텀 프롬프트로 “스키마 밖 속성/관계 사용 금지” 강화 가능

- 주의
    - allow_dangerous_requests=True는 학습용으로만(운영에선 제한 필요)

## 실습6
- 파일 : 6.neo4j_movie_graphVector_RAG.ipynb
### 내용
- 목적 : Neo4jVector로 의미 검색 후, LCEL로 RAG 체인을 구성

- 입력/출력
    - 입력 : 자연어 질의(예: “2차 세계대전 배경 영화”)
    - 출력 : 유사 영화 목록 + (옵션) 배우/필모그래피 확장 + LLM 답변

- 핵심 로직
    1. Neo4jVector.from_existing_index2.(index_name="movie_content_embeddings") 연결
    2. similarity_search_with_score(k=5)로 검색
    3. max_marginal_relevance_search()로 다양성 검색(MMR)
    4. (확장) 검색된 영화 → Cypher로 배우/필모그래피 탐색
    5. LCEL(Runnable)로 retriever → prompt → llm → parser 구성

- 개선 포인트
    - “의도(고평점/특정 시대/장르)”는 벡터만으로 부족 → 그래프 필터(정렬/조건) 결합 필요

## 실습7
- 파일 : 7.neo4j_movie_graphRAG_hybrid.ipynb

### Neo4j 기반 하이브리드 RAG
- Vector RAG + Graph RAG를 활용한 서비스 구현하기

