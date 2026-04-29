<h1 align="center">🫧 DermaLense by Flow</h1>


<p align="center">
 <b>복수의 뷰티 데이터를 통합한 RAG 기반 AI 스킨케어 의사결정 서비스</b><br>
 파편화된 성분 정보를 통합하고, 개인 피부에 맞는 제품을 데이터 기반으로 추천합니다.
</p>


<p align="center">
 <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
 <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
 <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
 <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
 <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" />
 <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
 <a href="https://www.notion.so/ohgiraffers/3RD_PROJECT-33d649136c11806eb17fda8ac18be5f4"><img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white" /></a>
</p>


---

## ✨ Key Features

- 🔍 **병렬 하이브리드 검색** — BM25 + Dense (HyDE) + RRF로 키워드·의미 맥락을 통합 반영
- 🤖 **질문 유형 자동 라우팅** — 성분 분석·제품 추천·일반 질의를 분류해 최적 검색 방식 적용
- ⚖️ **WoE 기반 이중 가중치** — 출처 신뢰도 × 성분 중요도 조합으로 결과 정렬
- 📸 **OCR 성분표 인식** — PaddleOCR로 화장품 패키지 촬영 시 성분 자동 추출·분석

---

## 🚀 Getting Started

> 프로젝트 루트 기준으로 실행합니다.

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env에 아래 키 입력:
# OPENAI_API_KEY=...
# COHERE_API_KEY=...
```

### 3. 데이터 파이프라인

```bash
python 03_scripts/01_validate_raw.py
python 03_scripts/02_make_dataset.py
python 03_scripts/03_build_features.py


# ⚠️ 02_src 폴더를 Mark Directory As → Sources Root 설정


# FAISS 인덱스 생성 (OpenAI API 비용 발생)
python 03_scripts/04_train.py --preset_id 1
python 03_scripts/04_train.py --preset_id 2
python 03_scripts/04_train.py --preset_id 3
python 03_scripts/04_train.py --preset_id 4
```

### 4. 서비스 실행

```bash
# 터미널 1 — FastAPI 백엔드
uvicorn api_server:app --reload


# 터미널 2 — Streamlit 프론트엔드
streamlit run streamlit_app.py
```

> ⚠️ **주의사항**
> - `04_train.py`는 OpenAI 임베딩 API를 호출하므로 비용이 발생합니다
> - PaddleOCR은 첫 실행 시 모델을 자동 다운로드합니다 (수 분 소요)
> - 4, 5단계는 터미널 두 개를 동시에 열어서 실행해야 합니다


---

## 🏗 시스템 아키텍처

### 전체 아키텍처

![Architecture](./06_assets/architecture.png)

### RAG 파이프라인

![RAG Pipeline](./06_assets/rag_pipeline_architecture.png)


---

## 🔬 Technical Deep Dive

<details>
<summary><b>📊 데이터 설계 & ERD</b></summary>

### ERD

![ERD](./06_assets/erd.png)

3개 소스(coos.kr, 화해, Paula's Choice)의 이종 데이터를 **CAS No. → INCI 영문명 → 표준 한글명 → 이명 사전** 순서의 계층적 매핑으로 표준화합니다.

| 테이블                | 설명            | 주요 컬럼                                                      |
|:-------------------|:--------------|:-----------------------------------------------------------|
| `ewg_chunk`        | 성분별 안전 등급·점수  | `ingredient_ko`, `hw_ewg`, `coos_score`, `pc_grade`        |
| `basic_info_chunk` | 기본 기능·카테고리    | `pc_effect`, `pc_category`, `coos_function`, `hw_purpose`  |
| `expert_chunk`     | 전문가 평가·국가별 기준 | `pc_description`, `coos_ai_desc`, `coos_KR`, `hw_category` |
| `product_chunk`    | 화장품 제품 DB     | 제품명, 성분 리스트, 브랜드, 카테고리                                     |

</details>


<details>
<summary><b>⚖️ WoE 기반 가중치 시스템</b></summary>


DermaLens는 3개 데이터 소스(coos.kr, 화해, Paula's Choice)의 이질적인 성분 안전성 등급을 통합하기 위해 **Weight of Evidence(WoE)** 방법론을 적용했습니다.

**핵심 공식:**

$$
W = -\ln\left(\frac{Dist_{Others}}{Dist_{Good}}\right)
$$

- **양수(+):** 해당 등급에 Good 성분이 밀집 → 추천 우선순위 상향
- **0 (중립):** 변별력 없음 → 추천 근거로 부적합
- **음수(−):** 해당 등급에 Others 성분이 밀집 → 추천에서 배제

**전처리 3단계:**

1. **데이터 범주화** — 성분을 기능·위험도 기준으로 구조화
2. **타겟 정의** — Good(고신뢰 성분) vs Others로 이진 분류
3. **결측치 처리** — Missing은 제외 (WoE = 0)

**사이트 신뢰도 가중치(Q값):**
관련성·타당성·신뢰성 3개 요인, 9개 문항(5점 척도)으로 6명이 평가한 설문 결과를 정규화하여 산출했습니다.

| 사이트            | 총점  | Q값            |
|----------------|-----|---------------|
| coos.kr        | 226 | 0.3419        |
| 화해             | 217 | 0.4989 (재정규화) |
| Paula's Choice | 218 | 0.5011 (재정규화) |

**Final Score 통합 공식:**


<p align="center">
 <img src="https://latex.codecogs.com/svg.image?Final%5C%20Score%20%3D%20%5Cunderbrace%7BV_%7Bcoos%7D%20%5Ctimes%20Q_%7Bcoos%7D%7D_%7BRegulatory%7D%20%2B%20%5Cunderbrace%7BWoE_%7Bhwahae%7D%20%5Ctimes%20Q_%7Bhwahae%7D%7D_%7BUser%5C%20Experience%7D%20%2B%20%5Cunderbrace%7BWoE_%7Bpaula%7D%20%5Ctimes%20Q_%7Bpaula%7D%7D_%7BExpert%7D" alt="Final Score formula"/>
</p>


최종 점수가 높을수록 안전하고 신뢰도 높은 성분이며, 낮을수록 주의·위험 성분으로 필터링됩니다. 이 구조는 임의 설계가 아닌 **설명 가능한 AI(XAI)** 원칙에 기반하여, 추천 이유를 사용자에게 자연어로 제공할
수 있습니다.


</details>


<details>
<summary><b>🔀 검색 모델 라우팅 & 실험 결과</b></summary>


LangGraph classify_node가 질문을 자동 분류한 뒤, NDCG@3 평가 기반으로 최적 검색 방식을 적용합니다.

| 질문 유형      | 선택 방식 | NDCG@3 | 예시                 |
|:-----------|:-----:|:------:|:-------------------|
| ingredient | Dense | 1.444  | "나이아신아마이드 EWG 등급?" |
| recommend  | BM25  | 1.822  | "지성 피부에 뭐 써?"      |
| general    | BM25  | 0.926  | "화장품 어떻게 보관?"      |

**FAISS 프리셋 (청크 가중치)**

|   프리셋   | EWG  | Basic Info | Expert | 용도                  |
|:-------:|:----:|:----------:|:------:|:--------------------|
| Preset1 | 0.33 |    0.33    |  0.33  | general (균등)        |
| Preset2 | 0.50 |    0.35    |  0.15  | ingredient (안전성 중심) |
| Preset3 | 0.40 |    0.45    |  0.15  | 안전성 + 기본정보 균형       |
| Preset4 | 0.45 |    0.45    |  0.10  | recommend           |

</details>


<details>
<summary><b>📁 폴더 구조</b></summary>

```
flow/
├── 00_data/
│   ├── 00_raw/          # 원본 데이터
│   ├── 01_interim/      # 전처리 중간 산출물
│   └── 02_processed/    # 최종 RAG용 청크
├── 01_notebooks/        # 실험 로그
├── 02_src/
│   ├── 00_common/       # 설정 로더, 로거
│   ├── 01_data/         # 데이터 파이프라인
│   ├── 02_model/        # FAISS 인덱싱, RAG 체인, 평가
│   └── 03_front/        # Streamlit UI
├── 03_scripts/          # 파이프라인 자동화
├── 04_configs/          # config.yaml
├── 05_artifacts/        # FAISS 인덱스
├── 06_assets/           # README 이미지
├── api_server.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

</details>


---

## 👥 Team Flow

**프로젝트 기간**: 2026.04.24 — 2026.04.27


<table>
<tr align="center">
<td><img src="https://i.namu.wiki/i/643jgTgLVQz0BSwMlgrrBHwReXJ19iRGS5bxDMLSxsPM4GkN-uOFsy6Pp9RiX7nEasn9WvHMLs09raOXZdp55UnSOTBlyHuBGlsFQUfmEKMihNHSeDJonYr23W2RjRrrLDY0wYrSCo3vvxYkZSFmVw.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/UNFQIgJYne_H9jkN5j24jyGY2laGmWrke_x3M-nEZkSD1J5wTNIRS7Wx_iJyGCYqcFMJ1aNHSn5HNlKF_8lM9_wR-zKUCdLHdDjRJ1Yn8X6nHJ9cOdwQP_obJfqsIVuIT4i90lSi1RpObI9txk28NQ.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/9ox-ZTFvJTp9NnfR7lfYejD5hQuBsARibzQva-1eZYOYFig3m4OrVnxdZXNhRdmOvHzjtC5jGb9P_IXejenqrWx6j6-kLwItI1oJE08p09mdCV2DPmhoTPPs4sOh1sdzg_GpB-koxeq_upE93UXs8w.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/4fCAU9Ybh2SEjBXvmxfujyqF8O1L7ErL8_wambUdwOsL9wOkoN_iQ9baUv1JV0xBRn33dxLfENat13pAQfDFgJ92IRe8ydxy91_YB9PRr_xehlgXqDZZZ8dtDpsoG69LeNDwvPLzvKYn8gywIlMxpQ.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/ZnpEAAI7med8T9czv4jHEO3F_SRO0vb-vuNnvRvONC898ryjJrEiG5vUAF_nuTApH9Fe2CDjEOEHq-kSIA1AvpStjcxh0h91B1iDVP3hM3QfeR7hj7K97FxKGJDiJpfGG6t6wSK5F4fbjbFPoKr8uA.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/4LeawTUtEIuFpBNrGYYmZDUfLflQiuQlvlU-sDR-BXgLVntn2krnY6XuBYPUgkOCEUqrdpoEHJqW2msV3JYWBTOAHoCFAYCAi7WW0tzSdO9uTbQJI2jLeUam-4O82pvIQ5Dnla5OvIqxb-njgjO2Uw.webp" width="100"></td>
</tr>
<tr align="center">
<td><b>김민하</b></td>
<td><b>배재현</b></td>
<td><b>윤지혜</b></td>
<td><b>전윤하</b></td>
<td><b>정다솔</b></td>
<td><b>홍진서</b></td>
</tr>
<tr align="center">
<td><a href="https://github.com/leedhroxx"><img src="https://img.shields.io/badge/GitHub-leedhroxx-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/rshyun24"><img src="https://img.shields.io/badge/GitHub-rshyun24-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/jjhhyy0926"><img src="https://img.shields.io/badge/GitHub-jjhhyy0926-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/yoonha315"><img src="https://img.shields.io/badge/GitHub-yoonha315-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/soll07"><img src="https://img.shields.io/badge/GitHub-soll07-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/Hong-Jin-seo"><img src="https://img.shields.io/badge/GitHub-Hong--Jin--seo-181717?style=flat-square&logo=github" /></a></td>
</tr>
<tr align="center">
<td>데이터 전처리<br>4가지 검색 병렬 실행<br>이중 가중치 재정렬<br>GitHub 총괄</td>
<td>Chunk 변환 & 임베딩<br>PPT 작성<br>기획 및 발표</td>
<td>데이터 수집<br>Langchain & RAGChain<br>Streamlit UI</td>
<td>Chunk 변환 & 임베딩<br>README 작성</td>
<td>데이터 수집<br>Langchain & RAGChain<br>Streamlit UI</td>
<td>데이터 전처리<br>4가지 검색 병렬 실행<br>이중 가중치 재정렬<br>Notion 총괄</td>
</tr>
</table>


---

## 💬 팀원 회고

### 👤 김민하

| 평가자 | 회고 내용 |
|--------|-----------|
| 정다솔 | 데이터 크롤링 초안을 잘 잡아주어서 화해 데이터 크롤링을 하는데 많은 도움이 되었고, 데이터 표준화 작업의 복잡함을 감당하면서도 팀 전체 방향을 잃지 않고 이끌어준 것이 든든했습니다. GitHub 총괄을 하면서 팀 전체 PR을 빠짐없이 리뷰하고 팀 전체 코드 흐름을 가장 잘 파악하고 있어서 막히는 부분이 생겼을 때 방향을 잡아주는 역할을 해줬습니다. |
| 홍진서 | 작업 과정에서 약간의 출력물의 문제가 있었지만 바로 다른 방법을 찾아 해결하였습니다. 막힌 부분이나 놓친 부분을 바로 캐치하여 작업 시간을 줄여주었습니다. GitHub 담당으로 pull request를 보내면 짧은 시간 안에 merge를 해주어 다음 작업을 빠르게 들어갈 수 있게 해주었습니다. |
| 배재현 | 특유의 결단력으로 회의가 산으로 가는 것을 방지하며, 팀의 작업이 효율적으로 이뤄질 수 있게 하였다. 철저한 시간관리를 통해 프로젝트를 기한 내에 끝낼 수 있게 하였으며, 깃 이슈 관리, 대본 수정, 파일 구조 변경 등 자칫 놓치기 쉬운 세세한 부분까지 꼼꼼히 챙겨주어 팀 전체의 완성도를 높이는 데 중요한 역할을 하였다. |
| 윤지혜 | BM25, Dense, RRF, HyDE 4가지 검색 방식을 병렬로 구현하고 Precision@3, Recall@3, MRR, NDCG 지표로 성능을 체계적으로 비교할 수 있는 evaluator를 설계한 점이 좋았습니다. 특히 팀 전체가 놓치기 쉬운 부분을 짚어주는 피드백을 꾸준히 해줘서, 설계 단계에서 미처 고려하지 못했던 엣지 케이스나 개선 포인트를 사전에 잡을 수 있었습니다. 꼼꼼한 시각이 결과물의 완성도를 높이는 데 많은 도움이 되었습니다. |
| 전윤하 | 책임감 있게 맡은 역할을 끝까지 완수했으며, 전 단계에서 문제 상황이 생겼을 때 빠르게 공유해 줘서 팀이 신속하게 수정 대응할 수 있었습니다. |

---

### 👤 배재현

| 평가자 | 회고 내용 |
|--------|-----------|
| 정다솔 | 프로젝트 초반에 크롤링부터 시작해 기획안까지 작성을 하며 프로젝트 전반적인 흐름을 만들어주었고, 임베딩 모델을 바꿔야 하는 상황에서도 흔들리지 않고 묵묵히 재구축해준 덕분에 전체 일정이 밀리지 않았습니다. OCR 전처리 부분에서 막혔을 때 방향을 잡아줘서 빠르게 해결할 수 있었습니다. |
| 홍진서 | 전체적인 작업의 기반을 만들어주고, 작업 흐름을 컨트롤하여 어떤 작업을 해야 하는지 알기 쉽게 해주었습니다. 자체 설문지를 만들어서 가중치 관련 작업을 해주었습니다. 작업 과정에서 문제가 있을 때 나서서 해결해주었습니다. 완벽한 발표를 했습니다. |
| 김민하 | 가히 PM이라고 할 만큼의 능력을 가졌습니다. 기획부터 발표까지 손대지 않은 것이 없었고, 회의 때도 책임감을 가지고 열정적으로 임했습니다. |
| 윤지혜 | 전처리부터 청크 변환, 임베딩까지 파이프라인 초반 흐름 전체를 담당하며 프로젝트의 구조적인 기반을 잡아줬습니다. 특히 데이터 가중치 설계에 필요한 자료를 직접 찾아오고 관련 논문까지 검토해서 팀에 공유해준 덕분에, 근거 있는 가중치 설계가 가능했습니다. 회의록도 꾸준히 작성해줘서 팀원들이 논의 내용을 놓치지 않고 작업을 이어갈 수 있었습니다. 발표도 내용을 군더더기 없이 정리해서 깔끔하게 전달해줘서 팀 결과물을 잘 대표해줬습니다. 프로젝트 시작 전, 전체 워크플로우를 큰 그림으로 파악하고 방향을 잡아준 덕분에 팀 전체가 흔들리지 않고 진행할 수 있었습니다. |
| 전윤하 | 프로젝트 전반을 적극적으로 리드하며 방향을 잘 잡아줬고, PPT 기획부터 발표까지 완성도를 높이는 데 크게 기여했습니다. |

---

### 👤 윤지혜

| 평가자 | 회고 내용 |
|--------|-----------|
| 정다솔 | 프로젝트 진행하면서 여러 의견을 많이 제시하고 RAG 파이프라인 전체를 책임지고 평가 코드까지 직접 짠 것이 팀 결과물의 신뢰성을 높이고 API 서버에서 연결하는 작업이 수월했습니다. 단순한 챗봇이 아니라 실제 서비스처럼 느껴지는 결과물이 나왔습니다. |
| 홍진서 | 프로젝트 초반 여러 좋은 의견을 내어 프로젝트 방향을 잡을 수 있었습니다. 중요할 수 있는 LangChain, RAG 부분에서 우여곡절이 많았지만 끝까지 해내면서 우리가 원하는 결과물을 내주었습니다. 배운 기술들을 잘 적용하면서 구현 기술과 사용 근거 등을 일목요연하게 정리하였습니다. |
| 배재현 | 초기 기획단계부터 열정적으로 의견을 내며, 팀의 초반 방향성과 분위기를 잡는 데 큰 역할을 하였다. RAG 파트를 맡아 성능을 끌어올리고, 준비된 자료들이 모두 활용될 수 있도록 만들어냈다. 발표 직전까지 RAG에 대해 설명해주며, 발표 준비를 도와주는 모습이 인상적이었다. |
| 김민하 | 프로젝트 초반 수립단계에 큰 틀을 짜두어 체계적으로 진행을 할 수 있었고, 맡은 업무가 많았음에도 불구하고 꼼꼼하게 처리했습니다. |
| 전윤하 | 맡은 바를 성실하게 수행했고, 묵묵히 자기 역할을 해내며 팀이 안정적으로 돌아갈 수 있도록 해줬습니다. |

---

### 👤 전윤하

| 평가자 | 회고 내용 |
|--------|-----------|
| 정다솔 | 청크 구조를 일관성 있게 잡아줘서 검색 결과 파싱이 훨씬 수월했습니다. 전체 모듈 주석 정리와 README 작업 덕분에 팀 전체가 같은 그림을 공유할 수 있도록 문서화에도 신경 썼습니다. |
| 홍진서 | 임베딩 과정에서 실수가 있었는데 바로 수정 작업을 거치면서 책임감을 느낄 수 있었습니다. GitHub의 README를 담당하여 좋은 README를 생성하였습니다. 회의록을 작성하여 우리가 놓칠 수 있는 부분이 없게 하였습니다. |
| 배재현 | 임베딩 파트와 모듈화 작업을 함께하며, 자신의 일에 충실하였다. README 작성을 자발적으로 맡아 프로젝트 전반의 내용을 체계적으로 정리하였고, 끝까지 자신이 할 일이 있는지 확인하며 열정적으로 해주었다. |
| 김민하 | 데이터 처리와 README 작성을 해주어 프로젝트의 시작과 끝을 맡아주었습니다. |
| 윤지혜 | 데이터 임베딩 작업을 담당하며 파이프라인 초반 구성에 기여했습니다. 기술적인 작업 외에도 README를 작성해 프로젝트 전체 구조와 실행 방법을 문서화해줬고, 회의록을 꾸준히 정리해서 팀원들이 논의 내용을 놓치지 않고 확인할 수 있는 환경을 만들어줬습니다. 또한 맡은 작업을 성실하게 마무리해 주어 팀 전체 일정과 흐름이 무리 없이 이어질 수 있었습니다. |

---

### 👤 정다솔

| 평가자 | 회고 내용 |
|--------|-----------|
| 홍진서 | 좋은 화면 구현을 해야 하는데 그 부분에서 여러 난관이 있었지만 끝까지 포기하지 않고 팀원 모두가 만족하는 결과물을 만들어주었습니다. OCR 같은 외부 기술을 도입하여 화면 구현에 풍부함을 더해주었습니다. |
| 배재현 | 프로젝트 초반 데이터 수집과 프론트 초안을 제시하여 전체적인 설계 틀을 잡을 수 있었다. 기획 의도에 맞춰 Streamlit 디자인 및 모듈화 작업을 진행하고, 문서, 코드, ERD 전반을 꼼꼼히 검토하며 필요한 부분을 적시에 수정 및 보완했다. 결과물 하나하나에 높은 기준을 두고 끝까지 완성도를 놓지 않는 태도가 돋보였다. |
| 김민하 | 맡은 일을 성실하고 꼼꼼하게 체계적으로 수행하고, 프로젝트를 진행하는 데에 있어서 필수적인 인물입니다. 여러 툴을 다양하게 사용하여 기술 스택을 넓히는 것이 멋있었습니다. |
| 윤지혜 | Streamlit UI를 단순 기능 구현에 그치지 않고 사용자 편의성을 고려한 설계로 완성해줬습니다. OCR 파트는 이미지에서 텍스트를 정확히 인식하는 것 자체가 기술적으로 까다로운 부분이었는데, 글자 인식 품질 문제를 직접 파고들어 해결해줘서 이미지 업로드 기반 성분 분석 기능이 실제로 동작할 수 있었습니다. 7, 8단계를 함께 작업하면서 워크플로우를 어떻게 나눌지 고민이 많았는데, 전체 흐름을 먼저 정리하고 역할 분담을 명확하게 잡아줘서 협업하기 편했습니다. 어떤 작업이든 믿고 의지할 수 있었고, 모르는 부분이 생겼을 때도 편하게 질문할 수 있어서 함께 작업하면서 많이 배울 수 있었습니다. |
| 전윤하 | Streamlit 프론트엔드를 잘 잡아줬고, 디자인 감각이 좋아서 UI 완성도를 높이는 데 큰 역할을 했습니다. |

---

### 👤 홍진서

| 평가자 | 회고 내용 |
|--------|-----------|
| 정다솔 | 데이터 전처리 과정에서 꼼꼼하게 이상치를 잡아줬고, 가중치 로직의 복잡도를 낮춰준 결정이 결과적으로 API 응답 결과를 해석하기도 쉽게 만들었습니다. Notion 정리해줘서 놓친 결정사항을 나중에 확인할 수 있었습니다. |
| 배재현 | 팀의 분위기메이커로서 의견 충돌이 생길 수 있는 순간마다 적절하게 중재하며 팀 내 소통이 원활하게 이어질 수 있도록 하였다. 핵심 파트인 이중 가중치 작업과, PPT 작업을 통해 발표의 흐름을 잡아 주는 것을 성실히 도왔다. 덕분에 팀 전체가 좋은 분위기 속에서 프로젝트를 마무리할 수 있었고, 결과물의 완성도까지 챙길 수 있었다. |
| 김민하 | 데이터 전처리 등 초반에 필요한 작업을 잘 해주었고, 맡은 일을 주중 주말 상관 없이 해내는 점이 인상깊었습니다. 또한 팀 내 소통역할의 중심을 도맡아주어 분위기 형성에 큰 역할을 해주었습니다. |
| 윤지혜 | 이중 가중치 재정렬 파이프라인을 chunk_weight × source_weight 조합으로 설계해 청크 유형과 데이터 출처를 동시에 반영하는 구조를 구현했고, Contextual Compression까지 연결해 GPT 입력 품질을 높이는 흐름을 모듈화한 점이 인상적이었습니다. 기술적인 기여뿐만 아니라, 팀 작업 전반에서 분위기를 편안하게 만들어 주셔서 막히는 부분이 생겼을 때 부담 없이 의견을 나눌 수 있는 환경을 만들어 주신 점도 프로젝트 진행에 큰 도움이 되었습니다. 특히 작업 중 나눈 대화에서 제가 놓치고 있던 부분을 짚어 주셔서, 프로젝트를 더 나은 방향으로 보완해 나갈 수 있었습니다. |
| 전윤하 | Notion 정리를 통해 프로젝트 흐름을 체계적으로 문서화해 줘서, 팀원들이 진행 상황을 한눈에 파악하고 효율적으로 협업할 수 있었습니다. |


---

## 📚 데이터 출처 & 라이선스

| 소스                                                                   | 제공 정보                    |
|:---------------------------------------------------------------------|:-------------------------|
| [coos.kr](https://coos.kr)                                           | 성분별 기능, 국가별 규제, AI 설명    |
| [화해](https://www.hwahae.co.kr)                                       | EWG 수치, 사용자 리뷰 토픽, 피부 타입 |
| [Paula's Choice](https://www.paulaschoice.com/ingredient-dictionary) | 전문가 평가 및 논문 근거           |

본 프로젝트는 [MIT License](./LICENSE) 하에 배포됩니다.

