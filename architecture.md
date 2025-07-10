---


# RAG-Powered First-Aid Chatbot – Architecture

---
## Overview

The **RAG-Powered First-Aid Chatbot** is designed to provide emergency first-aid assistance for **diabetes, cardiac, and renal** conditions. It uses a multi-layered Retrieval-Augmented Generation (RAG) pipeline to process queries, classify symptoms, retrieve relevant knowledge, and generate structured, cited responses.

Below is a high-level architectural diagram:

<details>
<summary>View Mermaid Diagram </summary>

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
    'fontSize': '13px',
    'primaryTextColor': '#2c3e50',
    'primaryBorderColor': '#3498db',
    'primaryColor': '#ecf0f1',
    'lineColor': '#34495e'
  }
}}%%
graph TB
    %% Main Flow
    User[User Input] --> MainBot[FirstAidChatbot]

    %% Core Pipeline
    MainBot --> Step1[Step 1: Triage Classification]
    MainBot --> Step2[Step 2: Multi-Source Retrieval]
    MainBot --> Step3[Step 3: Result Fusion]
    MainBot --> Step4[Step 4: Response Generation]

    %% Triage Components
    Step1 --> GroqLLM[Groq LLM]
    Step1 --> Keywords[Keyword Fallback]

    %% Retrieval Components
    Step2 --> Semantic[Semantic Search]
    Step2 --> Keyword[Keyword Search]
    Step2 --> Web[Web Search]

    %% Knowledge Base
    Semantic --> KB[Knowledge Base]
    Keyword --> KB
    KB --> Data[(Excel Data)]
    KB --> Embeddings[(Vector Embeddings)]

    %% External APIs
    Web --> SerperAPI[Serper API]
    GroqLLM --> GroqAPI[Groq API]

    %% Final Processing
    Semantic --> Step3
    Keyword --> Step3
    Web --> Step3
    Step3 --> Step4
    Step4 --> Output[Final Response]

    %% Styling
    classDef primary fill:#2980b9,stroke:#1f4e79,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef process fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#ffffff,font-weight:500
    classDef data fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff,font-weight:500
    classDef api fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff,font-weight:500
    classDef accent fill:#9b59b6,stroke:#7d3c98,stroke-width:2px,color:#ffffff,font-weight:500

    class User,MainBot,Output primary
    class Step1,Step2,Step3,Step4 process
    class KB,Data,Embeddings data
    class GroqAPI,SerperAPI,GroqLLM,Web api
    class Semantic,Keyword,Keywords accent
 ```
 </details>

 ---

## Processing Steps

### Step 1: Triage Classification
- Normalize and analyze symptoms using Groq LLM
- Fallback to keyword rules if LLM fails
- Outputs: domain, condition, confidence, urgency

### Step 2: Multi-Source Retrieval
- Run semantic and keyword search on local Excel KB
- Execute real-time web search via Serper API
- Collect relevant passages and metadata

### Step 3: Result Fusion
- Aggregate results from all retrieval sources
- Apply weighted scoring and remove duplicates
- Rank by relevance and diversity

### Step 4: Response Generation
- Send fused data to Groq API for structured guidance
- Insert medical disclaimers and citations
- Produce final first-aid response (≤ 250 words)

---


## Component Overview

### 1. FirstAidChatbot (Main Controller)
**Role:** Central orchestrator managing the query processing pipeline  
**Responsibilities:**
- Receive and preprocess queries
- Invoke triage, retrieval, fusion, and generation steps sequentially
- Handle fallbacks and error recovery

### 2. Triage Classification (`LLMTriageClassifier`)
**Primary:** Groq LLM for symptom classification and urgency scoring  
**Fallback:** Keyword-based pattern matching for core medical terms  
**Outputs:**
- `domain`: diabetes, cardiac, or renal
- `condition`: specific subtype (e.g., Hyperglycemia, Myocardial Infarction)
- `confidence`: 0–1 score
- `urgency`: low, medium, high

### 3. Knowledge Retrieval (`MedicalKnowledgeBase`)
**Semantic Search:** SentenceTransformer embeddings for similarity matching  
**Keyword Search:** Exact term matching on Excel repository  
**Data Source:** `.xlsx` file organized by domain and condition

### 4. External Knowledge (`WebSearcher`)
**API:** Serper for real-time web information  
**Enhancement:** Adds domain context to queries  
**Mock Mode:** Returns placeholder results if API fails

### 5. Result Fusion and Ranking
**Algorithm:**
1. Aggregate results from semantic, keyword, and web sources
2. Score each result by weighted relevance (configurable)
3. Deduplicate based on title/text overlap
4. Sort by final relevance score

### 6. Response Generation (`GroqLLMClient`)
**Model:** Groq LLM-llama3-70b-8192  
**Template Output:** Structured JSON-like response with:
- Summary of the condition
- Step-by-step first-aid advice (≤ 250 words)
- Citations and disclaimers  
**Fallback:** Static templates on API failure

---

## Data Flow

1. User Input → normalization & safety checks  
2. Triage Classification → domain, condition, confidence, urgency  
3. Knowledge Retrieval → semantic + keyword + web results  
4. Result Fusion & Ranking → merged, scored, and sorted results  
5. Response Generation → final structured output  

---

## Design Trade-offs

- **Local vs. External:** Balances latency (local KB) with freshness (web search)
- **Determinism vs. Flexibility:** Ensures minimum response coverage through keyword fallbacks

---

---


