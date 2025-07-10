# RAG-Powered First-Aid Chatbot – System Architecture

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
</details>

## Processing Steps

### Step 1: Triage Classification
- Analyzes symptoms using **Groq LLM**.
- Falls back to **keyword rules** when LLM confidence is low.
- **Outputs**:
  - `domain`: diabetes, cardiac, or renal
  - `condition`: e.g., Hyperglycemia, Myocardial Infarction
  - `confidence`: score between 0 and 1
  - `urgency`: low, medium, or high

### Step 2: Multi-Source Retrieval
- Semantic search via **SentenceTransformer embeddings**.
- Keyword search over structured **Excel knowledge base**.
- Web search using **Serper API**.
- Collects relevant medical passages.

### Step 3: Result Fusion
- Aggregates data from all sources.
- Applies **scoring weights**.
- Removes duplicates and ranks results.

### Step 4: Response Generation
- Uses **Groq LLM** to generate final structured output.
- **Includes**:
  - Medical summary
  - First-aid steps (≤ 250 words)
  - Citations and disclaimers
- Fallback to static template if API fails.

---

## Component Details

### 1. FirstAidChatbot (Main Controller)
- Orchestrates the entire pipeline.
- Handles:
  - Query intake and preprocessing
  - Step-by-step execution of pipeline
  - Failover handling

### 2. LLMTriageClassifier
- **Primary**: Groq LLM for domain and urgency classification.
- **Fallback**: Keyword pattern-matching.
- **Output**:
  - `domain`, `condition`, `confidence`, `urgency`

### 3. MedicalKnowledgeBase
- Semantic search with **transformer-based embeddings**.
- Keyword search over Excel file.
- Data source: `.xlsx` organized by domain-condition.

### 4. WebSearcher
- Uses **Serper API** for live web results.
- Augments queries with domain context.
- Mock fallback on API failure.

### 5. Result Fusion Engine
- Aggregates results from all sources.
- **Scoring algorithm**:
  - Apply configurable weights
  - Deduplicate similar text entries
  - Sort by final score

### 6. GroqLLMClient
- **Model**: Llama3-70b-8192 via Groq API
- **Template**:
  - Condition summary
  - First-aid guidance (≤ 250 words)
  - Source citations + medical disclaimer
- Fallback template if API fails

---

## Data Flow
1. User Input → normalization & safety checks
2. Triage Classification → domain, condition, confidence, urgency
3. Knowledge Retrieval → semantic + keyword + web results
4. Result Fusion & Ranking → merged, scored, and sorted
5. Response Generation → final structured output
6. Output → JSON response to user

