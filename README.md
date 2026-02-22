# MARCURA TARIFF SYSTEM
AI-Driven Port Tariff Ingestion & Port Dues Calculation Platform

## Introduction

Marcura Tariff System is an AI-enabled tariff intelligence platform designed to ingest complex Port Tariff PDF documents and compute applicable port dues for vessels in a deterministic, explainable, and production-ready manner. <br><br>
The system transforms unstructured regulatory tariff documents into a reliable financial calculation engine while preserving policy context and interpretability.
This solution leverages:
-	Hybrid Structured + Semantic Architecture
-	Deterministic financial computation
-	Guardrail-enforced LLM usage
-	RESTful API integration <br><br>

The result is a scalable and extensible system ensuring financial accuracy, transparency, and adaptability.<br>
## ARCHITECTURAL OVERVIEW
The system is intentionally designed with strict separation of responsibilities:
-	Structured financial data is stored deterministically.
-	Policy rules and semantic context are stored in a vector database.
-	LLM usage is confined to ingestion and explanation generation.
-	All financial calculations are deterministic and independent of LLM runtime behavior.

## High Level Design
![Architecture Diagram](docs/architecture.png)
## DESIGN PRINCIPLES
The system is built on the following engineering principles:
-	Deterministic Financial Accuracy
-	No financial calculation depends on LLM output at runtime.
-	Separation of Structured and Semantic Knowledge
-	Numeric data and policy logic are stored independently.
-	Controlled and Guarded LLM Usage
-	LLM calls are constrained, validated, and schema-enforced.
-	Ensures structured sequencing of ingestion, validation, and retrieval.
-	Production-Ready Extensibility

## API DESIGN
The system exposes two primary REST APIs
1. INGEST API <br>
Endpoint: POST /ingest<br>
Purpose:
Ingests a Port Tariff PDF document and processes it into structured and semantic knowledge stores.<br>
Input Parameter:<br>
pdf_path (string): File system path to the Port Tariff PDF document<br><br>
Processing Steps:
-	Reads the PDF document
-	Splits into logical tariff sections
-	Performs dual LLM extraction:
-	Extract numeric rates, tiers, fees and stores in SQLite
-	Extract rules, exemptions, and conditions and stores as embeddings in ChromaDB<br><br>
Output
-	Number of sections processed
-	Number of structured records extracted
-	Confirmation of successful ingestion

2.CALCULATE API <br>

Endpoint: POST /calculate <br>
Purpose:
Calculates applicable port dues for a vessel using previously ingested tariff data.<br><br>
Input Options:

Option A : Structured JSON Payload<br>
- port
-	gross_tonnage
-	arrival_date
-	departure_date
-	optional operational flags
-	include_explanation (boolean)

Option B : Natural Language Query<br>
Example:
"A 51,300 GT vessel arriving in Durban for 3 days, calculate all applicable dues."<br>

The system:
-	Parses natural language query (if provided)
-	Converts it into structured vessel parameters
-	Validates input using guardrails
-	Performs deterministic lookup from SQLite
-	Executes calculation engine
-	Optionally retrieves contextual rules from ChromaDB
-	Generates grounded explanation if requested

Output:<br>
-	Breakdown of all due types
-	Aggregated total
-	Optional explanation with semantic grounding

## PREREQUISITES
1.	Required Software<br>
•	Python 3.10 or higher<br><br>
2.	Required API Key
- Groq API Key (Required for ingestion and natural language queries)

## SETUP
- 1.	Step 1: Clone or Download the Project
Place the project folder in your desired directory, for example:
C:\Users\YourName\Downloads\pts\
Open PowerShell or Terminal inside the project root folder.
- 2.	Step 2: Create a Virtual Environment<br>
python -m venv .venv<br>
Activate it:<br>
 Windows:<br>
.venv\Scripts\activate<br>
Mac/Linux:
source .venv/bin/activate
- 3.	Step 3: Install Dependencies
pip install -r requirements.txt
- 4.	Step 4: Create the .env File
In the project root folder, create a file named .env<br>
Add the following:<br>
GROQ_API_KEY=your_groq_api_key_here<br>
GROQ_MODEL=llama-3.3-70b-versatile<br>
EMBEDDING_MODEL=all-MiniLM-L6-v2<br>
LOG_LEVEL=INFO<br>
API_PORT=8000<br>
Replace the API key with your actual Groq key.

## STEPS TO RUN THE PROJECT
-	Activate virtual environment
-	Ensure .env file exists
-	Start the server (Run: uvicorn api.app:app --port 8000)


