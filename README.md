
#  Healthcare Provider Validation System 

##  Project Overview

The Healthcare Provider Validation System is a production-ready application designed to validate and enrich provider data using a multi-step, AI-powered workflow. It leverages real-world data from the NPI Registry, simulates data corruption for robust testing, and uses web scraping and an LLM-based agent graph to verify provider contact and professional information, ensuring high data quality and confidence scoring.

This version specifically addresses and fixes critical JSON serialization issues related to MongoDB's `ObjectId` and Python's `datetime` objects, making it stable for production use.

##  Key Features

* **Real Data Integration:** Collects and utilizes over 200 real healthcare provider records from the NPI Registry API.
* **Simulated Corruption:** Simulates real-world data issues by corrupting approximately 30% of the dataset for realistic validation testing.
* **AI-Powered Validation:** Utilizes a LangGraph agent workflow powered by `gpt-4o-mini` for web search, data extraction, validation, and scoring.
* **Web Enrichment:** Implements web scraping (`httpx`, `BeautifulSoup`, `lxml`) to find and extract up-to-date contact information from external web sources.
* **Configurable Thresholds:** Allows users to configure confidence thresholds for **Enrichment (X)** and **Update/Manual Review (Y)** in the Streamlit UI.
* **Robust Logging & Monitoring:** Features a comprehensive logging system with a live activity monitor, debug console, and a log cleanup utility.
* **MongoDB/In-Memory Backend:** Uses MongoDB for persistent storage of provider records and validation logs, with an automatic fallback to in-memory storage if the URI is not configured.
* **Real-time UI:** Provides an interactive Streamlit dashboard with a live workflow diagram, database browser, and statistics.

##  System Architecture

The core of the system is built around a **LangGraph** state machine that manages the validation process for each provider:

1.  **Extractor:** Initializes the workflow state with the provider's original data.
2.  **Input Guardrail:** Checks incoming data for PII/PHI, injection, and basic factual validity (e.g., plausible city/state).
3.  **Validator Agent:** Performs initial web enrichment (basic search and extraction).
4.  **Scorer:** Calculates an **Overall Confidence Score** based on web-extracted data.
5.  **Conditional Routing:**
    * If confidence is **< Enrichment Threshold (X)**, the process routes to the **Enricher**.
    * If confidence is **>= Enrichment Threshold (X)**, the process routes to the **QA Agent**.
6.  **Enricher:** Performs intensive web enrichment (more URLs checked) and re-scores.
7.  **QA Agent:** Generates a final validation report and determines if the record **Needs Manual Review** (if confidence < Update Threshold Y).
8.  **Output Guardrail:** Validates the structure and content of the final report before marking the provider as validated.

## Setup and Installation

### Prerequisites

* Python 3.9+
* OpenAI API Key

###  Install Dependencies

The following packages are required. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
# Required
OPENAI_API_KEY="your_openai_api_key_here"

# Optional, but highly recommended for persistent storage
MONGODB_URI="your_mongodb_connection_string_here" 

# Note: The system does not explicitly use SERPAPI in the provided code, 
# but the environment variable is loaded if present. Web enrichment uses direct 
# httpx calls for Google search scraping.
# SERPAPI_API_KEY="your_serpapi_key_here"
streamlit run app.py
## Usage
Access the Dashboard: The Streamlit app will open in your browser.

Generate Dataset: In the sidebar, click "🚀 Generate Real Dataset (200)". This will fetch data from the NPI Registry, simulate the 30% corruption, and insert the records into the database.

Configure Thresholds: Adjust the X - Enrichment Threshold and Y - Update Threshold sliders in the sidebar as needed.

Start Validation: Click "▶️ Start Validation Workflow" to initiate the LangGraph process for the pending providers.

Monitor: Use the "🔄 Live Monitor" tab to see the active workflow diagram and live activity log.

Review: Check the "📊 Dashboard" and "📋 Database View" tabs to analyze the confidence scores and conflict flags applied to the providers.
