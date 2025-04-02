# 🍕 Pizza Sales Insight Bot

---

## Overview
The **Pizza Sales Insight Bot** is a powerful Streamlit-based chatbot application designed to provide actionable insights into pizza sales data. By integrating advanced tools like **LangChain**, **OpenAI**, and **SQL databases**, the bot can answer user queries, generate sales reports, recommend pizzas, and even send emails. Whether you're analyzing sales trends or exploring recommendations, this bot makes it easy and interactive.

---

## Key Features ✨

1. **Sales Insights**
   - Retrieve key metrics such as total revenue, profit, and order counts for any store.
   - Identify the top 3 most profitable pizzas and rank stores by profitability.

2. **Pizza Recommendations**
   - Get personalized pizza recommendations based on sales trends from similar stores in different regions.

3. **Email Functionality**
   - Send emails with custom subjects and bodies to specified recipients using Azure Communication Services.

4. **Interactive Chat**
   - Engage with the bot via text or voice input for a seamless experience.

5. **Document Querying**
   - Search and retrieve relevant information from preloaded documents using vector embeddings (FAISS).

6. **Python Code Execution**
   - Execute Python code snippets and visualize results directly within the app.

---

## Technologies Used 🛠️

| Category        | Tools/Libraries                                                                 |
|-----------------|---------------------------------------------------------------------------------|
| **Frontend**    | Streamlit                                                                      |
| **Backend**     | Python                                                                         |
| **Database**    | SQLite                                                                         |
| **APIs**        | LangChain, OpenAI, Azure Communication Services                                |
| **Libraries**   | `langchain`, `streamlit`, `sqlite3`, `FAISS`, `pyttsx3`, `speech_recognition`  |
|                 | `matplotlib`, `pandas`, `dotenv`                                               |

---

## Setup Instructions 🚀

### 1. Prerequisites
Before getting started, ensure you have the following:
- Python 3.8 or later installed.
- Required Python libraries (see `requirements.txt`).
- SQLite database file: `all_pizza_sales_insightsv6.2.db`.
- Preloaded FAISS vector database: `pizza_vector_database_v3`.
- `.env` file with the following keys:
  ```plaintext
  KEY=<Your OpenAI API Key>
  ENDPOINT=<Your Azure Endpoint>
  API_VERSION=<Your API Version>
  LLM_MODEL_NAME=<Your LLM Deployment Name>
  BOT_SECRET_TOKEN=<Your Bot Secret Token>
  ```

### 2. Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Files**:
   - Place the SQLite database (`all_pizza_sales_insightsv6.2.db`) and FAISS vector database (`pizza_vector_database_v3`) in their respective paths.
   - Ensure the `.env` file is located in the root directory.

### 3. Run the Application
Start the app using Streamlit:
```bash
streamlit run app_pizza_v2.py
```
Access the app in your browser at:  
👉 [http://localhost:8501](http://localhost:8501)

---

## Usage Guide 📋

### Interacting with the Bot
- Enter queries in the text input box or use the voice input feature.
- Example Queries:
  - "What is the total revenue of store SRT-1234 in the last quarter?"
  - "Recommend pizzas for store SRT-5678."
  - "Send an email with the subject 'Sales Report' to John Doe."

### Refreshing the App
Use the "Refresh" button to reset the app state.

---

## Key Components 🔧

1. **Chatbot Functionality**
   - **Agent Executor**: Uses LangChain tools to process queries and generate responses.
   - **Chat History**: Maintains session-based chat history for context-aware interactions.

2. **Tools**
   - **SQL Database Tool**: Queries the SQLite database for sales insights.
   - **Retriever Tool**: Searches documents using FAISS embeddings.
   - **Python REPL Tool**: Executes Python code snippets and visualizes results.
   - **Email Tool**: Sends emails using Azure Communication Services.

3. **Voice Input**
   - Transcribes user speech into text using `speech_recognition`.

4. **Visualization**
   - Displays charts and graphs using `matplotlib` and Streamlit.

---

## File Structure 📂

```
├── app_pizza_v2.py                # Main application file
├── all_pizza_sales_insightsv6.2.db # SQLite database file
├── pizza_vector_database_v3       # FAISS vector database
├── .env                           # Environment variables
├── requirements.txt               # Python dependencies
├── prompt_pizzza_v2.txt           # System prompt for the chatbot
├── sample_suggestion_qstns.csv    # Suggested questions for the chatbot
└── Nihilent_logo.gif              # Logo for the Streamlit app
```

---

## Environment Variables ⚙️

Ensure the `.env` file contains the following keys:
- `KEY`: Your OpenAI API key.
- `ENDPOINT`: Azure endpoint for OpenAI.
- `API_VERSION`: API version for Azure OpenAI.
- `LLM_MODEL_NAME`: Deployment name for the LLM.
- `BOT_SECRET_TOKEN`: Secret token for Azure Communication Services.

---

## Sample Queries 💡

- "What is the total revenue of store SRT-1234 in the last quarter?"
- "What are the top 3 most profitable pizzas for store SRT-5678?"
- "Recommend pizzas for store SRT-5678."
- "Send an email with the subject 'Sales Report' to John Doe."

---

## Troubleshooting 🛠️

- **Missing `.env` File**: Ensure the `.env` file exists and contains all required keys.
- **Database Connection Issues**: Verify the path to the SQLite database is correct.
- **FAISS Vector Database Issues**: Ensure the FAISS vector database is loaded correctly.
- **API Key Errors**: Double-check the OpenAI API key and Azure endpoint.

---

## License 📜

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments 🙌

- **LangChain**: For providing tools and frameworks for LLM-based applications.
- **Streamlit**: For enabling an interactive and user-friendly interface.
- **Azure Communication Services**: For seamless email functionality.
- **FAISS**: For efficient document retrieval.

---

Enjoy exploring pizza sales insights with the **Pizza Sales Insight Bot**! 🍕✨