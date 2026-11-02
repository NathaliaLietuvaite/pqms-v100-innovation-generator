# Project architecture

- Keep paper updates additive: preserve original reference documents in `public/`, index their summaries in `knowledge_base`, and share current directives through `PQMS_CORE_CONTEXT` for chatbot, paper, and code functions so all three use the same revision priorities.