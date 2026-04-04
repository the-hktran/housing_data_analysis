# Take-Home Exercise for Research Scientist Position: Structure of the data

You have access to:

1. A synthetic **SQLite database** (`data/resident_database.db`) with tables that describe residents, their maintenance history, and renewal offers.
2. Two JSON files that contain **renewal-related chat conversations** of residents and an AI agent:
   * `data/conversations_train.json`
   * `data/conversations_test.json`

Your goal is to use the database tables and the chat transcripts to predict **`renewal_decision`** – whether a resident will accept or decline a renewal offer.


# Instructions

## 1. Data description
The data consists of tabular data and conversation data, split into training and test datasets.

### 1.1 SQLite database

```
resident_database.db
├── residents_train               # Resident information (train)
├── maintenance_history_train     # Maintenance work order submission by resident (train)
├── renewal_offers_train          # Renewal offers provided to each unit (train)
├── residents_test                # ... (test)
├── maintenance_history_test      # ... (test)
└── renewal_offers_test           # ... (test) 
```

### 1.2 Conversation data

Each element in the JSON files has the following shape:

```json
{
  "resident_id": "<int>",
  "conversation": [
    {"role": "user",      "message": "…", "time_sent": "YYYY-MM-DD HH:MM:SS"},
    {"role": "assistant", "message": "…", "time_sent": "YYYY-MM-DD HH:MM:SS"}
  ]
}
```

Use `resident_id` to join transcripts to the residents in the database.