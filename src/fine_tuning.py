import json
import streamlit as st
import time
import random
from cohere.finetuning import FinetunedModel, Settings, BaseModel, Hyperparameters
from requests.exceptions import RequestException
from typing import List, Optional

def generate_fine_tune_dataset(chunks: List[str], cohere_client, fine_tune_dataset_path: str) -> List[dict]:
    """
    Generate fine-tuning dataset with diverse queries and UI editing.
    """
    dataset = []
    max_chunks = 20  # 40 API calls, ~2 min with delays
    max_retries = 3
    # Diverse prompt templates
    prompt_templates = [
        "Summarize this text in one sentence and generate a question it answers:\n\nText: {chunk}\n\nOutput format: {{ \"summary\": \"...\", \"query\": \"...\" }}",
        "Extract the main idea from this text and create a specific question about it:\n\nText: {chunk}\n\nOutput format: {{ \"summary\": \"...\", \"query\": \"...\" }}",
        "Identify a key detail in this text and generate a question targeting that detail:\n\nText: {chunk}\n\nOutput format: {{ \"summary\": \"...\", \"query\": \"...\" }}"
    ]
    
    st.subheader("Generated Fine-Tuning Dataset (Editable)")
    for i, chunk in enumerate(chunks[:max_chunks]):
        for attempt in range(max_retries):
            try:
                prompt = random.choice(prompt_templates).format(chunk=chunk)
                response = cohere_client.chat(model='command-r-08-2024', message=prompt)
                result = json.loads(response.text)
                query = result["query"]
                
                # Allow user to edit query
                edited_query = st.text_input(f"Query for Chunk {i+1}", value=query, key=f"ft_query_{i}")
                
                answer_prompt = f"Context: {chunk}\nQuestion: {edited_query}\nGenerate a concise answer:"
                answer_response = cohere_client.chat(model='command-r-08-2024', message=answer_prompt)
                
                # Allow user to edit answer
                edited_answer = st.text_area(f"Answer for Chunk {i+1}", value=answer_response.text, key=f"ft_answer_{i}")
                
                conversation = {
                    "messages": [
                        {"role": "System", "content": "You are a PDF expert assistant specializing in the uploaded document's content."},
                        {"role": "User", "content": edited_query},
                        {"role": "Chatbot", "content": edited_answer}
                    ]
                }
                dataset.append(conversation)
                time.sleep(6)  # 6-second delay for Trial key
                break
            except RequestException as e:
                if attempt < max_retries - 1:
                    st.warning(f"Error for chunk {i+1} on attempt {attempt + 1}: {str(e)}. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                st.warning(f"Failed to generate fine-tuning data for chunk {i+1}: {str(e)}")
    
    if len(dataset) < 2:
        st.error("Not enough data for fine-tuning (minimum 2 conversations).")
        return dataset
    
    # Save button for dataset
    if st.button("Save Fine-Tuning Dataset"):
        with open(fine_tune_dataset_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        st.success(f"Fine-Tuning Dataset Saved to {fine_tune_dataset_path} with {len(dataset)} conversations")
    
    return dataset

def start_fine_tuning(cohere_client, dataset_path: str, eval_path: Optional[str] = None):
    """
    Start fine-tuning using Cohere API.
    """
    try:
        dataset = cohere_client.datasets.create(
            name="pdf-rag-dataset",
            data=open(dataset_path, "rb"),
            eval_data=open(eval_path, "rb") if eval_path else None,
            type="chat-finetune-input",
        )
        cohere_client.wait(dataset)
        hp = Hyperparameters(
            train_epochs=1,
            learning_rate=0.01,
            train_batch_size=16
        )
        create_response = cohere_client.finetuning.create_finetuned_model(
            request=FinetunedModel(
                name="pdf-rag-finetuned",
                settings=Settings(
                    base_model=BaseModel(base_type="BASE_TYPE_CHAT"),
                    dataset_id=dataset.id,
                    hyperparameters=hp
                )
            )
        )
        return create_response
    except Exception as e:
        st.error(f"Error starting fine-tuning: {str(e)}")
        return None