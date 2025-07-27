Cost-Effective Question and Answer Generation with Context-Enhanced Prompting and Re-Ranking
============================================================================================

Project Description
-------------------

This project presents a **cost-effective pipeline for Question Generation (QG) and Answer Generation (AG)** leveraging **context-enhanced prompting** and **re-ranking** techniques. Instead of expensive model training or reinforcement learning (RLHF), I enrich the model inputs with additional context and apply lightweight post-processing to improve quality:

*   **Context-Enhanced Prompting:** Additional contextual information is retrieved and added to the model’s prompt (input) to make generated questions and answers more relevant and accurate. We use a T5 transformer model as the base (specifically a pre-trained T5-small fine-tuned for QG on SQuAD).
    
*   **Hybrid Retrieval & Re-Ranking:** For each query, we retrieve supporting passages using a hybrid approach combining **BM25** (sparse lexical retrieval) with a dense retriever (ColBERT for QG, DPR for AG). The retrieved contexts are then **re-ranked** – in QG, a ColBERT-based similarity score is combined with BM25 to select the most relevant context; in AG, BM25 is combined with DPR scores. This hybrid retrieval ensures the model considers both exact keyword overlap and semantic relevance. (In the QG pipeline, an additional BERT-based cross-encoder re-ranker was conceptually considered to further refine the top candidates.)
    

By using these strategies, the system aims to **improve the quality of generated questions and answers** without heavy fine-tuning or large model deployment. This makes advanced QG and AG more accessible and efficient for practical applications.

Dataset
-------

**SQuAD 2.0 (Stanford Question Answering Dataset)** is used for both QG and AG tasks. SQuAD 2.0 provides paragraphs with question-answer pairs, including unanswerable questions. Key characteristics:

*   **QG pipeline:** Uses passages from SQuAD as source contexts to generate new questions. Generated questions are compared to the original SQuAD questions for evaluation.
    
*   **AG pipeline:** Uses SQuAD questions (including unanswerable ones) and their contexts to generate answers. The real answers from SQuAD serve as references for evaluation.    

Usage
-----

The Colab notebook guides you through both pipelines step-by-step:

*   Question Generation (QG):
    
    *   Loads and samples from the SQuAD 2.0 dataset.
        
    *   Generates questions using a T5 model with context-enhanced prompts.
        
    *   Computes semantic similarity against ground truth questions.
        
*   Answer Generation (AG):
    
    *   Fine-tunes a T5 model on SQuAD QA pairs.
        
    *   Retrieves context using BM25 + DPR hybrid retrieval.
        
    *   Generates answers and evaluates them with metrics like EM, F1, BLEU, and ROUGE.
        
All results (tables, metrics, plots) are displayed directly within the notebook for easy inspection.

Evaluation
----------

**Question Generation Evaluation (Semantic Similarity):** Rather than exact string matching, we use a Sentence-BERT model to measure **semantic similarity** between each generated question and the reference question from SQuAD. This metric captures whether the model-generated question has the same meaning as the gold question, even if phrased differently.

*   The baseline T5 model (without retrieval augmentation) achieved a mean semantic similarity of approximately **0.560**.
    
*   With context-enhanced prompting and hybrid retrieval (BM25 + ColBERT) in the QG pipeline, the mean semantic similarity improved to about **0.625**, an **~11.6% increase**. This indicates the retrieved context helped generate questions that better align with the reference questions in meaning.
    

**Answer Generation Evaluation (F1, EM, BLEU, ROUGE):** We evaluate answers with traditional QA metrics:

*   **Exact Match (EM):** Percentage of predictions that exactly match the reference answer.
    
*   **F1 Score:** Harmonic mean of precision and recall, evaluating overlap in answer terms.
    
*   (We also report **BLEU** and **ROUGE** for informational purposes, which measure n-gram overlap and longer overlap respectively.)
    

The baseline fine-tuned T5 model (using gold context) achieved roughly **EM ≈ 44.7%** and **F1 ≈ 47.6%** on a subset of SQuAD, indicating it learned to produce many correct answers.

Using the **hybrid retrieval + re-ranking pipeline**:

*   The **EM** was about **43.0%**, essentially on par with the baseline on exact matches.
    
*   The **F1** dropped to about **23.6%**, suggesting that when the model missed the exact answer, its responses only partially overlapped with the reference.
    
*   The pipeline’s answers had high lexical overlap with references (e.g., BLEU ≈ 0.4286, ROUGE-1 ≈ 0.567), but slight phrasing differences or missing keywords led to a lower F1. This implies many generated answers were close to correct but not exact, which EM/F1 penalize heavily.
    

**Key Result:** The context retrieval + re-ranking method significantly **improved question generation quality**, but **did not boost answer generation performance** on SQuAD’s short answers. Because SQuAD answers are often very short (a word or two), retrieving additional context had limited benefit – the model already had the necessary context in the question prompt, and any re-ranking gains were minimal.

Future Work
-----------

Possible directions to extend and improve this work include:

*   **Evaluate on Different Datasets:** Apply the pipeline to datasets with _longer, descriptive answers_ or more open-ended questions. Retrieval augmentation may yield larger gains when answers require more context.
    
*   **Larger or More Advanced Models:** Leverage bigger transformer models (T5-3B, GPT-series, or FLAN-T5) to handle complex language and generate more diverse questions/answers. Larger models could especially improve the answer generation with richer context.
    
*   **Enhanced Dense Retrieval:** Use state-of-the-art retrievers like **ColBERTv2** or other dual encoders for better semantic search. Additionally, incorporate more powerful cross-encoders in the re-ranking step to better align questions with the most relevant context.
    
*   **Improved Evaluation Metrics:** Employ semantic-based evaluation metrics for answers (e.g., embedding similarity or answer containment) that reward correct information even if phrasing differs. This would better reflect the benefits of retrieval when exact wording is less important.
    
*   **Multi-turn QA Extension:** Extend the pipeline to _multi-turn question answering_ scenarios, where context from previous Q/A pairs must be retrieved and considered. Context-enhanced prompting could be useful for maintaining coherence across turns.

Acknowledgements
-----------
I, Khondaker Zahin Fuad, completed this project as part of an assignment for the Natural Language Processing (NLP) course in my Master’s program in Computer Science at American International University-Bangladesh (AIUB).
