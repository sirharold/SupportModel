def calculate_rag_metrics_real(question: str, context_docs: list, generated_answer: str, ground_truth: str):
    """Calculate comprehensive RAG metrics using real OpenAI API and BERTScore (with caching)"""

    global openai_cache, semantic_similarity_model

    # Get context links for cache key
    context_links = [doc.get('link', '') for doc in context_docs[:3]]

    # Create unique cache key including generated_answer and ground_truth
    cache_input = f"{question}|{generated_answer}|{ground_truth}"

    # Try to get from cache first
    if openai_cache:
        cached = openai_cache.get(cache_input, context_links, prompt_type="rag_metrics")
        if cached:
            # Return cached metrics (excluding timestamp and context_links)
            return {k: v for k, v in cached.items() if k not in ['timestamp', 'context_links']}

    # Not in cache, calculate metrics
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Prepare context
        context_text = "\n".join([doc.get('content', '')[:3000] for doc in context_docs[:3]])

        # OPTIMIZED: Single API call for all RAGAS metrics (6 calls → 1 call, 83% cost reduction)
        combined_prompt = f"""Evaluate this RAG system output across 5 dimensions. Respond ONLY with a JSON object.

Question: {question}

Context: {context_text}

Generated Answer: {generated_answer}

Ground Truth Answer: {ground_truth if ground_truth else "Not provided"}

Rate each dimension on a 1-5 scale and respond in this EXACT JSON format:
{{
  "faithfulness": <1-5>,
  "answer_relevancy": <1-5>,
  "answer_correctness": <1-5>,
  "context_precision": <1-5>,
  "context_recall": <1-5>
}}

Dimension definitions:
- faithfulness: Does the answer contradict the context? (1=contradicts, 5=fully supported)
- answer_relevancy: Is the answer relevant to the question? (1=irrelevant, 5=perfect)
- answer_correctness: How correct is the answer vs ground truth? (1=wrong, 5=correct, 3=no ground truth)
- context_precision: How relevant is the context for answering? (1=irrelevant, 5=precise)
- context_recall: Does context have all info needed for ground truth? (1=missing most, 5=complete, 3=no ground truth)

Respond with ONLY the JSON object, no other text."""

        # Initialize scores
        faithfulness_score = None
        relevancy_score = None
        correctness_score = None
        context_precision_score = None
        context_recall_score = None

        try:
            ragas_response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": combined_prompt}],
                max_tokens=150,
                temperature=0,
                response_format={"type": "json_object"}  # Force JSON response
            )

            # Parse JSON response
            metrics_json = json.loads(ragas_response.choices[0].message.content)

            # Extract and normalize scores (1-5 scale to 0-1)
            faithfulness_score = float(metrics_json.get("faithfulness", 0)) / 5.0 if metrics_json.get("faithfulness") else None
            relevancy_score = float(metrics_json.get("answer_relevancy", 0)) / 5.0 if metrics_json.get("answer_relevancy") else None
            correctness_score = float(metrics_json.get("answer_correctness", 0)) / 5.0 if metrics_json.get("answer_correctness") else None
            context_precision_score = float(metrics_json.get("context_precision", 0)) / 5.0 if metrics_json.get("context_precision") else None
            context_recall_score = float(metrics_json.get("context_recall", 0)) / 5.0 if metrics_json.get("context_recall") else None

        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse RAGAS JSON response: {e}")
            # Scores remain None
        except Exception as e:
            print(f"⚠️ Failed to calculate RAGAS metrics: {e}")
            # Scores remain None

        # 6. BERTScore metrics (precision, recall, f1) and Semantic Similarity
        bert_precision = None
        bert_recall = None
        bert_f1 = None
        semantic_similarity = None

        try:
            if ground_truth and generated_answer:
                # Calculate real BERTScore using bert-score library
                from bert_score import score as bert_score_fn

                P, R, F1 = bert_score_fn(
                    [generated_answer],
                    [ground_truth],
                    lang='en',
                    model_type='microsoft/deberta-base-mnli',
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    batch_size=1,
                    verbose=False
                )

                # Free up GPU memory to prevent error CUDA out of memory
                torch.cuda.empty_cache()

                bert_precision = float(P[0])
                bert_recall = float(R[0])
                bert_f1s = F1.tolist()

                # Eliminar variables para liberar memoria
                del P, R, F1

                # Calculate semantic similarity separately using sentence transformers
                # Use global model to avoid loading it 2067 times (GPU memory leak fix)
                if semantic_similarity_model is not None:
                    gt_embedding = semantic_similarity_model.encode(ground_truth)
                    answer_embedding = semantic_similarity_model.encode(generated_answer)

                    similarity = cosine_similarity(
                        gt_embedding.reshape(1, -1),
                        answer_embedding.reshape(1, -1)
                    )[0][0]
                    semantic_similarity = float(similarity)
                else:
                    # Fallback if model not initialized (shouldn't happen)
                    print("⚠️ Warning: semantic_similarity_model not initialized")
                    semantic_similarity = None

        except Exception as e:
            # Fallback in case of errors
            print(f"⚠️ Failed to calculate BERTScore/semantic similarity: {e}")
            bert_precision = None
            bert_recall = None
            bert_f1 = None
            semantic_similarity = None

        # Prepare result dict
        result = {
            # RAGAS metrics
            'faithfulness': faithfulness_score,
            'answer_relevancy': relevancy_score,  # Note: using 'answer_relevancy' (with y) as expected by Streamlit
            'answer_correctness': correctness_score,
            'context_precision': context_precision_score,
            'context_recall': context_recall_score,
            'semantic_similarity': semantic_similarity,

            # BERTScore metrics
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'bert_f1': bert_f1,

            # Additional fields
            'evaluation_method': 'Complete_RAGAS_OpenAI_BERTScore'
        }

        # Save to cache
        if openai_cache:
            openai_cache.set(
                cache_input,
                {**result, "context_links": context_links},
                context_links,
                prompt_type="rag_metrics"
            )

        return result

    except Exception as e:
        print(f"⚠️ CRITICAL error in RAG metrics calculation: {e}")
        return {
            # RAGAS metrics - all None on critical error
            'faithfulness': None,
            'answer_relevancy': None,
            'answer_correctness': None,
            'context_precision': None,
            'context_recall': None,
            'semantic_similarity': None,

            # BERTScore metrics - all None on critical error
            'bert_precision': None,
            'bert_recall': None,
            'bert_f1': None,

            # Additional fields
            'evaluation_method': 'Critical_Error_Fallback'
        }
