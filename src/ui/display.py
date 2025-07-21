def format_docs_output(docs: list[dict]) -> None:
    try:
        for i, doc in enumerate(docs, 1):
            score = doc.get('score')
            if score is not None:
                print(f"--- Document {i} (Score: {score:.4f}) ---")
            else:
                print(f"--- Document {i} ---")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Link: {doc.get('link', 'N/A')}")
            #print(f"Summary: {doc.get('summary', '')}")
            #print(f"Content Snippet: {doc.get('content', '')[:300]}...")
            print()
    except Exception as e:
        print("Error formatting output:", e)