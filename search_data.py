import argparse
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query(query_text)


def query(query_text: str):
    # 准备 Chroma DB
    query_text = f"query: {query_text.strip()}"
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 语义检索 top 5
    results = db.similarity_search_with_score(query_text, k=5)

    # 结构化输出结果
    print(f"\n🔍 Frage: {query_text}")
    print(f"📄 Ergebnisse (Top {len(results)}):\n")

    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        source = meta.get("source", "unbekannt")
        typ = meta.get("type", "unbekannt")
        snippet = doc.page_content.strip()

        print(f"🔸 Ergebnis {i}")
        print(f"📌 Typ: {typ} | Kapitel: {source} | Score: {score:.4f}")
        print(f"{snippet}")
        print("—" * 40)

    return results



if __name__ == "__main__":
    main()
