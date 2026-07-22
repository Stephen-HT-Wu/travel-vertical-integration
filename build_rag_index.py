"""Pre-builds the local RAG title cache from local_settings.json's
rag_sitemap_url, so the first chat session doesn't pay the full sitemap-walk
+ per-article-title-fetch cost inline at webapp startup.

Usage: python build_rag_index.py
"""
from local_settings import load_local_settings
from site_index import RAG_CACHE_PATH, build_index, save_index


def main() -> None:
    settings = load_local_settings()
    if not settings.rag_sitemap_url:
        print("local_settings.json 沒有設定 rag_sitemap_url，略過（RAG 功能目前停用）")
        return
    print(f"開始從 {settings.rag_sitemap_url} 建立索引（最多 {settings.rag_max_articles} 篇）…")
    index = build_index(settings.rag_sitemap_url, max_articles=settings.rag_max_articles)
    save_index(index, RAG_CACHE_PATH)
    print(f"完成，寫入 {len(index.articles)} 篇文章標題到 {RAG_CACHE_PATH}")


if __name__ == "__main__":
    main()
