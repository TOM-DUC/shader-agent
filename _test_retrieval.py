"""Test hybrid retrieval."""
from shader_agent.corpus.vector_store import ShaderVectorStore
from shader_agent.corpus.keyword_store import KeywordStore
from shader_agent.corpus.parent_store import ParentDocumentStore
from shader_agent.corpus.retriever import HybridRetriever

vstore = ShaderVectorStore()
kstore = KeywordStore.load()
pstore = ParentDocumentStore()

print(f"vector chunks: {vstore.chunk_count()}")
print(f"vector shaders: {vstore.count()}")
print(f"keyword chunks: {kstore.count()}")
print(f"parent docs: {pstore.count()}")

retriever = HybridRetriever(vstore, kstore, pstore)
queries = ["raymarching sphere", "glitch effect", "bloom filter"]
for q in queries:
    results = retriever.retrieve(q, top_k=3)
    print(f"\nQuery: {q!r} -> {len(results)} results")
    for r in results:
        print(f"  [{r.fused_score:.3f}] {r.shader_id} | {r.name}")
