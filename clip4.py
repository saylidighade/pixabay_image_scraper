'''
Use a hybrid score at first stage: score = α*s_img + β*s_meta already at retrieval.
Or run parallel retrievals (one by image, one by meta) and merge top-K results before re-ranking.

✅ Recommended practical approach
Stage 1: combined score retrieval (α≈0.5 for general tasks).
Stage 2: re-rank smaller candidate set using cross-modal model (e.g., CLIP cross-encoder or reranker).
Log which modality dominates per query — you’ll see patterns (visual vs textual dominance).

A **cross-modal reranker** takes both the query text *and* each candidate image (or image + metadata) together as input and scores their matching jointly, not via precomputed embeddings.
It’s slower but more accurate — used only on the top-N results from the first retrieval.
Examples: CLIP cross-encoder, BLIP, or SigLIP fine-tuned for text-image matching.

Not strictly compulsory — but highly recommended.
L2 normalization makes cosine similarity equivalent to a simple dot product, ensuring all embeddings lie on the same unit sphere.
Without it, vectors with larger magnitudes can dominate scores and distort similarity rankings, especially when mixing image and text embeddings.

Encode query once (text_emb), then compute separately: s_img = cos(text_emb, image_emb) and s_meta = cos(text_emb, meta_emb)
 and combine scores with weights: score = α*s_img + β*s_meta (tune α,β on a validation set).

retreive top k from query_to_img match and top k from query_to_meta match, merge and rerank
'''


