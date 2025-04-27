import argparse
import json
import csv
import os
from echoverse import compute_similarity_cuda_filtered, compute_all_pairs_batched_gpu, normalize_embeddings
import numpy as np
from tqdm import tqdm
import pycuda.driver as cuda

def load_embeddings(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ids, texts, embeddings = [], [], []
    for entry in data:
        emb = np.array(entry['Embedding'], dtype=np.float32)
        if np.linalg.norm(emb) == 0:
            continue
        ids.append(entry['ID'])
        texts.append(entry['Text'])
        embeddings.append(emb)

    if not embeddings:
        print("No valid embeddings found.")
        return [], [], np.empty((0, 768), dtype=np.float32)

    embeddings = normalize_embeddings(np.vstack(embeddings))
    return ids, texts, embeddings

def format_whispers(ids, texts, filtered_pairs):
    whispers = []
    for pair in tqdm(filtered_pairs, desc="Formatting pairs"):
        i, j, sim = pair['id1_idx'], pair['id2_idx'], pair['similarity']
        if 0 <= i < len(ids) and 0 <= j < len(ids):
            whispers.append({
                'ID1': ids[i],
                'ID2': ids[j],
                'Similarity': round(float(sim), 4),
                'Text1': texts[i],
                'Text2': texts[j]
            })
    return whispers

def save_csv(whispers, output_path):
    if not whispers:
        print("No whisper pairs found. Writing empty CSV.")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID1', 'ID2', 'Similarity', 'Text1', 'Text2'])
        writer.writeheader()
        writer.writerows(whispers)

def main():
    parser = argparse.ArgumentParser(description="GPU-based Whisper Echo Pair Detection")
    parser.add_argument('--input', required=True, help='Input Verse_Embeddings.json')
    parser.add_argument('--output', default='whispers_gpu.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.53, help='Cosine similarity threshold')
    parser.add_argument('--debug', action='store_true', help='Enable GPU kernel debug mode')
    parser.add_argument('--mode', choices=['filtered', 'allpairs'], default='filtered', help='Similarity computation mode')
    parser.add_argument('--batch_size', type=int, default=256, help='(Ignored in optimized allpairs mode)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    print("Loading embeddings...")
    ids, texts, embeddings = load_embeddings(args.input)

    if embeddings.size == 0:
        print("No embeddings to process.")
        save_csv([], args.output)
        return

    try:
        if args.mode == 'allpairs':
            print("Computing all-pairs cosine similarities with batching...")
            filtered_pairs = compute_all_pairs_batched_gpu(
                embeddings,
                threshold=args.threshold
                # batch_size is ignored by new GPU version
            )
        else:
            print(f"Computing cosine similarities (threshold = {args.threshold})...")
            filtered_pairs = compute_similarity_cuda_filtered(
                embeddings,
                threshold=args.threshold,
                debug_mode=args.debug
            )
    except cuda.LogicError as e:
        print("\n🚨 CUDA Error: Could not launch kernel. Check that your GPU matches the specified compute architecture.")
        print(f"Details: {str(e)}")
        print("Try removing 'arch=\"compute_75\"' or adjusting to match your GPU (e.g., sm_86 for RTX 30xx).")
        save_csv([], args.output)
        return

    print(f"Formatting {len(filtered_pairs)} matched pairs...")
    whispers = format_whispers(ids, texts, filtered_pairs)

    print(f"Saving results to: {args.output}")
    save_csv(whispers, args.output)
    print("Done.")

if __name__ == '__main__':
    main()
