import sqlite3
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import sys # Import sys for exiting
import json

# --- Database Loading ---
try:
    with sqlite3.connect("echoverse.db") as db:
        cursor = db.cursor()

        print("Loading verse embeddings (Star Map)...")
        cursor.execute("SELECT id, embedding FROM verses")
        rows = cursor.fetchall()

        ids = []
        embeddings = [] # Store embeddings to potentially use them
        id_to_index = {}
        for index, row in enumerate(rows):
            db_id = row[0]
            ids.append(db_id)
            id_to_index[db_id] = index # Map database ID to 0-based index
            emb = np.frombuffer(row[1], dtype=np.float32)
            embeddings.append(emb) # Store the embedding itself

        N = len(ids)
        if N == 0:
            print("No verses found. Exiting.")
            sys.exit() # Use sys.exit()

        embeddings = np.stack(embeddings) # Stack embeddings into a single numpy array

        # --- Added Check 1: Check embeddings for NaNs/Infs ---
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            print("Error: Embeddings contain NaN or Inf values. Exiting.")
            sys.exit()
        print("Embeddings check: OK")


        print("Loading archetype data (Charts & Compass)...")
        cursor.execute("SELECT verse_id, best_match_emotion_name, best_match_verse_archetype_text FROM archetypes")
        archetype_data = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        print("Loading whisper links (Oceans) and building graph structure...")
        cursor.execute("SELECT id1, id2, similarity FROM whispers")
        whispers_raw = cursor.fetchall() # Store raw whispers for easier neighbor lookup later

except sqlite3.Error as e:
    print(f"Database error: {e}")
    sys.exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    sys.exit()


# --- Reworking Host Graph Building (based on j -> i influences i) ---
# Re-read whispers to build the graph structure where influencer_indices[i] are verses that whisper TO i.
print("Rebuilding graph structure (j -> i influences i)...")
influencer_counts_dict = {} # How many verses whisper TO this verse?
# Filter whispers again, using original IDs
filtered_whispers_raw = [(id1, id2, sim)
                         for id1, id2, sim in whispers_raw
                         if id1 in id_to_index and id2 in id_to_index]

# Count how many whispers point TO each verse (id2 is the target)
for id1, id2, sim in filtered_whispers_raw:
    target_index = id_to_index[id2]
    influencer_counts_dict[target_index] = influencer_counts_dict.get(target_index, 0) + 1

influencer_counts_host = np.zeros(N, dtype=np.int32)
for index, count in influencer_counts_dict.items():
    influencer_counts_host[index] = count

total_influences = np.sum(influencer_counts_host)

# If total_influences is 0, exit as the graph is empty
if total_influences == 0:
    print("No valid whisper links found between loaded verses. Exiting.")
    sys.exit()


influencer_indices_host = np.zeros(total_influences, dtype=np.int32) # Indices of verses that whisper TO this verse
influencer_weights_host = np.zeros(total_influences, dtype=np.float64) # Weights of those whispers
influencer_list_starts_host = np.zeros(N, dtype=np.int32)

# Populate the influencer arrays based on the target index (id2_index)
# Sort filtered whispers by the target index (id2_index)
# Need to sort by the index of id2
filtered_whispers_raw.sort(key=lambda x: id_to_index[x[1]])

current_idx = 0
current_target_index = 0

for id1, id2, sim in filtered_whispers_raw:
    target_index = id_to_index[id2]
    # Ensure we process verses in increasing target_index order
    while target_index > current_target_index:
        current_target_index += 1
        # Fill in start indices for any verses with no incoming links
        # The start index for a vertex with no incoming links is the current global index
        if current_target_index < N: # Ensure index is within bounds
             influencer_list_starts_host[current_target_index] = current_idx


    if target_index == current_target_index:
        influencer_indices_host[current_idx] = id_to_index[id1] # The source vertex (id1) index influences the target (id2)
        influencer_weights_host[current_idx] = sim
        current_idx += 1

# Fill the remaining start indices for verses at the end with no incoming links
# These should all point to the end of the edge list (total_influences)
for i in range(current_target_index + 1, N):
     if i < N: # Ensure index is within bounds
         influencer_list_starts_host[i] = total_influences


# --- Added Check 2: Check weights for NaNs/Infs ---
if np.isnan(influencer_weights_host).any() or np.isinf(influencer_weights_host).any():
    print("Error: Influencer weights contain NaN or Inf values. Exiting.")
    sys.exit()
print("Influencer weights check: OK")

# --- Added Check 3: Check graph structure integrity (basic) ---
# Ensure start indices are non-decreasing and within bounds
if not np.all(np.diff(influencer_list_starts_host) >= 0):
     print("Error: Influencer list starts are not monotonically increasing. Graph structure error. Exiting.")
     sys.exit()
if influencer_list_starts_host[0] != 0:
    print("Warning: Influencer list starts[0] is not 0.")
if influencer_list_starts_host[-1] > total_influences:
     print("Error: Influencer list starts[-1] is out of bounds. Graph structure error. Exiting.")
     sys.exit()
# Ensure indices are within N
if np.min(influencer_indices_host) < 0 or np.max(influencer_indices_host) >= N:
    print("Error: Influencer indices are out of bounds [0, N-1]. Graph structure error. Exiting.")
    sys.exit()
print("Graph structure integrity check: OK")


print(f"Built graph structure for {N} vertices and {total_influences} influence links.")

# --- Optional: Print a summary of the graph structure ---
print("Graph structure summary:")
print(f"  Total verses (N): {N}")
print(f"  Total whispers (edges): {total_influences}")
print(f"  Max whispers per verse: {np.max(influencer_counts_host)}")
print(f"  Min whispers per verse: {np.min(influencer_counts_host)}")
# Optional: print the first few influencer indices and weights for verification
print("First few influencer indices and weights:")
for i in range(min(5, total_influences)):
    print(f"  Index {i}: Influencer {influencer_indices_host[i]}, Weight {influencer_weights_host[i]:.4f}")
# Optional: print the first few influencer list starts for verification
print("First few influencer list starts:")
for i in range(min(5, N)):
    print(f"  Verse {i}: Start index {influencer_list_starts_host[i]}, Count {influencer_counts_host[i]}")
# Optional: print the first few embeddings for verification
print("First few embeddings (L2 norms):")
for i in range(min(5, N)):
    print(f"  Verse {i}: Embedding norm {np.linalg.norm(embeddings[i]):.4f}")
