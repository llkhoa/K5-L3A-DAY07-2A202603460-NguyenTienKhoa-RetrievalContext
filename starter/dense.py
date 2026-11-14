
import argparse
import json
import time
import statistics
from pathlib import Path
import os
import numpy as np

# Google Gemini embedding SDK (google-genai). Assumes GEMINI_API_KEY env var is set.
import os, json, urllib.request, urllib.parse, numpy as np


def tokenize(text: str):
    # Not used for dense retrieval but kept for validation compatibility.
    return []


def validate(chunks, questions):
    # Re‑use validation from baseline (same schema)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
            if key not in c or c[key] is None:
                raise ValueError(f'chunk thiếu {key}')
        if not all(isinstance(c[k], str) and c[k].strip() for k in ('id', 'text', 'source_id', 'source')):
            raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
        if not isinstance(c['metadata'], dict) or c['id'] in ids:
            raise ValueError('metadata phải là object và ID chunk phải duy nhất')
        ids.add(c['id'])
    ids.clear()
    for q in questions:
        if not isinstance(q.get('id'), str) or not q['id'].strip() or q['id'] in ids:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        ids.add(q['id'])
        if q.get('split') not in ('dev', 'test') or not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError(f"question {q['id']}: split hoặc question không hợp lệ")
        if not isinstance(q.get('expected_answer'), str) or not q['expected_answer'].strip():
            raise ValueError(f"question {q['id']}: expected_answer phải là chuỗi không rỗng")
        if not isinstance(q.get('gold'), list) or not q['gold']:
            raise ValueError(f"question {q['id']}: gold phải là danh sách không rỗng")
        for g in q['gold']:
            if not all(key in g and g[key] is not None for key in ('source_id', 'page', 'evidence')) or not isinstance(g['evidence'], str) or not g['evidence']:
                raise ValueError(f"question {q['id']}: gold không hợp lệ")
            # gold existence check will be done later after embeddings are available


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text'] for g in gold)


class DenseIndex:
    def __init__(self, chunks, embed_model='gemini-embedding-002'):
        self.chunks = chunks
        self.embed_model = embed_model
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise EnvironmentError('GEMINI_API_KEY environment variable not set')
        # Prepare request for batch embedding of chunk texts
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.embed_model}:batchEmbedContents?key={api_key}'
        payload = {
            'requests': [{'content': {'role': 'user', 'parts': [{'text': c['text']}]}} for c in chunks]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode('utf-8'))
        # Extract embeddings
        self.chunk_vectors = [np.array(e['values'], dtype=np.float32) for e in response.get('embeddings', [])]
        # Normalise
        self.chunk_vectors = [v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v for v in self.chunk_vectors]

    def embed_question(self, question: str):
        api_key = os.getenv('GEMINI_API_KEY')
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.embed_model}:embedContent?key={api_key}'
        payload = {'content': {'role': 'user', 'parts': [{'text': question}]}}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode('utf-8'))
        vec = np.array(response['embedding']['value'], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def search(self, question, top_k=3):
        q_vec = self.embed_question(question)
        sims = np.dot(self.chunk_vectors, q_vec)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            chunk = self.chunks[idx]
            score = float(sims[idx])
            results.append({**chunk, 'score': score, 'strategy': 'dense'})
        return results


    def embed_question(self, question: str):
        resp = self.client.models.embed_content(
            model=self.embed_model,
            contents=question
        )
        vec = np.array(resp.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def search(self, question, top_k=3):
        q_vec = self.embed_question(question)
        # Compute cosine similarity with all chunk vectors
        sims = np.dot(self.chunk_vectors, q_vec)
        # Get top_k indices sorted by similarity descending
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            chunk = self.chunks[idx]
            score = float(sims[idx])
            results.append({**chunk, 'score': score, 'strategy': 'dense'})
        return results


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate(index, questions, budget=800):
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = index.search(q['question'], top_k=3)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'strategy': 'dense',
            'top_k': 3,
            'budget_chars': budget,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked)
        })
    return traces


def main():
    parser = argparse.ArgumentParser(description='Dense retrieval baseline for Day 7')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path, help='Output directory')
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)
    start = time.perf_counter()
    index = DenseIndex(chunks)
    index_seconds = time.perf_counter() - start
    traces = evaluate(index, selected)
    summary = {
        'strategy': 'dense',
        'split': args.split,
        'n': len(traces),
        'hit1': sum(t['hit1'] for t in traces),
        'hit3': sum(t['hit3'] for t in traces),
        'context_hit': sum(t['context_hit'] for t in traces),
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
        'index_seconds': index_seconds,
        'service_cost_usd': 0
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
