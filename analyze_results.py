import json
from collections import defaultdict

def analyze_results(results_file):
    """결과 파일 분석: recall, precision, F1, EM 등"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 기본 통계
    num_examples = len(results)
    recalls = []
    precisions = []
    f1_scores = []
    em_scores = []  # Exact Match
    num_retrieved = []
    
    for r in results:
        # final_doc_indices가 없으면 스킵 (손상된 결과)
        if 'final_doc_indices' not in r or 'supporting_doc_indices' not in r:
            continue
            
        retrieved_set = set(r['final_doc_indices'])
        supporting_set = set(r['supporting_doc_indices'])
        
        num_retrieved.append(len(retrieved_set))
        
        # Recall: supporting docs 중 몇 개를 찾았는가
        if len(supporting_set) > 0:
            recall = len(retrieved_set & supporting_set) / len(supporting_set)
        else:
            recall = 1.0 if len(retrieved_set) == 0 else 0.0
        recalls.append(recall)
        
        # Precision: 뽑은 docs 중 몇 개가 supporting인가
        if len(retrieved_set) > 0:
            precision = len(retrieved_set & supporting_set) / len(retrieved_set)
        else:
            precision = 1.0 if len(supporting_set) == 0 else 0.0
        precisions.append(precision)
        
        # F1 Score
        if recall + precision > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
        
        # EM (Exact Match): 뽑은 문서 집합이 supporting 문서 집합과 정확히 일치하는가
        em = 1.0 if retrieved_set == supporting_set else 0.0
        em_scores.append(em)
    
    # 계산
    num_valid_examples = len(recalls)  # 유효한 결과 개수
    avg_recall = sum(recalls) / num_valid_examples if num_valid_examples > 0 else 0.0
    perfect_recall_count = sum(1 for r in recalls if r == 1.0)
    perfect_recall_rate = perfect_recall_count / num_valid_examples if num_valid_examples > 0 else 0.0
    
    avg_precision = sum(precisions) / num_valid_examples if num_valid_examples > 0 else 0.0
    avg_f1 = sum(f1_scores) / num_valid_examples if num_valid_examples > 0 else 0.0
    em_rate = sum(em_scores) / num_valid_examples if num_valid_examples > 0 else 0.0
    em_count = int(sum(em_scores))
    
    avg_num_retrieved = sum(num_retrieved) / num_valid_examples if num_valid_examples > 0 else 0.0
    min_num_retrieved = min(num_retrieved) if num_retrieved else 0
    max_num_retrieved = max(num_retrieved) if num_retrieved else 0
    
    # recall@k 계산 (k = 뽑은 문서 수별)
    recall_at_k = defaultdict(list)
    f1_at_k = defaultdict(list)
    for i, r in enumerate(results):
        # 유효한 결과만 처리
        if 'final_doc_indices' not in r:
            continue
        if i >= len(recalls):  # recalls 리스트와 동기화
            break
        k = len(r['final_doc_indices'])
        recall_at_k[k].append(recalls[i])
        f1_at_k[k].append(f1_scores[i])
    
    # 출력
    print("=" * 60)
    print(f"Results Analysis: {results_file}")
    print("=" * 60)
    print(f"Total Examples: {num_examples} (Valid: {num_valid_examples})")
    print()
    
    print("[Retrieval Metrics]")
    print(f"Recall:     {avg_recall:.4f}")
    print(f"Precision:  {avg_precision:.4f}")
    print(f"F1 Score:   {avg_f1:.4f}")
    print()
    
    print("[Perfect Match Statistics]")
    print(f"Perfect Recall (recall=1.0):  {perfect_recall_count}/{num_examples} ({perfect_recall_rate:.2%})")
    print(f"Exact Match (EM):             {em_count}/{num_examples} ({em_rate:.2%})")
    print()
    
    print("[Document Retrieval Statistics]")
    print(f"Average # of Documents Retrieved: {avg_num_retrieved:.2f}")
    print(f"Min # of Documents: {min_num_retrieved}")
    print(f"Max # of Documents: {max_num_retrieved}")
    print()
    
    print("[Metrics by Number of Retrieved Documents]")
    print(f"{'K':>3} | {'Count':>5} | {'Recall':>7} | {'Precision':>9} | {'F1':>7}")
    print("-" * 50)
    for k in sorted(recall_at_k.keys()):
        count = len(recall_at_k[k])
        avg_recall_k = sum(recall_at_k[k]) / count
        avg_f1_k = sum(f1_at_k[k]) / count
        # precision은 정의상 k가 늘어나면 감소 경향이 있으므로 별도 계산
        precisions_k = [precisions[i] for i, r in enumerate(results) 
                       if 'final_doc_indices' in r and len(r['final_doc_indices']) == k and i < len(precisions)]
        avg_precision_k = sum(precisions_k) / len(precisions_k) if precisions_k else 0.0
        print(f"{k:3d} | {count:5d} | {avg_recall_k:7.4f} | {avg_precision_k:9.4f} | {avg_f1_k:7.4f}")
    print()
    
    # 분포 분석
    print("[Distribution of Retrieved Document Counts]")
    dist = defaultdict(int)
    for k in num_retrieved:
        dist[k] += 1
    for k in sorted(dist.keys()):
        print(f"  {k} documents: {dist[k]} examples ({dist[k]/num_examples:.1%})")
    print()
    
    return {
        'avg_recall': avg_recall,
        'avg_precision': avg_precision,
        'avg_f1': avg_f1,
        'em_rate': em_rate,
        'perfect_recall_rate': perfect_recall_rate,
        'avg_num_retrieved': avg_num_retrieved,
        'recall_at_k': dict(recall_at_k),
        'f1_at_k': dict(f1_at_k)
    }

if __name__ == '__main__':
    import sys
    
    # 두 결과 파일 비교
    file1 = 'results/multi_step_results.json'
    file2 = 'results/ircot_baseline_results.json'
    
    print(f"\n[1] Graph-based Multi-Step Retrieval")
    stats1 = analyze_results(file1)
    
    print(f"\n[2] IRCoT BM25 Baseline")
    try:
        stats2 = analyze_results(file2)
        
        print("\n" + "=" * 60)
        print("Comparison: Graph-based vs BM25 Baseline")
        print("=" * 60)
        print(f"Recall:            {stats1['avg_recall'] - stats2['avg_recall']:+.4f}")
        print(f"Precision:         {stats1['avg_precision'] - stats2['avg_precision']:+.4f}")
        print(f"F1 Score:          {stats1['avg_f1'] - stats2['avg_f1']:+.4f}")
        print(f"Exact Match (EM):  {stats1['em_rate'] - stats2['em_rate']:+.2%}")
        print(f"Perfect Recall:    {stats1['perfect_recall_rate'] - stats2['perfect_recall_rate']:+.2%}")
        print(f"Avg Docs Retrieved - Graph: {stats1['avg_num_retrieved']:.2f}, BM25: {stats2['avg_num_retrieved']:.2f}")
        print()
    except FileNotFoundError:
        print(f"\n⚠️  {file2} not found yet. Run baseline to compare.")
