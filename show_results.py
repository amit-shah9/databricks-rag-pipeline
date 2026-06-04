import mlflow
from mlflow import search_runs
mlflow.set_tracking_uri('databricks')
mlflow.set_experiment('/Users/amit.shah@quantum.de/rag_eval')
runs = search_runs(experiment_names=['/Users/amit.shah@quantum.de/rag_eval'],
                   max_results=5, order_by=['start_time DESC'])
for rid in runs['run_id']:
    traces = mlflow.search_traces(run_id=rid)
    has_corr = False
    for _, t in traces.iterrows():
        for a in t['assessments']:
            if a.get('assessment_name') == 'correctness':
                has_corr = True
    if not has_corr:
        continue
    passes = 0
    total = 0
    print('=== answerable run:', rid, '===')
    for _, t in traces.iterrows():
        req = str(t.get('request', ''))[:60]
        c = '?'
        r = '?'
        for a in t['assessments']:
            name = a.get('assessment_name')
            fb = a.get('feedback') or {}
            val = fb.get('value', '?')
            if name == 'correctness':
                c = val
            elif name == 'relevance_to_query':
                r = val
        if c == 'yes':
            passes += 1
        total += 1
        print(f"  {str(c):>4} corr | {str(r):>4} rel | {req}")
    print(f"  --> {passes}/{total} correct")
    break
