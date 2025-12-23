import pickle

m = pickle.load(open('models/nba/moneyline_noleak_v3.pkl', 'rb'))
print('NBA No-Leak Model Results:')
print(f"Accuracy: {m['metrics']['accuracy']:.1%}")
print(f"CV: {m['metrics']['cv_accuracy']:.1%}")
print(f"AUC: {m['metrics']['auc']:.3f}")
print(f"Precision: {m['metrics']['precision']:.1%}")
print(f"Recall: {m['metrics']['recall']:.1%}")
print()
print('Top Features:')
for f, v in m['top_features'][:8]:
    print(f'  {f}: {v:.4f}')
