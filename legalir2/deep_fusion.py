"""Depth-100 fusion fitted only on frozen training questions.

Store raw scores and channel ranks so feature normalization has exactly the
same meaning in training, development and inference. Old V9 scores remain an
explicit signal, including an indicator for candidates it never scored.
"""
import numpy as np

from .stack import build_features, feature_names, fit

FEATURES = list(feature_names(2))+['v9_score', 'v9_rr', 'v9_present']
DEPTH, EXTRA = 100, 4


def candidate_pool(fused, retained, known):
    ranked = [str(d) for d, _ in fused[:DEPTH]]
    if len(ranked) < 5 or len(set(ranked)) != len(ranked):
        raise ValueError('Invalid fused candidate ranking')
    docs = list(dict.fromkeys(ranked+list(retained)))
    if len(set(retained)) != len(retained) or not set(docs) <= known:
        raise ValueError('Invalid retained candidates')
    return docs


def features(docs, tables, channels, v9):
    if len(tables) != 2 or any(set(t) != set(docs) for t in tables):
        raise ValueError('Both rerankers must score every deep candidate')
    if any(not values or not np.isfinite(values).all() for t in tables for values in t.values()):
        raise ValueError('Invalid raw reranker scores')
    base = build_features(docs, tables, channels, version=2)
    values = np.asarray(list(v9.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing frozen V9 scores')
    low, span = values.min(), max(float(np.ptp(values)), 1e-12)
    ranks = {d: i+1 for i, d in enumerate(sorted(v9, key=lambda d: (-v9[d], d)))}
    extra = [[(v9[d]-low)/span if d in v9 else 0.,
              1/(1+ranks[d]) if d in ranks else 0., float(d in v9)] for d in docs]
    return np.column_stack((base, extra))


def train(rows):
    if not rows or any(not r['gold'] for r in rows):
        raise ValueError('Fusion training requires labeled training rows')
    samples = [(np.asarray(r['features']), np.asarray([float(d in r['gold']) for d in r['docs']])) for r in rows]
    # Fixed before development: no validation-selected epochs or weight grid.
    weights, history = fit(samples, epochs=300, l2=.001)
    if weights.shape != (len(FEATURES),) or not np.isfinite(weights).all():
        raise ValueError('Invalid learned fusion weights')
    return {'weights': weights.tolist(), 'features': FEATURES, 'epochs': 300,
            'l2': .001, 'history': history, 'questions': len(rows)}


def scores(row, model):
    if model['features'] != FEATURES:
        raise ValueError('Fusion feature version differs')
    result = np.asarray(row['features']) @ np.asarray(model['weights'])
    if result.shape != (len(row['docs']),) or not np.isfinite(result).all():
        raise ValueError('Invalid fusion predictions')
    return result


def shortlist(row, original, model):
    if not set(original['docs']) <= set(row['docs']):
        raise ValueError('Deep collection dropped original duo candidates')
    values = scores(row, model)
    order = sorted(range(len(values)), key=lambda i: (-values[i], row['docs'][i]))
    extra = [row['docs'][i] for i in order if row['docs'][i] not in original['docs']][:EXTRA]
    docs = list(original['docs'])+extra
    raw = dict(zip(row['docs'], values))
    selected = np.asarray([raw[d] for d in docs])
    prior = (selected-selected.mean())/max(float(selected.std()), .5)
    return docs, prior


def measured_edges(old_size, new_size):
    if old_size < 5 or new_size < old_size or new_size-old_size > EXTRA:
        raise ValueError('Invalid expanded tournament size')
    return [(i, j) for i in range(old_size, new_size) for j in range(i)]


def predict(row, policy):
    if policy == 'baseline':
        return list(row['baseline'])
    if policy != 'deep':
        raise ValueError('Unknown deep-pool policy')
    prior, ability = np.asarray(row['prior']), np.asarray(row['ability'])
    score = prior+.25*(ability-prior)
    return [d for _, d in sorted(zip(-score, row['docs']))][:5]
