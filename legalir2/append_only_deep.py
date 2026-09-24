"""Admit a scored newcomer without refitting priors or reranking existing answers."""
import numpy as np
from . import duo_agreement
from .h100 import digest

POLICY={'name':'append-only-deep-v1','minimum_pair_preference':.95,
        'protected_prefix':4,'maximum_replacements':1,
        'consistency':'positive mean comparison advantage against the protected first four',
        'pool':'only new documents in the saved 512 deep-pool comparisons',
        'old_prior':'never refitted; current agreement answers otherwise frozen'}


def predict(base,retrained,expanded):
    current,_=duo_agreement.predict(base,retrained)
    if (any(expanded[k]!=base[k] for k in ('id','question')) or expanded['base']!=digest(base)
            or expanded['docs'][:len(base['docs'])]!=base['docs']):
        raise ValueError('Expanded comparisons do not extend the original scored pool')
    docs=expanded['docs']; p=np.asarray(expanded['probabilities'],dtype=float); size=len(docs)
    if (len(set(docs))!=size or p.shape!=(size,size) or not np.isfinite(p).all()
            or (p<0).any() or (p>1).any() or not np.allclose(p+p.T,1.,atol=1e-6,rtol=0)
            or not np.array_equal(p[:len(base['docs']),:len(base['docs'])],base['probabilities'])):
        raise ValueError('Original comparisons changed or expanded matrix is invalid')
    protected=[docs.index(d) for d in current[:4]]
    victim=docs.index(current[4]);eligible=[]
    for i in range(len(base['docs']),size):
        confidence=float(p[i,victim])
        consistency=float(np.mean(p[i,protected]-p[victim,protected]))
        if confidence>=POLICY['minimum_pair_preference'] and consistency>0:
            eligible.append((confidence,consistency,docs[i]))
    answer=list(current)
    trace={'policy':POLICY['name'],'eligible':len(eligible),'replaced':False}
    if eligible:
        confidence,consistency,doc=min(eligible,key=lambda r:(-r[0],-r[1],r[2]))
        answer[4]=doc
        trace.update(replaced=True,removed=current[4],added=doc,
                     pair_preference=confidence,anchor_advantage=consistency)
    if len(set(answer))!=5 or answer[:4]!=current[:4]:
        raise ValueError('Protected answer changed')
    return answer,trace
