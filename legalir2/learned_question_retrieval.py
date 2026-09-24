"""Train-only multi-label retrieval, correcting correlations between query features.

The existing question memory transfers labels from lexical nearest neighbours.
This learner fits a regularized question-to-document map on all training labels.
No evaluation labels, document priors or private predictions enter fitting.
"""
from collections import Counter
import math

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import csr_matrix

from .cache import normalise_question
from .question_memory import QuestionMemory,terms

POLICY={'name':'learned-question-retrieval-v2','ridge':1.,'maximum_additions':4,
        'features':'existing question-memory word/bigram/character TF-IDF',
        'targets':'binary training document labels','label_frequency_scaling':False,
        'duplicates':'keep one of identical labels; exclude conflicting-label text groups'}


class LearnedQuestions:
    def __init__(self,records,known,excluded_ids=(),excluded_texts=()):
        if not records:
            raise ValueError('Training questions required')
        self.known=set(known)
        self.training_ids=set(records)
        self.training_texts={normalise_question(row['question']) for row in records.values()}
        excluded={normalise_question(text) for text in excluded_texts}
        if self.training_ids&set(excluded_ids) or self.training_texts&excluded:
            raise ValueError('Training overlaps excluded evaluation questions')
        for row in records.values():
            labels=row.get('answer')
            if not labels or len(labels)!=len(set(labels)) or not set(labels)<=self.known:
                raise ValueError('Invalid training document labels')
        groups={}
        for key,row in sorted(records.items()):
            groups.setdefault(normalise_question(row['question']),[]).append(key)
        clean={}; conflicts=0
        for keys in groups.values():
            if len({tuple(sorted(records[key]['answer'])) for key in keys})!=1:
                conflicts+=len(keys)
                continue
            clean[keys[0]]=records[keys[0]]
        self.cleaning={'input_rows':len(records),'fit_rows':len(clean),'conflicting_rows_excluded':conflicts,
                       'identical_duplicates_removed':len(records)-len(clean)-conflicts}
        if not clean:
            raise ValueError('No unambiguous training rows')
        records=clean
        self.memory=QuestionMemory(records)
        self.documents=sorted({d for row in records.values() for d in row['answer']})
        positions={d:i for i,d in enumerate(self.documents)}
        target=np.zeros((len(records),len(self.documents)),dtype=np.float64)
        for i,key in enumerate(self.memory.ids):
            for doc in records[key]['answer']:
                target[i,positions[doc]]=1.
        matrix=self.memory.matrix.astype(np.float64)
        kernel=(matrix@matrix.T).toarray()
        kernel.flat[::len(records)+1]+=POLICY['ridge']
        factor=cho_factor(kernel,lower=True,check_finite=True,overwrite_a=True)
        self.coefficients=cho_solve(factor,target,check_finite=True)
        if not np.isfinite(self.coefficients).all():
            raise ValueError('Non-finite retrieval fit')
        self.label_counts=Counter(d for row in records.values() for d in row['answer'])

    def scores(self,query,qid=None):
        if qid in self.training_ids or normalise_question(query) in self.training_texts:
            raise ValueError('Training-query evaluation requires a separately fitted excluded fold')
        pairs=[(self.memory.vocabulary[t],self.memory.weight(t,n)) for t,n in terms(query).items()
               if t in self.memory.vocabulary]
        if not pairs:
            return {}
        indices,values=zip(*pairs)
        vector=csr_matrix((values,([0]*len(indices),indices)),shape=(1,len(self.memory.vocabulary)),
                          dtype=np.float64).multiply(self.memory.idf).tocsr()
        norm=math.sqrt(float(vector.multiply(vector).sum()))
        if not norm:
            return {}
        vector/=norm
        similarity=(self.memory.matrix@vector.T).toarray().ravel()
        values=similarity@self.coefficients
        if not np.isfinite(values).all():
            raise ValueError('Non-finite learned document score')
        return {doc:float(score) for doc,score in zip(self.documents,values)}

    def propose(self,query,existing,qid=None):
        if not set(existing)<=self.known:
            raise ValueError('Unknown candidate identity')
        scores=self.scores(query,qid)
        order=sorted((d for d in scores if d not in existing and scores[d]>0),key=lambda d:(-scores[d],d))
        return [{'doc':d,'score':scores[d],'training_support':self.label_counts[d]}
                for d in order[:POLICY['maximum_additions']]]
