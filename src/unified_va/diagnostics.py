"""Original auxiliary-head condition decoding, evaluated after VA prediction."""
from contextlib import nullcontext
import numpy as np


def classification(true,predicted,supported):
    true=np.asarray(true,dtype=int);predicted=np.asarray(predicted,dtype=int)
    known=np.isin(true,[i for i in set(supported) if i>0]) & (true>0)
    y,p=true[known],predicted[known];labels=sorted(set(y.tolist()));scores=[];confusion=[]
    for label in labels:
        tp=int(np.sum((y==label)&(p==label)));fp=int(np.sum((y!=label)&(p==label)));fn=int(np.sum((y==label)&(p!=label)))
        scores.append(2*tp/max(2*tp+fp+fn,1))
        for guess in sorted(set(p[y==label].tolist())):
            confusion.append(dict(true_id=label,predicted_id=guess,count=int(np.sum((y==label)&(p==guess)))))
    return dict(total=len(true),known=int(known.sum()),unsupported=int((~known).sum()),
        coverage=float(known.mean()) if known.size else None,accuracy=float((y==p).mean()) if y.size else None,
        macro_F1=float(np.mean(scores)) if scores else None,macro_F1_class_ids=labels,confusion=confusion)


def run(predictor,rows):
    import torch
    if predictor.stage!='stage2' or predictor.auxiliary is None:raise ValueError('The original trained auxiliary condition head is required')
    captured=[]
    def after_forward(module,args,output):
        cache=predictor.engine._aggregate_backbone_conditional_forward_cache(module,detach=True)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if predictor.device.type=='cuda' else nullcontext()
        with torch.no_grad(),context:language,domain=predictor.auxiliary(cache['lang_emb'],cache['domain_emb'])
        captured.extend(zip(language.argmax(-1).cpu().tolist(),domain.argmax(-1).cpu().tolist()))
    handle=predictor.model.register_forward_hook(after_forward)
    try:predictions=predictor.predict(rows)
    finally:handle.remove()
    if len(captured)!=len(rows):raise RuntimeError('Auxiliary forwards do not align with input rows')
    # Only now encode the supplied evaluation labels. They never select VA conditions.
    true_lang,true_domain=predictor.auxiliary.encode_group_ids([r.get('lang') for r in rows],[r.get('domain') for r in rows],device=torch.device('cpu'))
    training=[name.split('_',1) for name in predictor.cfg['matched_protocol']['training_targets']]
    supported=predictor.auxiliary.encode_group_ids([r[0] for r in training],[r[1] for r in training],device=torch.device('cpu'))
    array=np.asarray(captured,dtype=int)
    return dict(language=classification(true_lang.tolist(),array[:,0],supported[0].tolist()),
        domain=classification(true_domain.tolist(),array[:,1],supported[1].tolist()),
        language_vocabulary=predictor.auxiliary.lang_to_idx,domain_vocabulary=predictor.auxiliary.domain_to_idx,
        predictions=predictions,scope='Original auxiliary-head decodability of continuous input conditions, not hard-routing accuracy.',
        labels_joined_after_VA_forward=True,unsupported_labels_excluded_from_classification_scores=True)
