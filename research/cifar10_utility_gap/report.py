"""Prespecified exploratory metrics/flags, never a formal noninferiority test."""
import math
from numbers import Integral


def relative_reduction(before, after):
    if not (math.isfinite(before) and math.isfinite(after)) or before < 0 or after < 0:
        raise ValueError('Finite nonnegative metrics required')
    return (before-after)/before if before > 0 else None


def drops_ten_percent(before, after):
    relative_reduction(before, after)
    return before > 0 and after <= .9*before


def paired_counts(teacher, student, labels):
    if not (len(teacher)==len(student)==len(labels)==1000): raise ValueError('Full 1000 diagnostic predictions required')
    if any(isinstance(value,bool) or not isinstance(value,Integral) or not 0 <= value < 10 for values in (teacher,student,labels) for value in values):
        raise ValueError('Class predictions/labels outside fixed ten classes')
    tc=[int(t)==int(y) for t,y in zip(teacher,labels)]
    sc=[int(s)==int(y) for s,y in zip(student,labels)]
    teacher_correct=sum(tc); student_correct=sum(sc)
    return dict(n=1000, teacher_correct=teacher_correct, student_correct=student_correct,
        teacher_accuracy=teacher_correct/1000, accuracy=student_correct/1000,
        teacher_drop=(teacher_correct-student_correct)/1000,
        both_correct=sum(t and s for t,s in zip(tc,sc)),
        teacher_correct_student_wrong=sum(t and not s for t,s in zip(tc,sc)),
        teacher_wrong_student_correct=sum(not t and s for t,s in zip(tc,sc)),
        both_wrong=sum(not t and not s for t,s in zip(tc,sc)),
        prediction_disagreements=sum(int(t)!=int(s) for t,s in zip(teacher,student)))


def contrasts(cells):
    """All eight prespecified checkpoint results; missing means inconclusive.

    Each available item has fit_nmse, diagnostic_nmse, diagnostic_kl, correct.
    Count differences (>=10/1000) avoid rounded percentages at thresholds.
    """
    names=[f'w{width}_{loss}@{step}' for width in (8,16) for loss in ('mse','kl') for step in (1024,2048)]
    if set(cells) != set(names): raise ValueError('Entire checkpoint matrix, including None, required')
    for value in cells.values():
        if value is None: continue
        if type(value.get('correct')) is not int or not 0 <= value['correct'] <= 1000:
            raise ValueError('Unrounded correct count required')
        for key in ('fit_nmse','diagnostic_nmse','diagnostic_kl'):
            relative_reduction(value[key],value[key])
    result=dict(exploratory=True, formal_noninferiority=False,
        primary_contrast='w8_mse:1024->2048', complete_matrix=all(v is not None for v in cells.values()),
        budget={}, width={}, alignment={})
    for width in (8,16):
        for loss in ('mse','kl'):
            name=f'w{width}_{loss}'; before=cells[name+'@1024']; after=cells[name+'@2048']
            if before is None or after is None:
                result['budget'][name]=dict(status='missing/inconclusive'); continue
            sensitive=drops_ten_percent(before['fit_nmse'],after['fit_nmse']) and drops_ten_percent(before['diagnostic_nmse'],after['diagnostic_nmse'])
            result['budget'][name]=dict(status='observed', primary=(name=='w8_mse'),
                budget_sensitivity=sensitive, task_relevant=sensitive and after['correct']-before['correct']>=10,
                fit_relative_reduction=relative_reduction(before['fit_nmse'],after['fit_nmse']),
                diagnostic_relative_reduction=relative_reduction(before['diagnostic_nmse'],after['diagnostic_nmse']),
                accuracy_gain_count=after['correct']-before['correct'])
    for loss in ('mse','kl'):
        narrow=cells[f'w8_{loss}@2048']; wide=cells[f'w16_{loss}@2048']; early=cells[f'w8_{loss}@1024']
        if narrow is None or wide is None:
            result['width'][loss]=dict(status='missing/inconclusive'); continue
        sensitive=drops_ten_percent(narrow['fit_nmse'],wide['fit_nmse']) and drops_ten_percent(narrow['diagnostic_nmse'],wide['diagnostic_nmse'])
        plateau=None if early is None else relative_reduction(early['fit_nmse'],narrow['fit_nmse'])
        result['width'][loss]=dict(status='observed',width_sensitivity=sensitive,
            task_relevant=sensitive and wide['correct']-narrow['correct']>=10,
            width_benefit_after_apparent_plateau=sensitive and wide['correct']-narrow['correct']>=10 and plateau is not None and abs(plateau)<.05,
            proven_insufficient_expressivity=False,accuracy_gain_count=wide['correct']-narrow['correct'])
    for width in (8,16):
        mse=cells[f'w{width}_mse@2048']; kl=cells[f'w{width}_kl@2048']
        if mse is None or kl is None:
            result['alignment'][str(width)]=dict(status='missing/inconclusive'); continue
        helps=kl['correct']-mse['correct']>=10 and drops_ten_percent(mse['diagnostic_kl'],kl['diagnostic_kl'])
        result['alignment'][str(width)]=dict(status='observed',
            alignment_evidence=helps and kl['diagnostic_nmse']>=mse['diagnostic_nmse'],
            alternative_objective_helps_without_isolated_alignment=helps and kl['diagnostic_nmse']<mse['diagnostic_nmse'],
            uniquely_causal=False,accuracy_gain_count=kl['correct']-mse['correct'])
    return result
