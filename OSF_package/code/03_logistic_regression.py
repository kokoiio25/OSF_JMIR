#!/usr/bin/env python3
""" Binary logistic regression predicting NEGATIVE sentiment (negative vs neutral/positive),
with Topic x Time-of-day and Topic x Region interactions. Primary model for RQ3.
Reproducible: reads table4_logistic_dataset.csv (analysis variables only; no raw text or user IDs).

Outcome:    negative (1 = negative, 0 = neutral or positive), from a validated LLM sentiment classifier.
Predictors: topic (paper Topic 1-8, numbered and labelled as in Table 2),
            tod (time of day, 4 levels), region (IP-based province/location, English name).
Reference:  topic = 2 (Social Anxiety and Interpersonal Struggles, the most neutral topic);
            tod = Morning; region = Guangdong (largest region).
            Note: likelihood-ratio tests are invariant to the choice of reference category.

Tested with: Python 3.11, pandas 2.x, numpy 1.26, statsmodels 0.14, scipy 1.11.
"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats

d = pd.read_csv('table4_logistic_dataset.csv')

# Topic numbering and labels follow Table 2 of the manuscript.
NAME = {1: 'Family Dynamics and Parental Relationships',
        2: 'Social Anxiety and Interpersonal Struggles',
        3: 'Existential Distress and Life Perspective',
        4: 'Medical Diagnosis and Healthcare Navigation',
        5: 'Workplace Stress & Daily Life Pressures',
        6: 'Somatic Anxiety Symptoms',
        7: 'Emotional Dysregulation & Physical Distress',
        8: 'Clinical Depression and Treatment'}


TOPIC = 'C(topic, Treatment(reference=2))'
TOD   = "C(tod, Treatment(reference='Morning'))"
REG   = "C(region, Treatment(reference='Guangdong'))"

def fit(formula, data):
   
    try:
        return smf.logit(formula, data).fit(disp=0)
    except np.linalg.LinAlgError:
        return smf.logit(formula, data).fit(disp=0, method='bfgs', maxiter=1000)

def lrtest(full, red):
    s = 2 * (full.llf - red.llf); df = int(full.df_model - red.df_model)
    return s, df, stats.chi2.sf(s, df)

def ortab(mod, key):
    ci = mod.conf_int(); out = []
    for n in mod.params.index:
        if n == 'Intercept' or key not in n: continue
        out.append(dict(term=n, B=mod.params[n], SE=mod.bse[n], OR=np.exp(mod.params[n]),
                        lo=np.exp(ci.loc[n, 0]), hi=np.exp(ci.loc[n, 1]), p=mod.pvalues[n]))
    return pd.DataFrame(out)

print(' MODEL 1: negative ~ Topic * Time of day')
dT = d.dropna(subset=['tod']).copy()
null  = fit('negative ~ 1', dT)
main  = fit(f'negative ~ {TOPIC}+{TOD}', dT)
full  = fit(f'negative ~ {TOPIC}*{TOD}', dT)
noTop = smf.logit(f'negative ~ {TOD}', dT).fit(disp=0)
noTod = smf.logit(f'negative ~ {TOPIC}', dT).fit(disp=0)
print(f'N={len(dT)}, negative={int(dT.negative.sum())} ({dT.negative.mean()*100:.1f}%)')
print('LR Topic:        chi2=%.2f df=%d p=%.3g' % lrtest(main, noTop))
print('LR Time of day:  chi2=%.2f df=%d p=%.3g' % lrtest(main, noTod))
print('LR TopicxTime:   chi2=%.2f df=%d p=%.3g' % lrtest(full, main))
print('McFadden R2 (main-effects model)=%.3f' % (1 - main.llf / null.llf))
ot = ortab(main, 'tod'); ot['level'] = ot.term.str.extract(r'\[T\.([^\]]+)\]')
print('\nTime-of-day odds ratios (ref = Morning):')
for _, r in ot.iterrows():
    print('  %-10s B=%.3f SE=%.3f OR=%.2f [%.2f, %.2f] p=%.4f' % (r.level, r.B, r.SE, r.OR, r.lo, r.hi, r.p))
otop = ortab(main, 'topic'); otop['tn'] = otop.term.str.extract(r'\[T\.(\d+)\]').astype(int)
print('Topic odds ratios (ref = Topic 2, Social Anxiety and Interpersonal Struggles):')
for _, r in otop.iterrows():
    print('  T%d %-42s OR=%.2f [%.2f, %.2f] p=%.3g' % (r.tn, NAME[r.tn], r.OR, r.lo, r.hi, r.p))

print(' MODEL 2: negative ~ Topic + Region (main effects)')
dR = d.dropna(subset=['region']).copy(); rc = dR.region.value_counts(); keep = list(rc[rc >= 50].index)
dR = dR[dR.region.isin(keep)].copy()
null2  = fit('negative ~ 1', dR)
main2  = fit(f'negative ~ {TOPIC}+{REG}', dR)
noReg  = smf.logit(f'negative ~ {TOPIC}', dR).fit(disp=0)
noTop2 = smf.logit(f'negative ~ {REG}', dR).fit(disp=0)
print(f'N={len(dR)}, regions(>=50)={len(keep)}, negative={int(dR.negative.sum())}')
print('LR Topic (adj. region):  chi2=%.2f df=%d p=%.3g' % lrtest(main2, noTop2))
print('LR Region (adj. topic):  chi2=%.2f df=%d p=%.3g' % lrtest(main2, noReg))
print('McFadden R2 (main-effects)=%.3f' % (1 - main2.llf / null2.llf))
try:
    full2 = fit(f'negative ~ {TOPIC}*{REG}', dR)
    print('LR TopicxRegion: chi2=%.2f df=%d p=%.3g' % lrtest(full2, main2))
except Exception:
    print('Topic x Region saturated logistic: NOT ESTIMABLE '
          '(empty cells / perfect separation) -> interaction reported as exploratory only')
