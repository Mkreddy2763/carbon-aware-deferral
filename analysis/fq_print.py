import json, glob
for f in sorted(glob.glob('data/forecasts/*_quality.json')):
    q = json.load(open(f))
    print(q['region'], q['calib_start'][:10], q['test_start'][:10], 'F2miss', q.get('F2_missing_frac_test'))
    for m, d in q['models'].items():
        mh = d['mae_by_h']
        print(f"  {m:4s} mae {d['mae']:6.1f} bias {d['bias']:6.1f} h1 {mh[0]:6.1f} h6 {mh[5]:6.1f} h24 {mh[23]:6.1f} h47 {mh[46]:6.1f}",
              ' | '.join(f"{k} {d[k]['coverage']:.3f}/{d[k]['width']:.0f}" for k in ['split_0.1', 'aci_0.1', 'split_0.2', 'aci_0.2']))
