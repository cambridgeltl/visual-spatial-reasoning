# Analysis scripts
Using these scripts you can print by-relation and by-meta-category performances of the models.

**Print by-relation performance:**
```bash
 python eval_compute_acc_by_rel.py ../data/splits/random/test.jsonl ../tmp/lxmert_random_split/best_checkpoint/preds.txt
```
`sys.argv[1]` is the test jsonl file; `sys.argv[2]` points to the model's predictions (can be saved using `--save_preds` during evaluation).

Output (format: `f"{relation}\t{accuracy}\t{frequency}"`):
```
touching	0.6356	236
behind	0.5956	136
on	0.6484	128
in front of	0.6638	116
under	0.6429	112
on top of	0.6322	87
...
```

**Print per meta-category performace:**
```
python eval_compute_acc_by_rel_meta_cat.py ../data/splits/random/test.jsonl ../tmp/lxmert_random_split/best_checkpoint/preds.txt rel_meta_category_dict.txt
```
`sys.argv[1]` is the test jsonl file; `sys.argv[2]` points to the model's predictions; `sys.argv[3]` is the meta-category-to-relation dictionary.

Output (format: `f"{category}\t{accuracy}\t{frequency}"`):
```
Projective	0.6158	773
Topological	0.6108	591
Proximity	0.5366	123
Adjacency	0.5317	284
Directional	0.5222	90
Unallocated	0.5098	51
Orientation	0.5000	112
```

