

Print by-relation performance:
```bash
 python eval_compute_acc_by_rel.py ../data/splits/random/test.jsonl ../tmp/lxmert_random_split/best_checkpoint/preds.txt
```
Output (format: relation accuracy relation__frequency):
```
touching	0.6356	236
behind	0.5956	136
on	0.6484	128
in front of	0.6638	116
under	0.6429	112
on top of	0.6322	87
...
	```

Print per meta-category performace:
```
python eval_compute_acc_by_rel_meta_cat.py ../data/splits/random/test.jsonl ../tmp/lxmert_random_split/best_checkpoint/preds.txt rel_meta_category_dict.txt
```
Output (format: category accuracy num_relation):
```
Projective	0.6158	773
Topological	0.6108	591
Proximity	0.5366	123
Adjacency	0.5317	284
Directional	0.5222	90
Unallocated	0.5098	51
Orientation	0.5000	112
```

