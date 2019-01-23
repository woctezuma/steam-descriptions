# Data

The file `aggregate.json` was created with [`aggregate_game_text_descriptions.py`](https://github.com/woctezuma/steam-api/blob/master/aggregate_game_text_descriptions.py) in my [`steam-api`](https://github.com/woctezuma/steam-api) Github repository.

The file `aggregate_prettyprint.json` is a pretty-print version, created by running:

```bash
python -m json.tool aggregate.json > aggregate_prettyprint.json
```

The file `tokens.json` results from:

```python
import json

from gensim.utils import simple_preprocess

with open('data/aggregate_prettyprint.json', 'r') as f:
    data = json.load(f)

tokens = {}
for app_id in data:
    tokens[app_id] = list(simple_preprocess(data[app_id]['text'], deacc=True))    
```
