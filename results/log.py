import pandas as pd
import os

for root, dirs, files in os.walk('./results'):
    dirs.sort()
    files.sort()

    for file in filter(lambda x: x.endswith('.csv'), files):
        result = os.path.join(root, file)
        df = pd.read_csv(result)

        CER_init = df['CER_init'].mean()
        CER_post = df['CER_post'].mean()
        CER_reduction = ((CER_init - CER_post) / CER_init) * 100

        WER_init = df['WER_init'].mean()
        WER_post = df['WER_post'].mean()
        WER_reduction = ((WER_init - WER_post) / WER_init) * 100

        print(result)
        print('CER_init:', round(CER_init, 4), 'CER_post:', round(CER_post, 4), 'CER %:', round(CER_reduction, 2))
        print('WER_init:', round(WER_init, 4), 'WER_post:', round(WER_post, 4), 'WER %:', round(WER_reduction, 2))
        print()