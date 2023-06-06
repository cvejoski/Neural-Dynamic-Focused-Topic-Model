#!/bin/bash

# # ACL dataset
# python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/acl -o data/preprocessed/acl/language -e embeddings/acl/skipgram_emb_300d.txt -min-df-tp 100  -tt-ratio 0.7 0.15  -max-doc-len 100 --num-workers 14
# # NIPS dataset
# python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/nips -o data/preprocessed/nips/language -e embeddings/glove.6B.300d.txt -min-df-tp 100  -tt-ratio 0.7 0.15  -max-doc-len 120 --num-workers 14
# # UN dataset
# python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/un -o data/preprocessed/un/language -e embeddings/un/skipgram_emb_300d.txt -min-df-tp 30  -tt-ratio 0.7 0.15 --split-by-paragraph --num-workers 14 --samples-per-timestep 200


for seed in {87..91}
do
    # ACL dataset
    python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/acl -o data/preprocessed/acl/language/$seed -e embeddings/acl/skipgram_emb_300d.txt -min-df-tp 10  -tt-ratio 0.7 0.15  --num-workers 14 --random-seed $seed
    # NIPS dataset
    # python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/nips/papers.csv -o data/preprocessed/nips/language/$seed -e embeddings/glove.6B.300d.txt -min-df-tp 10  -tt-ratio 0.7 0.15  --num-workers 14  --random-seed $seed
    # UN dataset
    python scripts/preprocessing/supervised_dynamic_language_dataset_preprocessor.py -i data/raw/un -o data/preprocessed/un/language/$seed -e embeddings/un/skipgram_emb_300d.txt -min-df-tp 30  -tt-ratio 0.7 0.15 --split-by-paragraph --num-workers 14 --random-seed $seed
done
