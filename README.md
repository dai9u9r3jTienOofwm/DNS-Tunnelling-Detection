# DNS Tunneling Detection b?ng ML và 1D-CNN

## Cài d?t
`pip install -r requirements.txt`

## D? li?u
T?i d? li?u t? `https://github.com/aasthac67/DNS-Tunneling-Detection/tree/main/Tool` vào `data/raw`. Chu?n hóa thành CSV có hai c?t b?t bu?c: `domain` và `label` (0/1), r?i luu thành `data/processed/train.csv`.

## Ch?y
`python run.py --data data/processed/train.csv`

D? doán t?c thì: `python predict.py suspicious.example.com`

Ðánh giá: `python -m src.evaluate --data data/processed/test.csv`
