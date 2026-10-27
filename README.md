# Kredi Kartı Temerrüt Tahmini — Tabular Classification Projesi

Bu projede dengesiz bir tablo verisi üzerinde uçtan uca bir sınıflandırma hattı kuruyoruz. Hattın adımları:

1. Veri kalitesi kontrolü
2. Eksik veri doldurma (MVI)
3. İki yöntemle sentetik veri üretimi
4. Üç modelin karşılaştırılması: Perceptron, LightGBM + Optuna, TabNet

**Veri seti:** [Default of Credit Card Clients (UCI #350)](https://archive.ics.uci.edu/dataset/350)
**Ekip planı:** [`.claude/plans/ekip-is-paylasimi.plan.md`](.claude/plans/ekip-is-paylasimi.plan.md)

## Kısıtlar

Değerler `01_data_quality_mvi` notebook'u çalışınca dolacak.

| Kısıt | Gerekli | Bu proje |
|---|---|---|
| Satır | ≥ 20.000 | |
| Özellik | ≥ 20 | 23 |
| Kategorik oranı | ≥ %30 | 9/23 ≈ %39 |
| Eksik hücre | ≥ %5 | |
| Dengesizlik | ≥ 1:3 | |

## Kurulum

Python 3.10 veya 3.11 gerekli.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest                            # ortak sözleşme testleri
```

Ham veri ilk çalıştırmada `ucimlrepo` ile indirilir ve `data/raw/` klasörüne kaydedilir. Bu klasör git'e eklenmez.

## Çalıştırma sırası

| Notebook | Sahibi | Ürettikleri |
|---|---|---|
| `00_smoke_test` | Ortak | Kurulum kontrolü |
| `01_data_quality_mvi` | Öğrenci 1 | `data/processed/clean_*.parquet`, `nan_mask.parquet`, Grafik 1 |
| `02_synthetic_data` | Öğrenci 1 | `train_smote`, `train_ctgan`, `synth_ctgan_full`, Grafik 2-3 |
| `03_perceptron_lgbm` | Öğrenci 2 | `models/preds/{perceptron,lgbm}.csv`, önem tablosu |
| `04_tabnet` | Öğrenci 3 | `models/preds/tabnet.csv`, maskeler |
| `05_evaluation_plots` | Öğrenci 3 | Grafik 4-6, final tablo |

## Ortak sözleşme

Herkes veriyi aynı yoldan yükler:

```python
from src.data import load_split, to_xy

train = load_split("train", variant="v0")   # Öğrenci 1'in dosyaları gelene kadar ham veri
X, y = to_xy(train)
```

`variant` parametresinin alabileceği değerler:

| Değer | Anlamı | Hangi split'lerde |
|---|---|---|
| `v0` | Ham veri, NaN yok | train, val, test |
| `clean` | Doldurulmuş ve temizlenmiş, ölçeklenmemiş | train, val, test |
| `smote` | Artırılmış veri | Sadece train |
| `ctgan` | Artırılmış veri | Sadece train |

- **Sabit bölme:** Bölme `data/splits.csv` dosyasında sabit. Oranlar 70/15/15, stratified, `SEED=42`. Bu dosya değiştirilmez.
- **Ön işleme:** `src.features.make_preprocessor(kind)` kullanılır, `kind` = `"linear"`, `"tree"` veya `"tabnet"`. Ön işleyici her zaman sadece train'e fit edilir.
- **Tahmin dosyaları:** `src.metrics.save_preds()` ile yazılır. Kolonlar: `row_id, y_true, y_score, y_pred, threshold`. Eşik val setinde `pick_threshold` ile seçilir.
- **Ortak kod:** `src/` altındaki ortak dosyalar yalnızca PR ile değişir.
