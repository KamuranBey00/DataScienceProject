# Plan: Öğrenci 1 — Veri Kalitesi, MVI ve Sentetik Veri

**Kaynak plan:** `.claude/plans/ekip-is-paylasimi.plan.md` (Bölüm 6 ve 7)
**Seçilen kısım:** Aşama 0 (ortak iskelet, `main`) + Öğrenci 1 Hafta 1-2 (`ogrenci1/veri`)
**Karmaşıklık:** Orta
**Repo:** https://github.com/KamuranBey00/DataScienceProject

## Özet
İlk iş, diğer iki kişinin ilk günden çalışabilmesi için ortak iskeleti (`src/`, `splits.csv`, smoke test) `main`'e koymak. Ardından Öğrenci 1'in iki notebook'u `ogrenci1/veri` branch'inde hazırlanır: `01_data_quality_mvi` ve `02_synthetic_data`. Bu notebook'lar sözleşmedeki parquet dosyalarını üretir. Tekrar kullanılabilir mantık `src/` modüllerine konur ve testlerle doğrulanır. Notebook'lar yalnızca bu fonksiyonları çağırıp sonuçları gösterir.

## Uyulacak Kalıplar
Repo boş, taklit edilecek mevcut kod yok. Bu plan kalıpları kendisi belirliyor:

| Kategori | Kalıp |
|---|---|
| İsimlendirme | Modüller `snake_case`. Sabitler `config.py`'de BÜYÜK_HARF. Notebook'lar `NN_konu.ipynb` |
| Veri erişimi | Yalnızca `src.data.load_raw()` / `load_split()`. Yollar `config.py`'den gelir, hiçbir yere elle yol yazılmaz |
| Hatalar | Sözleşme ihlalinde açık `ValueError` / `FileNotFoundError`, sessiz geri dönüş yok |
| Testler | `tests/test_*.py`, pytest. Ağa bağlı olmayan küçük sentetik DataFrame'lerle |
| Stil | black + flake8 (max-line-length 100) |

## Değişecek Dosyalar

| Dosya | İşlem | Branch | Neden |
|---|---|---|---|
| `.gitignore`, `requirements.txt`, `setup.cfg`, `README.md` | OLUŞTUR | main | Aşama 0 |
| `src/__init__.py`, `config.py`, `data.py`, `features.py`, `metrics.py`, `plots.py` | OLUŞTUR | main | Ortak sözleşme |
| `data/splits.csv` | OLUŞTUR | main | Sabit bölme |
| `notebooks/00_smoke_test.ipynb` | OLUŞTUR | main | 3 makinede kurulum doğrulaması |
| `tests/test_data.py` | OLUŞTUR | main | Bölme determinizmi ve sızıntısızlık |
| `src/quality.py` | OLUŞTUR | ogrenci1/veri | Geçerlilik düzeltmeleri, aykırı değer tablosu |
| `src/missing.py` | OLUŞTUR | ogrenci1/veri | NaN ekleme, 3 imputer, NRMSE |
| `src/synthetic.py` | OLUŞTUR | ogrenci1/veri | KS, Wasserstein, TVD, korelasyon farkı |
| `tests/test_missing.py`, `tests/test_synthetic.py` | OLUŞTUR | ogrenci1/veri | Oran, korunan kolonlar, NaN kalmaması |
| `notebooks/01_data_quality_mvi.ipynb` | OLUŞTUR | ogrenci1/veri | Hafta 1 |
| `notebooks/02_synthetic_data.ipynb` | OLUŞTUR | ogrenci1/veri | Hafta 2 |
| `data/processed/*.parquet`, `reports/*.csv`, `figures/fig1-3*.png` | ÜRET | ogrenci1/veri | Sözleşme çıktıları |

## Görevler

### Görev 0: Aşama 0 iskeleti → `main`
- **Yapılacak:** venv kurulumu, sürüm sabitleme, `src/` sözleşme fonksiyonları, `make_splits()` ile `splits.csv`, smoke notebook'u, README iskeleti.
- **Doğrulama:** `pytest tests/test_data.py` geçer ve `00_smoke_test` hatasız çalışır.

### Görev 1: Kısıtlar ve veri kalitesi (01, Adım 1.1–1.2)
- **Yapılacak:** Kısıt tablosu, duplicate analizi, EDUCATION/MARRIAGE düzeltmesi, PAY kodlarının açıklaması, IQR aykırı değer tablosu ve boxplot.
- **Doğrulama:** `reports/constraints.csv` ve `reports/outliers.csv` üretilir.

### Görev 2: %8 eksik veri ekleme + Grafik 1 (Adım 1.3)
- **Yapılacak:** `inject_missing()` ile ~%5 MCAR + ~%3 MAR. Korunan kolonlar: LIMIT_BAL, AGE, PAY_0.
- **Doğrulama:** Toplam NaN ≥ %5 ve korunan kolonlarda 0 NaN (testle doğrulanır). `figures/fig1_missingness.png` üretilir.

### Görev 3: MVI karşılaştırması (Adım 1.4)
- **Yapılacak:** median/mode, KNN (ölçekli) ve MICE yöntemleri. Değerlendirme: val'de NRMSE ve kategorik doğruluk, LightGBM faydası, süre.
- **Doğrulama:** `reports/mvi_comparison.csv` üretilir ve imputer'lar yalnızca train'e fit edilir.

### Görev 4: Encoding/ölçekleme + clean çıktılar (Adım 1.5)
- **Yapılacak:** `make_preprocessor("linear")` gösterimi, `clean_{train,val,test}.parquet` ve `nan_mask.parquet`.
- **Doğrulama:** Clean dosyalarında 0 NaN, kategorikler geçerli kodlarda, satır kimlikleri `splits.csv` ile aynı.

### Görev 5: SMOTE-NC + CTGAN (02, Adım 2.1–2.2)
- **Yapılacak:** `train_smote.parquet`, `train_ctgan.parquet` (azınlık aynı oranda artırılır) ve `synth_ctgan_full.parquet`.
- **Doğrulama:** Şema clean ile aynı (+ `is_synthetic` kolonu), sınıf oranları yazdırılır.

### Görev 6: Kalite ölçümü + Grafik 2-3 (Adım 2.3–2.4)
- **Yapılacak:** `fidelity.csv`, `tstr_trtr.csv`, `fig2_real_vs_synth.png`, `fig3_corr_diff.png`.
- **Doğrulama:** `pytest tests/` geçer ve notebook'lar Restart & Run All ile hatasız çalışır.

## Doğrulama Komutları
```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_quality_mvi.ipynb
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_synthetic_data.ipynb
.venv/Scripts/python -m black --check src tests && .venv/Scripts/python -m flake8 src tests
```

## Riskler

| Risk | Olasılık | Önlem |
|---|---|---|
| KNNImputer 21k satırda yavaş | Orta | Süre ölçülür. Gerekirse `n_neighbors=5` ile sınırlanır |
| CTGAN CPU'da uzun sürer | Orta | Eğitilen model `models/ctgan.pkl`'de önbelleğe alınır. Olmazsa GaussianCopula |
| Repo sahibi KamuranBey00, push yetkisi olmayabilir | Orta | Push başarısız olursa collaborator daveti istenir |

## Kabul Kriterleri
- [ ] Görevler tamam
- [ ] pytest, black ve flake8 temiz
- [ ] Notebook'lar baştan sona çalışıyor, çıktılar commit'li
- [ ] `main`'de Aşama 0, `ogrenci1/veri`'de Öğrenci 1'in işleri push edilmiş
