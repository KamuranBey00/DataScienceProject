# Plan: Kredi Kartı Temerrüt Tahmini — 3 Kişilik Bağımsız İş Paylaşımı

**Kaynak:** `Tabular_Data_Classification_Project_HW.docx` (eski OMC taslağı: `archive/proje-yol-haritasi.omc-taslak.md`)
**Kapsam:** Tüm zorunlu maddeler + opsiyonel TabNet (Model 3)
**Karmaşıklık:** Orta
**Süre:** ~3,5 hafta. Aşama 0: 1 gün. Paralel çalışma: 2 hafta. Birleştirme ve sunum: 1 hafta.
**Durum:** onay bekliyor

---

## 1. Özet

Veri seti **Default of Credit Card Clients (UCI, id=350)**: 30.000 satır, 23 özellik, %39 kategorik, ≈1:3,5 dengesizlik. Veride eksik değer yok, %8 eksikliği kendimiz ekleyeceğiz. Ekleme MCAR ve MAR karışık olacak.

Üç kişinin birbirini beklemeden çalışabilmesi için **ilk gün 4 şey sabitlenir**:

1. **Sabit bölme dosyası.** `data/splits.csv` repoya commit'lenir. Herkes aynı train/val/test satırlarını kullanır.
2. **Dosya sözleşmesi.** Kim hangi dosyayı, hangi şemayla, ne zaman üretiyor (Bölüm 2.2).
3. **Ortak küçük kod (`src/`).** Yükleme, metrik ve kaydetme fonksiyonları.
4. **"v0" yedeği.** Öğrenci 1'in temiz verisi gelene kadar `load_split(..., variant="v0")` ham veriyi aynı bölmeyle döndürür. Ham veride NaN olmadığı için Öğrenci 2 ve 3 ilk günden model eğitebilir. Veri geldiğinde **tek bir parametre değişir**: `variant="clean"`.

Her kişi **yalnızca kendi dosyalarına** dokunur. Bu yüzden tek GitHub reposunda birleşirken merge çakışması olmaz.

---

## 2. Ortak Sözleşmeler (Aşama 0'da dondurulur)

### 2.1 Kolon şeması (`src/config.py`)

| Grup | Kolonlar | Not |
|---|---|---|
| ID | `row_id` | 0..N-1, duplicate'ler atıldıktan sonra verilir |
| Hedef | `default` | 0/1 (ham adı: `default payment next month` / `Y`) |
| Kategorik (nominal) | `SEX`, `EDUCATION`, `MARRIAGE` | One-hot (lineer model), int kod (ağaç ve TabNet) |
| Kategorik (ordinal durum) | `PAY_0`, `PAY_2`…`PAY_6` | −2…8 arası durum kodu |
| Sayısal | `LIMIT_BAL`, `AGE`, `BILL_AMT1`…`6`, `PAY_AMT1`…`6` | RobustScaler (yalnızca lineer model ve TabNet) |

`CAT_LEVELS` sözlüğü her kategorik kolonun **tüm geçerli kodlarını** sabit tutar. TabNet'in `cat_dims` değeri buradan gelir. Böylece test setinde görülmemiş bir kod çökme yaratmaz.

### 2.2 Dosya sözleşmesi

| Dosya | Şema | Üreten | Ne zaman | Tüketen | Gelene kadar ne yapılır |
|---|---|---|---|---|---|
| `data/splits.csv` | `row_id, split` (train/val/test = 70/15/15, stratified, SEED=42) | Aşama 0 | Gün 1 | Herkes | — |
| `data/processed/clean_{train,val,test}.parquet` | `row_id` + 23 özellik + `default`. NaN yok, kategorikler int kod, **ölçeklenmemiş** | Ö1 | **CP-A** (Hafta 1 sonu) | Ö2, Ö3 | `variant="v0"` (ham veri) |
| `data/processed/nan_mask.parquet` | `row_id` + 23 bool kolon | Ö1 | CP-A | Ö3 (Grafik 1'in final hali) | — |
| `data/processed/train_smote.parquet` | clean şeması + `is_synthetic` | Ö1 | **CP-B** (Hafta 2 ortası) | Ö2 (Ö3 opsiyonel) | Sadece clean ile çalışılır |
| `data/processed/train_ctgan.parquet` | clean şeması + `is_synthetic` | Ö1 | CP-B | Ö2 | 〃 |
| `data/processed/synth_ctgan_full.parquet` | clean şeması (tamamen sentetik, train boyutunda) | Ö1 | CP-B | Ö3 (Grafik 2-3'ün final hali) | — |
| `reports/mvi_comparison.csv`, `fidelity.csv`, `tstr_trtr.csv` | Tablolar | Ö1 | CP-A / CP-B | Ö1 (sunum), Ö3 (README) | — |
| `models/preds/{perceptron,lgbm,tabnet}.csv` | `row_id, y_true, y_score, y_pred, threshold` (**test** seti) | Ö2 / Ö3 | v0 sürümü Hafta 1 sonu, final **CP-C** | Ö3 | `make_dummy_preds()` (LogReg) |
| `models/importance/lgbm_importance.csv` | `feature, gain, split` (23 ham özellik) | Ö2 | CP-C | Ö3 (Grafik 4) | Dummy |
| `models/importance/tabnet_masks.npz`, `tabnet_global.csv` | adım × özellik maskeleri, global önem | Ö3 | Hafta 2 | Ö3 | — |
| `reports/train_variant_comparison.csv` | `model, variant, macro_f1, roc_auc, pr_auc` | Ö2 | CP-C | Ö1 (sunum) | — |

**Önemli:** LightGBM ve TabNet **23 ham kolonla** eğitilir (one-hot yok). Böylece Grafik 4'teki iki önem grafiği aynı özellik ekseninde yan yana durur.

### 2.3 Ortak kod (`src/`)

Aşama 0'da birlikte yazılır, toplam ~150 satır. Sonrasında **yalnızca küçük PR ile ve diğer ikisine haber verilerek** değişir.

| Dosya | İçerik | Sonraki sahibi |
|---|---|---|
| `config.py` | `SEED=42`, yollar, kolon listeleri, `CAT_LEVELS`, `TARGET` | Ö1 |
| `data.py` | `load_raw()` (ucimlrepo → `data/raw/` önbellek, kolon adlandırma, duplicate atma, `row_id`), `make_splits()`, `load_split(split, variant)` | Ö1 |
| `features.py` | `make_preprocessor(kind)`: `"linear"` = OneHot + RobustScaler, `"tree"` = passthrough, `"tabnet"` = sadece sayısallarda RobustScaler | Ö1 |
| `metrics.py` | `evaluate(y_true, y_score, y_pred)` → macro_f1, weighted_f1, roc_auc, pr_auc, accuracy. Ayrıca `pick_threshold(y_val, s_val)` (val'de Macro-F1'i maksimize eden eşik), `save_preds()`, `load_preds()` | Ö2 |
| `plots.py` | `set_style()`, `save_fig(name)` (dpi=200, `figures/`), `make_dummy_preds()` | Ö3 |

**Eşik kuralı (3 model için aynı):** Skor (`decision_function` veya `predict_proba`) → eşik val setinde `pick_threshold` ile seçilir → teste uygulanır.

---

## 3. Repo Yapısı ve Sahiplik

```
credit-default-tabular/
├── README.md                       # Ö3 (son hali), herkes kendi bölümünü yazar
├── requirements.txt                # Aşama 0
├── .gitignore                      # Aşama 0
├── .claude/                        # ECC ayarları + planlar
├── src/                            # Aşama 0 (bkz. 2.3)
├── notebooks/
│   ├── 00_smoke_test.ipynb         # Aşama 0
│   ├── 01_data_quality_mvi.ipynb   # Ö1
│   ├── 02_synthetic_data.ipynb     # Ö1
│   ├── 03_perceptron_lgbm.ipynb    # Ö2
│   ├── 04_tabnet.ipynb             # Ö3
│   └── 05_evaluation_plots.ipynb   # Ö3
├── data/
│   ├── raw/                        # .gitignore
│   ├── splits.csv                  # commit
│   └── processed/                  # commit (toplam birkaç MB)
├── models/{preds,importance}/      # commit
├── reports/                        # CSV tablolar, commit
├── figures/                        # 6 zorunlu grafik + ekler, commit
└── presentation/                   # Ö1 (slayt PDF)
```

## 4. Git Çalışma Düzeni

- **Branch'ler:** `ogrenci1/veri`, `ogrenci2/modeller`, `ogrenci3/tabnet-grafikler`. `main` her zaman çalışır durumda olmalı.
- **Kural:** Herkes sadece tabloda kendine yazılan dosyaları değiştirir. `src/` değişikliği ayrı ve küçük bir PR olur, diğer ikisi onaylar.
- **Checkpoint'te merge:** CP-A, CP-B ve CP-C'de sahibi PR açar, bir kişi review edip merge eder. Diğerleri `git pull origin main` yapar.
- **Commit formatı:** `feat(mvi): KNN ve MICE karşılaştırması`, `fix(tabnet): cat_dims düzeltmesi` gibi.
- **Notebook çıktıları silinmeden commit'lenir**, çünkü hoca çıktıları görecek. Merge çakışması olmaz, çünkü her notebook'un tek sahibi var.
- **Commit'lenmeyenler:** `.venv/`, `data/raw/`, `.ipynb_checkpoints/`, `__pycache__/`, büyük model dosyaları (`*.zip`, `*.pkl`).

## 5. Zaman Çizelgesi ve Senkron Noktaları

| Zaman | Öğrenci 1 | Öğrenci 2 | Öğrenci 3 | Senkron |
|---|---|---|---|---|
| **Gün 1** | Aşama 0 (birlikte) | Aşama 0 | Aşama 0 | Smoke test 3 makinede geçti |
| **Hafta 1** | 01: kalite, NaN ekleme, MVI, ölçekleme | 03: Perceptron + Optuna-LGBM (v0) | 04: TabNet kurulumu + grid (v0), `plots.py` | — |
| **H1 sonu** | `clean_*.parquet` PR | v0 tahminleri + importance push | — | **CP-A** (15 dk toplantı) |
| **Hafta 2 (1. yarı)** | 02: SMOTE-NC + CTGAN | Clean veride yeniden çalıştırma | Clean veride yeniden çalıştırma, 05 iskeleti | — |
| **H2 ortası** | `train_smote/ctgan` PR | — | — | **CP-B** |
| **Hafta 2 (2. yarı)** | Fidelity, TSTR/TRTR, Grafik 2-3 | Orijinal/SMOTE/CTGAN karşılaştırması, final tahminler | TabNet final + maskeler, Grafik 4-5-6 | — |
| **H2 sonu** | Tablolar push | Final preds PR | Final preds PR | **CP-C** |
| **Hafta 3** | Sunum (12-15 slayt) | Kod savunma hazırlığı (Ö1'in kodu dahil) | Final tablo, README, Restart & Run All, PEP-8 | Bilgi aktarım oturumu |
| **Hafta 3,5** | 2 süreli prova | Deneme savunma | Deneme savunma | Teslim |

---

## 6. Aşama 0: Ortak Kurulum (hep birlikte, ~3 saat)

1. **Repo:** Ö1 GitHub'da `credit-default-tabular` reposunu açar ve diğer ikisini collaborator olarak ekler. Herkes klonlar.
2. **Ortam:** Python 3.11, `python -m venv .venv`. `requirements.txt` içeriği: `pandas numpy pyarrow scipy scikit-learn imbalanced-learn sdv lightgbm optuna torch pytorch-tabnet missingno matplotlib seaborn ucimlrepo jupyter black flake8 nbqa`. Torch CPU sürümü yeterli.
3. **Versiyon sabitleme:** Kurulum herkeste çalıştıktan sonra Ö1 `pip freeze` ile sürümleri `requirements.txt`'e yazar. Herkes aynı sürümü kurar.
4. **Klasörler ve `.gitignore`:** Bölüm 3'teki yapı.
5. **`src/` dosyaları:** Bölüm 2.3'teki fonksiyonlar. Birlikte yazılır: biri yazar, diğer ikisi ekrandan takip eder.
6. **`make_splits()`:** Bir kez çalıştırılır (duplicate'ler atıldıktan sonra) ve `data/splits.csv` commit'lenir.
7. **`00_smoke_test.ipynb`:** Tüm import'lar, `load_split("train","v0").shape`, sınıf oranı ve `splits.csv`'nin md5 değeri.
8. **README iskeleti:** Proje amacı, kısıt tablosu (boş), çalıştırma sırası.

**Bitti sayılması için:** 3 bilgisayarda `00_smoke_test` hatasız çalışıyor. Split boyutları ve md5 değeri **aynı**.

---

## 7. Öğrenci 1: Data Architect & Quality Specialist

**Rubrik:** Veri kalitesi, MVI ve ön işleme (20) + sentetik veri (10) + sözlü sunum (16)
**Dosyalar:** `01_data_quality_mvi.ipynb`, `02_synthetic_data.ipynb`, `src/config.py`, `src/data.py`, `src/features.py`, `presentation/`

### Hafta 1: `01_data_quality_mvi.ipynb`

**Adım 1.1: Kısıt doğrulama (Gün 1, ~1 saat)**
- Veriyi `load_raw()` ile yükle. Şunları **kodla yazdır**: satır sayısı (≥20.000), özellik sayısı (23), kategorik oranı (9/23 = %39,1), sınıf dağılımı ve oranı (≈1:3,5).
- Sonucu tablo olarak `reports/constraints.csv`'ye kaydet. Bu tablo 8 puanlık "uyum" kanıtı.

**Adım 1.2: Veri kalitesi (Gün 1-2)**
- **Duplicate:** `df.drop(columns="row_id").duplicated().sum()`. Sayıyı raporla. Aşama 0'da atıldıklarını belirt.
- **Geçerlilik kontrolleri:**
  - `EDUCATION` ∈ {0,5,6} → 4 ("other"), `MARRIAGE` = 0 → 3 ("other"). Kaç satırın değiştiğini yaz.
  - `PAY_x` kodlarının anlamını markdown'da açıkla (−2: kullanım yok, −1: zamanında ödendi, 0: revolving kredi, 1..8: x ay gecikme).
  - Negatif `BILL_AMT` = alacak bakiyesi. Geçerli, korunur. Gerekçesini yaz.
  - `AGE` aralığı ve `LIMIT_BAL > 0` kontrolü.
- **Aykırı değerler:** Para kolonlarında IQR yöntemiyle aykırı değer yüzdesi tablosu ve log ölçekli boxplot. **Karar:** silme yok. Gerçek müşteri davranışı oldukları için RobustScaler ile yönetilecek. Bu karar sunumda anlatılacak.
- Temizleme sonrası kategorik kodları `CAT_LEVELS` ile karşılaştır, uyuşmuyorsa `config.py`'yi güncelle (PR).

**Adım 1.3: %8 eksik veri ekleme (Gün 2-3)**
- **Dokunulmayacak kolonlar:** `LIMIT_BAL`, `AGE`, `PAY_0`. MAR mekanizmasının "gözlenen" sürücüleri bunlar olacak.
- **MCAR (~%5):** Kalan 20 kolonda rastgele hücreler. `EDUCATION` ve `MARRIAGE` de dahil, böylece mode doldurması da test edilmiş olur.
- **MAR (~%3):**
  - `BILL_AMT2-6` için eksik olma olasılığı `AGE` ile artar (örn. lojistik fonksiyon).
  - `PAY_AMT2-6` için `LIMIT_BAL` düştükçe artar.
- Maske `SEED` ile **tüm veri setine** uygulanır. Maske öğrenilen bir şey olmadığı için sızıntı yaratmaz. Gerçek değerler `load_raw()`'dan her zaman geri alınabilir.
- Toplam NaN oranını yazdır (hedef ≈ %8, alt sınır %5). Kolon bazlı eksiklik tablosu ekle.
- **Grafik 1:** `msno.matrix` + `msno.heatmap` + `msno.dendrogram` → `figures/fig1_missingness.png`. Maske `nan_mask.parquet` olarak kaydedilir.

**Adım 1.4: MVI karşılaştırması (Gün 3-5)**
- **Yöntemler** (hepsi **sadece train'e fit**, val/test'e transform):
  1. *Baseline:* `SimpleImputer(median)` sayısallara, `SimpleImputer(most_frequent)` kategoriklere.
  2. *KNNImputer* (`n_neighbors` ∈ {5, 10}). Mesafe ölçekten etkilendiği için önce RobustScaler uygula, doldur, sonra geri dönüştür (inverse transform). Kategorikleri en yakın geçerli koda yuvarla.
  3. *IterativeImputer (MICE)* (`max_iter=10`, `estimator=BayesianRidge`, istenirse `ExtraTreesRegressor` de denenebilir). Kategorikleri yuvarla ve geçerli aralığa kırp.
- **Değerlendirme (3 eksen):**
  - (a) Maskelenen hücrelerde **NRMSE** (RMSE / kolon std'si, sayısallar için) ve **doğruluk** (kategorikler için).
  - (b) **Model faydası:** Varsayılan parametreli LightGBM'i doldurulmuş train'de eğit, val'de Macro-F1 ve ROC-AUC ölç.
  - (c) **Süre.**
- Sonuçlar `reports/mvi_comparison.csv`'ye yazılır. Opsiyonel: bir kolonda doldurma sonrası histogramı göster (median'ın oluşturduğu tepeyi görmek için).
- **En iyi yöntemi seç** ve gerekçesini 2-3 cümleyle yaz.

**Adım 1.5: Encoding ve ölçekleme (Gün 5)**
- `src/features.py::make_preprocessor()`'ü son haline getir: `"linear"` = OneHot(handle_unknown="ignore") + RobustScaler, `"tree"` = passthrough, `"tabnet"` = sayısallarda RobustScaler.
- Notebook'ta train'e fit edip val/test'e transform et, çıkan boyutları göster. "Neden sızıntı yok?" başlıklı bir markdown hücresi ekle.
- `clean_{train,val,test}.parquet` dosyalarını kaydet. Bunlar seçilen yöntemle doldurulmuş, temizlenmiş ve **ölçeklenmemiş** dosyalar.
- **PR aç → CP-A.**

### Hafta 2: `02_synthetic_data.ipynb` (yalnızca `clean_train` kullanılır)

**Adım 2.1: SMOTE-NC (Gün 6)**
- `SMOTENC(categorical_features=[SEX, EDUCATION, MARRIAGE, PAY_0..PAY_6], k_neighbors=5, random_state=SEED)`.
- Azınlık sınıfını 1:1'e çıkar (alternatif: 0,5). Seçilen oran CTGAN'da da aynı olacak.
- `train_smote.parquet` olarak kaydet: gerçek satırlar + sentetik satırlar, `is_synthetic` kolonuyla.

**Adım 2.2: CTGAN (Gün 6-8)**
- `SingleTableMetadata.detect_from_dataframe` ile metadata çıkar. Kategorik kolonları ve `default`'u `categorical` olarak işaretle.
- `CTGANSynthesizer(metadata, epochs=300)`. Önce 100 epoch ile süreyi ölç.
- İki çıktı üretilir:
  - (a) **Tam sentetik veri:** train boyutunda koşulsuz örnekleme → `synth_ctgan_full.parquet`. TSTR ve fidelity ölçümleri için.
  - (b) **Azınlık artırma:** `sample_from_conditions(Condition({"default": 1}, num_rows=n))` → `train_ctgan.parquet`. SMOTE ile **aynı oranda** artırılır ki karşılaştırma adil olsun.
- CTGAN çok yavaş kalırsa yedek plan: `GaussianCopulaSynthesizer`.

**Adım 2.3: Kalite ölçümü (Gün 8-9)**
- **İstatistiksel benzerlik (fidelity):**
  - Sayısal kolonlar: KS istatistiği ve p-değeri, Wasserstein mesafesi (standartlaştırılmış değerler üzerinde).
  - Kategorik kolonlar: Total Variation Distance.
  - Karşılaştırmalar:
    - **SMOTE:** sentetik azınlık satırları ↔ gerçek azınlık satırları. SMOTE yalnızca sınıf 1 ürettiği için böyle yapılır, bunu açıkça yaz.
    - **CTGAN:** tam sentetik ↔ gerçek train ve sentetik azınlık ↔ gerçek azınlık.
  - **Korelasyon farkı:** ortalama ve maksimum |ΔCorr|.
  - Sonuçlar `reports/fidelity.csv`'ye yazılır.
- **Fayda (TSTR / TRTR):** Varsayılan parametreli LightGBM kullanılır. Değerlendirme **gerçek val** setinde yapılır, test final modellere ayrılır.

  | Senaryo | Eğitim | Değerlendirme |
  |---|---|---|
  | TRTR | Gerçek train | Gerçek val |
  | TSTR | `synth_ctgan_full` | Gerçek val |
  | TRA-SMOTE | Gerçek + SMOTE | Gerçek val |
  | TRA-CTGAN | Gerçek + CTGAN azınlık | Gerçek val |

  Metrikler: Macro-F1, ROC-AUC, PR-AUC → `reports/tstr_trtr.csv`.
- **Trade-off yorumu** (sunum için):
  - SMOTE dağılımı yerel olarak korur ama yeni bilgi üretmez.
  - CTGAN yeni örnek üretir ama korelasyonu bozabilir.
  - TSTR < TRTR beklenir, aradaki fark "fayda kaybı"nı gösterir.

**Adım 2.4: Grafik 2 ve 3 (Gün 9)**
- **Grafik 2:** 6 önemli özellik (`PAY_0, LIMIT_BAL, BILL_AMT1, PAY_AMT1, AGE, PAY_2`) için 2×3 KDE/histogram ızgarası. Her panelde gerçek, SMOTE ve CTGAN üst üste → `figures/fig2_real_vs_synth.png`.
- **Grafik 3:** 1×2 heatmap: |Corr_real − Corr_SMOTE| ve |Corr_real − Corr_CTGAN|. Aynı renk ölçeği (`vmin=0`, ortak `vmax`) → `figures/fig3_corr_diff.png`.
- **PR → CP-B** (parquet dosyaları), sonra tabloları ve grafikleri push et.

### Hafta 3: Sunum (yalnız sunucu Ö1)
**Slayt akışı (12-15 slayt):**
1. Başlık
2. İş problemi: temerrüdün maliyeti, FN ve FP'nin iş açısından anlamı
3. Veri seti ve kısıt tablosu
4. Veri kalitesi bulguları
5. Eksiklik ekleme ve Grafik 1
6. MVI karşılaştırma tablosu
7. Encoding, ölçekleme ve sızıntı önlemi
8. SMOTE-NC ve CTGAN nasıl çalışır
9. Fidelity (Grafik 2 + KS/WD tablosu)
10. Grafik 3
11. TSTR/TRTR ve trade-off
12. Model sonuçları (final tablo + Grafik 5)
13. Hata analizi (Grafik 6, recall ve precision'ın iş açısından anlamı)
14. İş önerisi
15. Sınırlılıklar

- Ö2 ve Ö3, Hafta 3'ün başında Ö1'e model sonuçlarını 30 dakikada anlatır. Ö1 bu kısmı da sunacak.
- Süre tutularak **2 prova** yapılır.

**Ö1 için "bitti" kontrol listesi**
- [ ] Kısıt tablosu notebook çıktısında ve `reports/constraints.csv`'de var
- [ ] Duplicate, geçerlilik ve aykırı değer analizi kararlarıyla birlikte yazılı
- [ ] Toplam NaN ≥ %5 (hedef %8) yazdırılmış, MCAR ve MAR açıklanmış
- [ ] MVI tablosu: 3 yöntem × (NRMSE, kategorik doğruluk, Macro-F1, AUC, süre)
- [ ] Scaler ve imputer'lar sadece train'e fit edilmiş
- [ ] 2 sentetik yöntem, fidelity tablosu, TSTR/TRTR tablosu
- [ ] Grafik 1, 2, 3 `figures/` içinde
- [ ] `clean_*`, `train_smote`, `train_ctgan`, `synth_ctgan_full`, `nan_mask` main'de
- [ ] Sunum 2 kez prova edildi, süre tutuldu

---

## 8. Öğrenci 2: Classical & Tree-Based Model Specialist

**Rubrik:** Model mühendisliği 20 puanın Perceptron ve GBDT kısmı + kod savunması (16, Ö3 ile ortak)
**Dosyalar:** `03_perceptron_lgbm.ipynb`, `src/metrics.py`
**Dikkat:** Dokümana göre Ö2 kod savunmasında **ön işleme ve MVI kodunu da** satır satır anlatacak (Code Presenter 1). Bu nedenle Hafta 3'te Ö1'in kodunu öğrenmek bu planın bir parçası.

### Hafta 1: v0 verisiyle, kimseyi beklemeden

**Adım 1: İskelet (Gün 1)**
- `load_split(..., variant=VARIANT)` ile başla. `VARIANT = "v0"` notebook'un en üstünde tek satır olarak durur.
- Referans çizgisi: `DummyClassifier(strategy="stratified")` metrikleri. "Model rastgeleden ne kadar iyi?" sorusunun cevabı.

**Adım 2: Perceptron (Gün 1-2)**
- `Pipeline([make_preprocessor("linear"), Perceptron(max_iter=1000, random_state=SEED)])`.
- **Taranacak parametreler:**
  - `eta0` ∈ {1e-4, 1e-3, 1e-2, 0.1, 1}
  - `penalty` ∈ {None, "l2", "elasticnet"}
  - `alpha` ∈ {1e-4, 1e-3}
  - `class_weight` ∈ {None, "balanced"}
- Seçim val setinde Macro-F1'e göre yapılır.
- **Savunma notu:** `penalty=None` ve başlangıç ağırlıkları 0 iken `eta0` sadece ağırlıkları ölçekler, karar sınırı değişmez. Bu yüzden `eta0` penalty ile birlikte taranır. Bunu tabloyla göster.
- **Öğrenme takibi:** `partial_fit` ile epoch'a göre train ve val Macro-F1 ile hinge kaybı eğrisi → `figures/extra_perceptron_curve.png`. Rubrikteki "baseline performance tracking" maddesi bu.
- Skor olarak `decision_function` kullanılır (`predict_proba` yok). Eşik `pick_threshold` ile seçilir.

**Adım 3: LightGBM + Optuna (Gün 3-5)**
- Veri: `make_preprocessor("tree")`. 23 ham kolon, kategorikler `categorical_feature` ile verilir.
- **Amaç fonksiyonu:** train üzerinde 5-fold `StratifiedKFold`. Her fold'da `n_estimators=2000` + `early_stopping(100)`. Ortalama **PR-AUC** maksimize edilir (eşikten bağımsız, dengesiz veriye uygun).
- **Arama uzayı:**

  | Parametre | Aralık |
  |---|---|
  | `num_leaves` | 15–255 (log) |
  | `learning_rate` | 0.01–0.2 (log) |
  | `max_depth` | −1 veya 3–12 |
  | `min_child_samples` | 10–200 |
  | `subsample` | 0.5–1.0 (`subsample_freq=1` ile) |
  | `colsample_bytree` | 0.5–1.0 |
  | `reg_alpha`, `reg_lambda` | 1e-8–10 (log) |
  | `scale_pos_weight` | 1–4 |
- **Ayarlar:** `TPESampler(seed=SEED)`, `MedianPruner`. Study `sqlite:///models/optuna_lgbm.db`'ye kaydedilir, böylece kesilirse kaldığı yerden devam eder. v0'da 10 deneme, final'de 50 deneme.
- **Çıktılar:**
  - En iyi parametreler ve CV skoru (ortalama ± std)
  - `optuna.visualization` ile optimization history ve parametre önemi grafikleri → `figures/extra_optuna_*.png`
  - Feature importance (gain ve split) → `models/importance/lgbm_importance.csv`

**Adım 4: Erken teslim (Hafta 1 sonu)**
- v0 test tahminlerini `models/preds/perceptron.csv` ve `lgbm.csv` olarak push et. Böylece Ö3 gerçek dosyalarla grafik geliştirmeye başlar.

### Hafta 2: Gerçek veriyle

**Adım 5: CP-A sonrası (Gün 6-7)**
- `VARIANT = "clean"`. Perceptron taramasını ve 50 denemelik Optuna'yı yeniden çalıştır.

**Adım 6: CP-B sonrası: eğitim seti karşılaştırması (Gün 8-9)**
- Her model (Perceptron ve LGBM) × her varyant (`clean`, `smote`, `ctgan`) eğitilir.
- **Sızıntı uyarısı:** Artırılmış veride CV **yapılmaz**. SMOTE örnekleri val fold'undaki gerçek noktalardan türeyebilir ve skoru şişirir. Karşılaştırma yalnızca **gerçek val** setinde yapılır.
- Dengelenmiş veride `scale_pos_weight=1` kullanılır. İstenirse her varyant için 20 denemelik kısa bir Optuna çalıştırılır.
- Tablo: 2 model × 3 varyant × (Macro-F1, ROC-AUC, PR-AUC) → `reports/train_variant_comparison.csv`.

**Adım 7: Final (Gün 10)**
- Her model için en iyi varyantla final eğitim yap. Eşiği val'de seç. Test'e **tek bir kez** tahmin yap.
- `models/preds/perceptron.csv`, `lgbm.csv` ve importance CSV'sini kaydet → **PR → CP-C**.

### Hafta 3: Kod savunması hazırlığı
- **Ö1 ile 1 saatlik oturum:** `01_data_quality_mvi` hücre hücre anlatılır. Ö2 aynı notebook'u Ö1'e geri anlatır (ters anlatım).
- Kendi notebook'unu yüksek sesle, süre tutarak baştan sona anlat.
- Bölüm 11'deki soru bankasından kendi sorularını yanıtla.

**Ö2 için "bitti" kontrol listesi**
- [ ] Perceptron tarama tablosu, seçilen parametreler, öğrenme eğrisi
- [ ] Optuna: 50 deneme, en iyi parametreler, CV ortalama ± std, history grafiği
- [ ] Orijinal/SMOTE/CTGAN karşılaştırma tablosu (gerçek val üzerinde)
- [ ] 2 tahmin dosyası + importance CSV'si main'de
- [ ] Accuracy yalnızca bilgi amaçlı. Ana metrikler Macro-F1 ve PR-AUC
- [ ] 01 ve 03 notebook'larının her hücresini anlatabiliyor

---

## 9. Öğrenci 3: TabNet & Comparative Analytics Specialist

**Rubrik:** Model mühendisliği 20 puanın TabNet kısmı + grafikler ve değerlendirme (10) + kod savunması (16, Ö2 ile ortak) + repo hijyeni (8, liderlik)
**Dosyalar:** `04_tabnet.ipynb`, `05_evaluation_plots.ipynb`, `src/plots.py`, `README.md` (son hali)

### Hafta 1: v0 verisiyle

**Adım 1: Kurulum (Gün 1)**
- `import torch, pytorch_tabnet` çalışıyor mu kontrol et. Küçük bir örneklemle (2k satır) 5 epoch'luk deneme yap.

**Adım 2: Kategorik hazırlık (Gün 1-2)**
- Kategorik kolonlar (`SEX, EDUCATION, MARRIAGE, PAY_0..PAY_6`) `CAT_LEVELS` kullanılarak 0..k−1 aralığına map'lenir.
- `cat_idxs` ve `cat_dims` değerleri `CAT_LEVELS`'tan hesaplanır, **veriden hesaplanmaz**. Böylece görülmemiş bir kod çökme yaratmaz. `cat_emb_dim` 2 alınır.
- Sayısal kolonlar `make_preprocessor("tabnet")` ile ölçeklenir. Ölçekleyici train'e fit edilir.

**Adım 3: Tuning (Gün 2-4)**
- **Grid:**
  - `n_d = n_a` ∈ {8, 16, 32}
  - `n_steps` ∈ {3, 5}
  - `gamma` ∈ {1.3, 1.5}
  - `mask_type` ∈ {"sparsemax", "entmax"} (rubrikteki "mask tuning" maddesi)
  - `lambda_sparse` ∈ {1e-4, 1e-3}
- 48 kombinasyon fazla olursa önce `mask_type` ve `lambda_sparse` sabit tutulup 12 çalıştırma yapılır, sonra en iyi 2 konfigürasyonda bu ikisi taranır.
- **Eğitim ayarları:** `max_epochs=100`, `patience=20`, `batch_size=1024`, `virtual_batch_size=128`, `eval_metric=["auc","logloss"]` (val üzerinde), dengesizlik için `weights=1`.
- Sonuçlar `reports/tabnet_grid.csv`'ye yazılır.

**Adım 4: Eğitim eğrileri (Gün 4)**
- `clf.history`'den loss ve val AUC eğrileri → `figures/extra_tabnet_training.png`. Rubrikteki "loss curve optimization" maddesi bu.

**Adım 5: Grafik altyapısı (Gün 5)**
- `src/plots.py` içine şu fonksiyonlar yazılır:
  - `plot_importance_and_masks()` (Grafik 4)
  - `plot_roc_pr()` (Grafik 5)
  - `plot_cm_grid()` (Grafik 6)
  - `plot_missingness()`, `plot_kde_grid()`, `plot_corr_diff()` (1-3'ün final halleri)
- Ö2'nin v0 tahminleri gelene kadar `make_dummy_preds()` (LogReg) ile test edilir.

### Hafta 2

**Adım 6: CP-A sonrası (Gün 6-7)**
- `VARIANT = "clean"`, en iyi 3 konfigürasyonu yeniden çalıştır.
- Opsiyonel: TabNet'i `train_smote` ile de dene. Sonuç 2 satırlık bir tablo olur.

**Adım 7: Final TabNet (Gün 8)**
- En iyi konfigürasyonu eğit, eşiği val'de seç, test tahminini `models/preds/tabnet.csv` olarak kaydet.
- `explain_matrix, masks = clf.explain(X_test)`:
  - Adım bazlı maskelerin test ortalaması (`n_steps × 23`) → `tabnet_masks.npz`
  - `clf.feature_importances_` → `tabnet_global.csv`
- Model `clf.save_model("models/tabnet")` ile kaydedilir (gitignore'da).

**Adım 8: `05_evaluation_plots.ipynb` (Gün 8-10)**
- **Grafik 4:**
  - Sol panel: TabNet adım maskeleri heatmap'i (satırlar adımlar, sütunlar 23 özellik).
  - Orta panel: TabNet global önemi.
  - Sağ panel: LGBM gain önemi (toplamı 1'e normalize).
  - Özellik sırası tüm panellerde aynı → `figures/fig4_importance_masks.png`.
- **Grafik 5:**
  - Sol panel: ROC. 3 model, legend'da AUC değerleri, rastgele sınıflandırıcı köşegeni.
  - Sağ panel: PR. 3 model, legend'da AP değerleri, pozitif oran çizgisi.
  - → `figures/fig5_roc_pr.png`
- **Grafik 6:** 1×3 normalize confusion matrix (`normalize="true"`), ortak renk çubuğu, başlıkta kullanılan eşik → `figures/fig6_confusion.png`.
- **Grafik 1-3'ün final halleri:** Ö1'in parquet dosyalarından aynı stille yeniden çizilir (`figures/final/`). Doküman "tüm diagnostic grafikler" sorumluluğunu Ö3'e veriyor, savunmada bu kodlar da sorulabilir.
- **Final tablo:** model × (Macro-F1, Weighted-F1, ROC-AUC, PR-AUC, Accuracy\*) + eğitim varyantı + eşik → `reports/final_metrics.csv` ve README. \*Accuracy yalnızca bilgi amaçlı.
- **Opsiyonel bonus:** Test seti üzerinde 1000 bootstrap ile Macro-F1 ve PR-AUC için %95 güven aralığı.
- **PR → CP-C.**

### Hafta 3: Birleştirme lideri
- Temiz bir klonda `.venv` kur, notebook'ları 00 → 05 sırasıyla **Restart & Run All** ile çalıştır, hataları sahiplerine bildir.
- PEP-8: `nbqa black notebooks/`, `nbqa flake8 notebooks/`, `black src/`, `flake8 src/`.
- README'nin son hali:
  - Kısıt tablosu (gerçek sayılarla)
  - Kurulum ve çalıştırma sırası
  - 6 grafiğin küçük önizlemeleri
  - Final tablo
  - Kim neyi yaptı
- Kod savunması: 04 ve 05 notebook'larını ve `plots.py`'yi yüksek sesle anlatma provası.

**Ö3 için "bitti" kontrol listesi**
- [ ] TabNet grid tablosu (n_d/n_a, n_steps, gamma, mask_type), loss ve AUC eğrileri
- [ ] `tabnet.csv` tahminleri + maske dosyaları
- [ ] `figures/` içinde fig1…fig6 eksiksiz
- [ ] Final tablo: 3 model × 4 metrik, accuracy ana metrik değil
- [ ] Tüm notebook'lar temiz klonda sırayla hatasız çalışıyor
- [ ] black ve flake8 temiz, README tamam

---

## 10. Birleştirme Haftası (Hafta 3, ortak)

| Gün | İş | Kim |
|---|---|---|
| 1 | Sonuç aktarımı: Ö2 ve Ö3, model sonuçlarını Ö1'e anlatır (30 dk) | Hepsi |
| 1 | Bilgi aktarımı: Ö1, 01 notebook'unu Ö2'ye anlatır (1 saat) | Ö1 → Ö2 |
| 2 | Temiz klonda Restart & Run All, hata listesi | Ö3 |
| 2-3 | Hata düzeltmeleri (herkes kendi dosyasında) | Hepsi |
| 3 | Sızıntı denetimi: tüm `fit` çağrılarını tara, sadece train'e yapılmış mı | Ö2 + Ö3 çapraz kontrol |
| 4 | Sunum provası 1 + deneme kod savunması (birbirine soru sorulur) | Hepsi |
| 5 | Düzeltmeler, prova 2, `v1.0` tag'i, teslim | Hepsi |

## 11. Kod Savunması Soru Bankası

| Soru | Cevaplayan |
|---|---|
| Neden sızıntı yok? Hangi nesneler nereye fit edildi? | Ö2 |
| KNNImputer'dan önce neden ölçekleme yaptınız? | Ö2 |
| MICE nasıl çalışır, kaç iterasyon kullandınız? | Ö2 |
| MCAR ve MAR farkı nedir, sizin MAR mekanizmanız ne? | Ö2 (Ö1 sunumda) |
| Perceptron'da `eta0` neden tek başına etkisiz? | Ö2 |
| Neden accuracy kullanmadınız? | Ö2 / Ö3 |
| Optuna'da TPE ve pruner ne işe yarıyor? Neden PR-AUC'yi optimize ettiniz? | Ö2 |
| SMOTE'lu veride neden CV yapmadınız? | Ö2 |
| `scale_pos_weight` ile SMOTE arasındaki fark nedir? | Ö2 |
| n_d, n_a ve n_steps ne yapar, gamma neyi kontrol eder? | Ö3 |
| Sparsemax ve entmax farkı nedir, maskeler neden seyrek? | Ö3 |
| `cat_dims` nasıl belirlendi? | Ö3 |
| TabNet maskeleri ile LGBM önemleri neden farklı? | Ö3 |
| PR eğrisinde taban çizgisi neden 0,22? | Ö3 |
| Eşik nasıl seçildi, neden val setinde? | Ö3 |
| TSTR neden TRTR'den düşük? | Ö1 (sunum) |

## 12. Riskler ve Önlemler

| Risk | Olasılık | Önlem |
|---|---|---|
| sdv / torch / pytorch-tabnet sürüm uyumsuzluğu | Orta | Python 3.11, Aşama 0'da sürümler sabitlenir, smoke test 3 makinede geçmeli |
| CTGAN CPU'da yavaş | Orta | Önce 100 epoch. Olmazsa GaussianCopula (yine "2 farklı teknik" şartını sağlar) |
| Veri sızıntısı | Orta | Sabit `splits.csv`, tüm fit'ler train'de, artırılmış veride CV yok, Hafta 3'te çapraz denetim |
| CP-A gecikirse | Orta | Ö2 ve Ö3 v0 ile devam eder, geçiş tek satır. CP-A en geç Hafta 2'nin 2. günü |
| `src/` değişince başkasının kodu kırılır | Düşük | `src/` sadece PR ile değişir, fonksiyon imzaları sabit kalır |
| TabNet `cat_dims` hatası | Düşük | Değerler sabit `CAT_LEVELS`'tan gelir |
| Hoca PAY_x'in kategorik sayılmasına itiraz eder | Düşük | PAY_x durum kodu. Gerekirse one-hot yapılır, oran yine ≥ %30 kalır |
| Ö2, MVI kodunu savunamaz | Orta | Hafta 3 bilgi aktarımı + ters anlatım |

## 13. Teslim Öncesi Son Kontrol (rubrik eşlemesiyle)

- [ ] **(8)** 30.000 satır, 23 özellik, %39 kategorik, ≥%5 NaN ve ≥1:3 notebook'ta ve README'de yazdırılmış
- [ ] **(20)** Kalite analizi, baseline + 2 ileri MVI karşılaştırması, sızıntısız ölçekleme ve encoding
- [ ] **(10)** SMOTE-NC + CTGAN, KS/WD + TSTR/TRTR, Grafik 2-3
- [ ] **(20)** Perceptron (lr taraması + eğri), LGBM + Optuna + CV + importance, TabNet (n_d/n_a/n_steps/mask + loss eğrisi)
- [ ] **(10)** 6 grafik, Macro/Weighted-F1, ROC-AUC, PR-AUC ana metrikler
- [ ] **(16)** Sunum 2 kez prova edildi, süreye uyuldu
- [ ] **(16)** Ö2 ve Ö3 deneme savunma yaptı, soru bankası çalışıldı
- [ ] Tüm notebook'lar temiz klonda sırayla çalışıyor, PEP-8 temiz

## 14. ECC ile Çalışma (her kişi için)

- Bir adıma başlarken: `/ecc:plan <adım>` (örn. "MVI karşılaştırması").
- Commit öncesi: `/ecc:python-review` (PEP-8 ve idiomlar).
- Commit ve PR: `/ecc:prp-commit`, `/ecc:pr`. Checkpoint PR'larında review için `/ecc:code-review`.
- Hafta 3 sızıntı denetimi için ECC'nin `mle-reviewer` ajanından "data leakage" odaklı inceleme istenebilir.
