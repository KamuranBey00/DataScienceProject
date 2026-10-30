# Plan: Öğrenci 2 — Perceptron + LightGBM Roadmap

**Sahibi:** Ö2 (Armağan) · **Branch:** `ArmaganBranch`
**Dayanak:** `ekip-is-paylasimi.plan.md` Bölüm 8 (Ö2), Bölüm 11 (soru bankası)
**Dosyalar:** `notebooks/03_perceptron_lgbm.ipynb`, `src/metrics.py`
**Çalışma şekli:** Her sprint'te önce kavram anlatılır, sonra kod hücre hücre birlikte yazılır.
Bir sprint "bitti" kriteri sağlanmadan sonrakine geçilmez.

**Veri kararı:** Ekip planındaki `v0` (ham veri) yalnızca Ö1'in verisi gelene kadar bir yedekti
(ekip planı satır 20, 273). Ö1 CP-A ve CP-B'yi `ogrenci1/veri` branch'inde bitirdi, bu yüzden
doğrudan `VARIANT = "clean"` ile başlanır. Artırılmış train setleri: `smote`, `ctgan`.

**Varsayımlar**
- Hocanın dokümanı (`Tabular_Data_Classification_Project_HW.docx`) ekip planıyla uyumlu.
- Ö1, `ogrenci1/veri` için main'e ayrıca PR açacak; biz o branch'i şimdiden buraya merge ediyoruz.

---

## Phase 0: Hazırlık

### Sprint 0.1: Ortam
- [x] `py -3.11 -m venv .venv`, torch CPU, `pip install -r requirements.txt`
- [x] `pytest` ve `00_smoke_test.ipynb` çalıştır (7 test geçti, özet `6552357087371317484` = Ö1'inki)
- **Bitti:** testler geçiyor, `splits` özeti ekip arkadaşlarıyla aynı

### Sprint 0.2: Branch ve Ö1'in verisi
- [x] `ArmaganBranch` main'den açıldı
- [ ] `origin/ogrenci1/veri` bu branch'e merge
- [ ] Ö1'e main'e PR açmasını hatırlat
- **Bitti:** `load_split("train", "clean")`, `"smote"`, `"ctgan"` çalışıyor

## Phase 1: Perceptron

### Sprint 1.1: Notebook iskeleti + baseline
- [ ] `03_perceptron_lgbm.ipynb`, en üstte `VARIANT = "clean"`
- [ ] `DummyClassifier(strategy="stratified")` metrikleri
- **Kavramlar:** macro-F1, PR-AUC, accuracy neden yanıltıcı

### Sprint 1.2: Perceptron taraması
- [ ] `Pipeline(make_preprocessor("linear"), Perceptron)`
- [ ] `eta0 × penalty × alpha × class_weight` taraması, val'de macro-F1 ile seçim
- [ ] Skor `decision_function`, eşik `pick_threshold`
- **Kavramlar:** Pipeline, sızıntı, `eta0` neden tek başına etkisiz

### Sprint 1.3: Öğrenme eğrisi
- [ ] `partial_fit` ile epoch bazlı train/val macro-F1 + hinge kaybı
- [ ] `figures/extra_perceptron_curve.png`

## Phase 2: LightGBM + Optuna

### Sprint 2.1: Default LGBM + CV fonksiyonu
- [ ] 5-fold `StratifiedKFold`, `n_estimators=2000`, `early_stopping(100)`, ortalama PR-AUC
- **Kavramlar:** gradient boosting, early stopping, `categorical_feature`

### Sprint 2.2: Optuna
- [ ] Arama uzayı (ekip planı Bölüm 8, Adım 3), `TPESampler(seed=SEED)`, `MedianPruner`
- [ ] Study `sqlite:///models/optuna_lgbm.db`; önce 10 deneme (süre ölçümü), sonra 50
- **Kavramlar:** TPE, pruner, neden PR-AUC optimize ediliyor

### Sprint 2.3: Çıktılar + erken teslim
- [ ] En iyi parametreler, CV ortalama ± std, `figures/extra_optuna_*.png`
- [ ] `models/importance/lgbm_importance.csv` (feature, gain, split)
- [ ] Clean tahminleri: `models/preds/{perceptron,lgbm}.csv` push (Ö3 grafikleri için)

## Phase 3: Varyantlar ve final

### Sprint 3.1: Varyant karşılaştırması
- [ ] 2 model × {clean, smote, ctgan}, yalnızca gerçek val'de; artırılmış veride CV yok
- [ ] Dengelenmiş veride `scale_pos_weight=1`
- [ ] `reports/train_variant_comparison.csv`
- [ ] Ö1'in bulgusuyla kıyas: sabit parametreli LGBM'de artırma kazanç sağlamadı
  (val Macro-F1 0,707 → SMOTE 0,698 / CTGAN 0,689, `reports/tstr_trtr.csv`)
- **Kavramlar:** SMOTE + CV sızıntısı, `scale_pos_weight` vs SMOTE

### Sprint 3.2: Final + CP-C
- [ ] En iyi varyantla final eğitim, eşik val'de, test'e tek tahmin
- [ ] Tahmin dosyaları + importance → PR (CP-C)

## Phase 4: Savunma

### Sprint 4.1: Ö1'in kodu
- [ ] `01_data_quality_mvi` hücre hücre; ters anlatım (ekip planı satır 271)

### Sprint 4.2: Prova
- [ ] Kendi notebook'unu süre tutarak anlat
- [ ] Soru bankası (Ö2 soruları)
- [ ] Restart & Run All, `nbqa flake8` temiz
