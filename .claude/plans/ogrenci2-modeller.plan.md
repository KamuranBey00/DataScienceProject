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
- [x] `origin/ogrenci1/veri` bu branch'e merge (çakışmasız, 20 test geçti, 5 varyant doğrulandı)
- [ ] Ö1'e main'e PR açmasını hatırlat
- **Bitti:** `load_split("train", "clean")`, `"smote"`, `"ctgan"` çalışıyor

## Phase 1: Perceptron

### Sprint 1.1: Notebook iskeleti + baseline
- [x] `03_perceptron_lgbm.ipynb`, en üstte `VARIANT = "clean"`
- [x] `DummyClassifier` metrikleri (val): stratified macro-F1 0,502 / PR-AUC 0,222;
  most_frequent accuracy 0,779 ama macro-F1 0,438
- **Kavramlar:** macro-F1, PR-AUC, accuracy neden yanıltıcı

### Sprint 1.2: Perceptron taraması
- [x] `Pipeline(make_preprocessor("linear"), Perceptron)`
- [x] `eta0 × penalty × alpha × class_weight` taraması, val'de macro-F1 ile seçim
  (ekip planından sapma: 5 tohum ortalaması, çünkü tohum varyansı ±0,05 > konfigürasyon farkları)
- [x] Skor `decision_function`, eşik `pick_threshold`
- Sonuç: l2, eta0=1e-3, alpha=1e-3 → macro-F1 0,636 ± 0,030, PR-AUC 0,407 (`reports/perceptron_sweep.csv`)
- Savunma notu: `eta0` sabit epoch'ta tamamen etkisiz (5 tohumda kanıtlı); sklearn'deki küçük
  farkın sebebi `tol`'un mutlak eşik olması → durma epoch'u değişiyor
- **Kavramlar:** Pipeline, sızıntı, `eta0` neden tek başına etkisiz

### Sprint 1.3: Öğrenme eğrisi
- [x] `partial_fit` ile epoch bazlı train/val macro-F1 + Perceptron kaybı (5 tohum × 50 tur)
- [x] `figures/extra_perceptron_curve.png`
- Sonuç: öğrenme 1. turda bitiyor (val 0,601 → 50. tur 0,583), train ≈ val → overfitting yok,
  underfitting + yakınsamama. Tavan ≈0,60–0,64.
- Tuzak notu: `random_state=int` ile `partial_fit` her tur aynı karıştırma sırasını kullanıyor
  (sklearn her çağrıda üreteci yeniden kuruyor) → `np.random.RandomState(seed)` nesnesi verildi

## Phase 2: LightGBM + Optuna

### Sprint 2.1: Default LGBM + CV fonksiyonu
- [x] 5-fold `StratifiedKFold`, `n_estimators=2000`, `early_stopping(100)`, ortalama PR-AUC
  (`cv_lgbm`, Optuna için `trial` ile fold bazlı pruning raporu)
- [x] Kategorik ayar CV'de ölçüldü: sayısal / nominal kat. / nominal+PAY kat. = 0,563 / 0,563 /
  0,561 (std 0,011) → anlamı doğru olan seçildi: nominal kategorik, PAY_x sayısal (sıralı)
- Tuzak notu: LightGBM kategorik kolondaki negatif değerleri eksik sayar (PAY_x −2/−1) →
  `CAT_LEVELS` ile sabit pandas `category` tipi
- Sonuç: varsayılan LGBM CV PR-AUC 0,563 ± 0,011 (59 ağaç); val macro-F1 0,707, ROC-AUC 0,778,
  PR-AUC 0,553 (Perceptron 0,636 / 0,676 / 0,407), Ö1'in 0,707 / 0,777'siyle örtüşüyor
- **Kavramlar:** gradient boosting, early stopping, `categorical_feature`

### Sprint 2.2: Optuna
- [x] Arama uzayı (ekip planı Bölüm 8, Adım 3), `TPESampler(seed=SEED)`, `MedianPruner`
- [x] Study `sqlite:///models/optuna_lgbm.db`; önce 10 deneme (süre ölçümü), sonra 50
- Sonuç: #32 → CV PR-AUC 0,567 ± 0,013 (varsayılan 0,563), 176 ağaç, lr 0,017, num_leaves 66,
  scale_pos_weight ≈ 1. Val macro-F1 0,711 (varsayılan 0,707), PR-AUC aynı → kazanç std altında
- Tuzak notu: pruner'a koşu *ortalaması* raporlanınca 50 denemede hiç kesme olmadı (pruner
  denemenin tüm adımlardaki en iyisini kullanır, ortalama fold'larla düşer) → kümülatif *toplam*
  raporlandı: 30/50 kesildi, 13 → 9 sn/deneme. Kesilen denemenin `value`'su da toplam olduğu
  için sıralama yalnızca COMPLETE denemelerle yapılır
- **Kavramlar:** TPE, pruner, neden PR-AUC optimize ediliyor

### Sprint 2.3: Çıktılar + erken teslim
- [x] En iyi parametreler, CV ortalama ± std, `figures/extra_optuna_{history,importance}.png`
  (plotly/kaleido kurulu değil → matplotlib, proje stiliyle)
- [x] `models/importance/lgbm_importance.csv` (feature, gain, split)
- Sonuç: PAY_0 gain'in %33'ü (split'te 15.), BILL_AMT1/LIMIT_BAL split'te 1-2 ama gain ≈%5.
  fANOVA düz (ilk üç ≈%18): skor yüzeyi düz, Optuna kazancının küçük olmasıyla tutarlı
- ~~Clean test tahminleri erken~~ → 3.2'ye bırakıldı (test'e yalnızca bir kez tahmin)

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
