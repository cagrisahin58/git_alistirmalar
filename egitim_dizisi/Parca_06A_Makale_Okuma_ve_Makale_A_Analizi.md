---
title: "Parça 6A — Akademik Makale Okuma Metodolojisi & Makale A Analizi"
subtitle: "Sıfırdan Adversarial Derin Öğrenme Araştırmacısı Yetiştirme Serisi"
author: "Eğitim Dizisi"
date: 2025
toc: true
toc-depth: 4
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.4
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
---

\pagebreak

# PARÇA 6A — Akademik Makale Okuma Metodolojisi & Makale A Analizi

> *"5 parça boyunca öğrendiğimiz her şey, bu iki makaleyi anlamamız içindi.
> Artık tüm yapı taşlarına sahibiz — şimdi onları birleştirme zamanı."*

---

## Ön Koşullar ve Bağlantılar

| Ön Koşul | Parça | Konu |
|-----------|-------|------|
| Lineer cebir, gradyan | 1A | Vektörler, matrisler, normlar, türev |
| Sinir ağları, backpropagation | 2A–2B | MLP, kayıp fonksiyonları, PyTorch |
| CNN mimarileri | 3A | Evrişim, ResNet, EfficientNet |
| Transformer & ViT | 3B | Attention, patch embedding, [CLS] |
| Adversarial saldırılar | 4A | FGSM, PGD, C&W, AutoAttack |
| Savunmalar & metrikler | 4B | Adversarial training, robust accuracy |
| Frekans analizi | 5A | FFT, DCT, frekans spektrumu |
| Açıklanabilirlik | 5B | Grad-CAM, Attention Maps, ECL |

---

## 6.1 Giriş & Motivasyon

### Yolculuğun Sonu, Araştırmanın Başlangıcı

Bir düşünün: Parça 1'de bir vektörün ne olduğunu öğreniyorduk. Şimdi ise adversarial pertürbasyonların frekans domeninde nasıl davrandığını, CNN ile ViT'in gradyan karakteristiklerinin neden farklı olduğunu anlayabilir durumdayız.

```
Parça 1: Matematik & Python
    │
    ▼
Parça 2: ML & Sinir Ağları
    │
    ▼
Parça 3: CNN & Transformer/ViT         ──► Makale A'nın MİMARİ temeli
    │
    ▼
Parça 4: Adversarial ML                ──► Makale A'nın ANA KONUSU
    │
    ▼
Parça 5: Frekans & Açıklanabilirlik    ──► Makale B'nin ANA KONUSU
    │
    ▼
Parça 6: ENTEGRASYON                   ──► Her şeyi birleştiriyoruz
    (Şimdi buradasınız!)
```

### Bu Parçada Ne Yapacağız?

1. **Akademik makale nasıl okunur?** — Bilimsel yayın okuma metodolojisi
2. **Makale A'yı bileşen bileşen çözeceğiz** — Her bir cümleyi öğrendiğimiz kavramlarla eşleştireceğiz
3. **Eleştirel düşünce** — Makalenin güçlü/zayıf yönlerini tartışacağız
4. (Parça 6B'de) Makale B'yi analiz edecek, kapanış projesi yapacağız

> 📌 **Makale Bağlantısı:**
> Bu parçanın doğrudan konusu **Makale A**: *"A Comparative Study of Convolutional and Transformer Architectures Under Adversarial Perturbations: Gradient Characteristics and Transferability Analysis"*

---

## 6.2 Akademik Makale Okuma Metodolojisi

### 6.2.1 Bir Araştırma Makalesi Nedir?

**Analoji:** Bir araştırma makalesi, bir dedektifin vaka raporuna benzer:

| Dedektif Raporu | Araştırma Makalesi | Açıklama |
|----------------|-------------------|----------|
| Vaka özeti | **Abstract** | 200-300 kelimelik tam özet |
| Suçun tanımı | **Introduction** | Problem nedir? Neden önemli? |
| Benzer vakalar | **Related Work** | Daha önce bu konuda ne yapılmış? |
| Soruşturma yöntemi | **Method** | Nasıl araştırdık? |
| Kanıtlar ve bulgular | **Experiments** | Ne bulduk? |
| Sonuç ve yorum | **Discussion / Conclusion** | Bu bulgular ne anlama geliyor? |
| Kaynaklar | **References** | Kime/neye dayandık? |

### 6.2.2 Makale Bölümlerinin Detaylı Anatomisi

```
┌─────────────────────────────────────────────────────┐
│                     BAŞLIK (Title)                    │
│          Yazarlar, Kurum, Tarih                       │
├─────────────────────────────────────────────────────┤
│  ABSTRACT (~200-300 kelime)                          │
│  ┌─────────────────────────────────────────────┐    │
│  │ Motivasyon → Problem → Yöntem → Sonuç → Etki │    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│  1. INTRODUCTION                                     │
│     • Geniş bağlam ve motivasyon                     │
│     • Spesifik problem tanımı                        │
│     • Mevcut yaklaşımların sınırlılıkları            │
│     • Bu çalışmanın katkıları (contributions)        │
│     • Makalenin organizasyonu                        │
├─────────────────────────────────────────────────────┤
│  2. RELATED WORK                                     │
│     • Konuyla ilgili önceki çalışmalar               │
│     • Bu çalışmanın farkı                            │
├─────────────────────────────────────────────────────┤
│  3. METHOD / PROPOSED APPROACH                       │
│     • Matematiksel formülasyon                       │
│     • Algoritma / mimari tasarım                     │
│     • Teorik analiz (varsa)                          │
├─────────────────────────────────────────────────────┤
│  4. EXPERIMENTS                                      │
│     • Deney kurulumu (setup)                         │
│     • Veri setleri, modeller, hiperparametreler      │
│     • Sonuç tabloları ve grafikler                   │
│     • Ablation study (bileşen analizi)               │
├─────────────────────────────────────────────────────┤
│  5. DISCUSSION / CONCLUSION                          │
│     • Bulguların yorumlanması                        │
│     • Sınırlılıklar                                  │
│     • Gelecek çalışma önerileri                      │
└─────────────────────────────────────────────────────┘
```

### 6.2.3 Üç Geçişli Okuma Stratejisi

Bir makaleyi tek geçişte anlamaya çalışmak, bir roman gibi baştan sona okumak gibidir — verimsizdir. Bunun yerine **üç geçişli strateji** kullanırız:

#### Birinci Geçiş: Kuş Bakışı (5-10 dakika)

**Hedef:** "Bu makale benim için uygun mu? Genel olarak ne yapılmış?"

| Oku | Atla |
|-----|------|
| Başlık, abstract, anahtar kelimeler | Detaylı matematik |
| Giriş paragrafının ilk/son cümlesi | Related work detayları |
| Tüm bölüm başlıkları | Kanıt/ispat |
| Tüm şekil ve tablo BAŞLIKLARI | Deney detayları |
| Sonuç (conclusion) | Referanslar |

**Birinci geçiş sonunda cevaplayabilmeniz gereken sorular:**
1. Bu makalenin ana konusu nedir?
2. Hangi problemi çözmeye çalışıyor?
3. Temel yaklaşımı ne? (tek cümle)
4. Ana sonuç ne? (tek cümle)
5. Bu makale hangi kategoriye giriyor? (deneysel/teorik/survey?)

#### İkinci Geçiş: Yapısal Anlama (30-60 dakika)

**Hedef:** "Detayları anlıyorum ama hepsini tekrar üretemem"

- Tüm makaleyi okuyun, ama detaylı kanıtları atlayın
- Her şekil ve tabloyu dikkatlice inceleyin
- Anlamadığınız terimlerin altını çizin
- Anahtar referansları not edin (sonra okumak için)
- Kendi kelimelerinizle 1 paragraflık özet yazın

**Dikkat edilecek noktalar:**
- Şekillerdeki eksen etiketleri, birimler
- Tabloların sütun başlıkları ve kalın yazılmış (bold) değerler
- Denklemlerin sezgisel anlamı (her terimin fiziksel karşılığı)

#### Üçüncü Geçiş: Eleştirel Analiz (1-2 saat)

**Hedef:** "Bu makaleyi detaylarıyla anlıyorum ve yeniden uygulayabilirim"

- Her varsayımı sorgulayın
- Denklemleri kendiniz türetin
- Deney tasarımını eleştirin
- Sonuçların iddialarla tutarlılığını kontrol edin
- "Bu çalışmayı nasıl genişletebilirim?" sorusunu sorun

### 6.2.4 Not Alma Şablonu

Her makale için şu şablonu doldurun:

```
╔══════════════════════════════════════════════════════╗
║  MAKALE NOT ŞABLONU                                  ║
╠══════════════════════════════════════════════════════╣
║  Başlık: ___________________________________________║
║  Yazarlar: _________________________________________║
║  Yıl/Venue: _______________________________________║
║                                                      ║
║  Problem: (1-2 cümle)                                ║
║  ___________________________________________________ ║
║                                                      ║
║  Yöntem: (1-2 cümle)                                 ║
║  ___________________________________________________ ║
║                                                      ║
║  Temel Sonuçlar:                                     ║
║  1. ________________________________________________ ║
║  2. ________________________________________________ ║
║  3. ________________________________________________ ║
║                                                      ║
║  Güçlü Yönler:                                       ║
║  • _________________________________________________ ║
║  • _________________________________________________ ║
║                                                      ║
║  Zayıf Yönler / Sorular:                             ║
║  • _________________________________________________ ║
║  • _________________________________________________ ║
║                                                      ║
║  Kendi Çalışmamla İlişki:                            ║
║  ___________________________________________________ ║
║                                                      ║
║  Önemli Referanslar (sonra okunacak):                ║
║  • _________________________________________________ ║
╚══════════════════════════════════════════════════════╝
```

> **💡 Özet Kutusu — Makale Okuma**
>
> - Bir makaleyi 3 geçişte okuyun: kuş bakışı → yapısal → eleştirel
> - Her şekil ve tabloyu dikkatlice inceleyin — genellikle ana mesajı taşırlar
> - Not şablonu kullanarak sistematik olarak bilgi çıkarın
> - Anlamadığınız şeylerin listesini tutun — araştırmanın doğal parçasıdır

---

## 6.3 Makale A: Bileşen Bileşen Analiz

### Makale A'nın Tam Başlığı

> **"A Comparative Study of Convolutional and Transformer Architectures Under Adversarial Perturbations: Gradient Characteristics and Transferability Analysis"**

Başlığı kelime kelime çözelim:

| Kelime/İfade | Anlam | Hangi Parçada Öğrendik? |
|-------------|-------|------------------------|
| Comparative Study | Karşılaştırmalı çalışma | — |
| Convolutional | CNN (evrişimli) mimariler | Parça 3A |
| Transformer | Transformer mimarisi | Parça 3B |
| Adversarial Perturbations | Adversarial pertürbasyonlar | Parça 4A |
| Gradient Characteristics | Gradyan özellikleri | Parça 1A + 4A |
| Transferability | Aktarılabilirlik | Parça 4A |

**Başlığın tek cümlelik çevirisi:**
*"Evrişimli ve Transformer mimarilerinin adversarial pertürbasyonlar altındaki karşılaştırmalı incelemesi: Gradyan özellikleri ve aktarılabilirlik analizi"*

---

### 6.3.1 Araştırma Sorusu ve Hipotezler

#### Ana Araştırma Sorusu

> **CNN ve Transformer (ViT) mimarileri adversarial pertürbasyonlara neden ve nasıl farklı tepki veriyor?**

Bu soru iki alt soruya ayrılır:

```
Ana Soru: CNN vs. ViT adversarial davranış farkları
    │
    ├──► Alt Soru 1: Gradyan Karakteristikleri
    │    "İki mimari sınıfının gradyanları nasıl farklı?
    │     Bu fark saldırı etkinliğini nasıl etkiliyor?"
    │
    └──► Alt Soru 2: Transferability (Aktarılabilirlik)
         "Bir mimari için üretilen adversarial örnek,
          diğer mimariyi ne ölçüde kandırabiliyor?
          Aynı aile içi vs. aileler arası fark var mı?"
```

#### Neden Bu Sorular Önemli?

**Teorik önemi:**

CNN ve ViT temelden farklı mekanizmalarla çalışır (Parça 3A-3B'den hatırlayın):

| Özellik | CNN | ViT |
|---------|-----|-----|
| Temel işlem | Yerel evrişim (convolution) | Global dikkat (self-attention) |
| İndüktif önyargı | Güçlü (locality, shift equivariance) | Zayıf (sadece patch yapısı) |
| Alıcı alan | Katman katman büyür | İlk katmandan itibaren global |
| Öğrenilen özellikler | Doku ağırlıklı (texture bias) | Şekil ağırlıklı (shape bias) |

*Bu temel farklılıklar, adversarial pertürbasyonlara tepkilerini de farklılaştırır.*

**Pratik önemi:**

- Savunma stratejileri mimari-bağımlı mı tasarlanmalı?
- Saldırganlar bir mimari için ürettikleri saldırıyı diğerine transfer edebilir mi?
- Hibrit mimariler (CNN + ViT) daha robust olabilir mi?

#### Hipotezler (Muhtemel)

Parça 3-4'te öğrendiklerimize dayanarak şu hipotezler formüle edilebilir:

| # | Hipotez | Dayanak |
|---|---------|---------|
| H1 | CNN gradyanları daha yerel (sparse), ViT gradyanları daha dağınık (distributed) | CNN yerel evrişim kullanır; ViT global attention |
| H2 | ViT, adversarial pertürbasyonlara CNN'den daha dayanıklıdır | ViT'in zayıf indüktif önyargısı, yerel pertürbasyonlara daha az duyarlı olabilir |
| H3 | Aynı mimari ailesi içi transfer daha yüksek | Benzer mimariler benzer karar sınırları öğrenir |
| H4 | CNN→ViT transferi, ViT→CNN transferinden düşüktür | Farklı özellik öğrenme mekanizmaları |

---

### 6.3.2 Deneysel Tasarım

#### Kullanılan Modeller

Makale A, hem CNN hem ViT ailesinden çeşitli modeller kullanır. Parça 3'te öğrendiğimiz mimarileri hatırlayalım:

**CNN Ailesi:**

```
┌──────────────────────────────────────────────────┐
│           CNN Model Ailesi                        │
│                                                    │
│  ResNet-50          VGG-19          DenseNet-121   │
│  ┌────┐            ┌────┐           ┌────┐        │
│  │skip│            │deep│           │dense│        │
│  │conn│            │    │           │conn │        │
│  └────┘            └────┘           └────┘        │
│  (2015)            (2014)           (2017)        │
│  25.6M param       144M param       8M param      │
│                                                    │
│  EfficientNet-B0                                   │
│  ┌────┐                                           │
│  │comp│  (2019, compound scaling)                  │
│  │scal│  5.3M param                                │
│  └────┘                                           │
└──────────────────────────────────────────────────┘
```

**ViT Ailesi:**

```
┌──────────────────────────────────────────────────┐
│         Transformer Model Ailesi                  │
│                                                    │
│  ViT-B/16         ViT-L/16        DeiT-B          │
│  ┌────┐           ┌────┐          ┌────┐          │
│  │base│           │lrge│          │dist│          │
│  │/16 │           │/16 │          │ill │          │
│  └────┘           └────┘          └────┘          │
│  (2020)           (2020)          (2021)          │
│  86M param        307M param      86M param       │
│                                                    │
│  Swin-T                                           │
│  ┌────┐                                           │
│  │shft│  (2021, shifted window attention)          │
│  │wind│  28M param                                 │
│  └────┘                                           │
└──────────────────────────────────────────────────┘
```

**Model Seçiminin Mantığı:**

| Seçim Kriteri | Açıklama |
|--------------|----------|
| Mimari çeşitlilik | Her iki aileden farklı derinlik ve tasarımlarda modeller |
| Parametre çeşitliliği | Küçük (5M) → Büyük (307M) arası |
| Yaygın kullanım | Literatürde standart benchmark modelleri |
| Karşılaştırılabilirlik | Benzer performans aralığında modeller |

#### Kullanılan Saldırı Yöntemleri

Parça 4A'dan hatırlayalım:

| Saldırı | Tipi | Norm | Formül | Parça |
|---------|------|------|--------|-------|
| **FGSM** | Tek adımlı | L∞ | $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$ | 4A §4.4 |
| **PGD** | İteratif | L∞ | $x_{t+1} = \Pi_\epsilon(x_t + \alpha \cdot \text{sign}(\nabla_x L))$ | 4A §4.5 |
| **C&W** | Optimizasyon | L₂ | $\min \||\delta\||_2 + c \cdot f(x+\delta)$ | 4A §4.6 |
| **AutoAttack** | Ensemble | L∞ | APGD-CE + APGD-DLR + FAB + Square | 4A §4.7 |

**Saldırı parametreleri (tipik değerler):**

| Parametre | Değer | Anlam |
|-----------|-------|-------|
| ε (L∞) | 4/255, 8/255, 16/255 | Pertürbasyon bütçesi |
| α (PGD adım boyutu) | 2/255 | Her iterasyondaki adım |
| PGD iterasyon | 20, 50 | Adım sayısı |
| C&W confidence | κ = 0, 20, 40 | Yanlış sınıflandırma güveni |

#### Veri Setleri

| Veri Seti | Görüntü Sayısı | Sınıf | Boyut | Kullanım |
|-----------|---------------|-------|-------|----------|
| **ImageNet** | ~1.2M eğitim, 50K validasyon | 1000 | 224×224 | Ana benchmark |
| **CIFAR-10** | 50K eğitim, 10K test | 10 | 32×32 | Hızlı deney |
| **CIFAR-100** | 50K eğitim, 10K test | 100 | 32×32 | İnce taneli sınıflandırma |

#### Değerlendirme Metrikleri

Parça 4B'den hatırlayalım:

| Metrik | Formül | Ne Ölçer? |
|--------|--------|-----------|
| **Clean Accuracy** | $\text{Acc}_{clean} = \frac{\text{Doğru tahmin (temiz)}}{N}$ | Normal performans |
| **Robust Accuracy** | $\text{Acc}_{robust} = \frac{\text{Doğru tahmin (saldırı altında)}}{N}$ | Adversarial dayanıklılık |
| **Attack Success Rate (ASR)** | $\text{ASR} = \frac{\text{Başarılı saldırı}}{N}$ | Saldırı etkinliği |
| **Transferability Rate** | $\text{TR} = \frac{\text{Transfer edilen başarılı saldırı}}{N}$ | Mimari arası aktarılabilirlik |

#### Deneysel Protokol

```
┌─────────────────────────────────────────────────────────────┐
│                    Deneysel Protokol                          │
│                                                               │
│  1. Tüm modelleri temiz veriyle eğit / pre-trained yükle     │
│     │                                                         │
│     ▼                                                         │
│  2. Her model için clean accuracy ölç                         │
│     │                                                         │
│     ▼                                                         │
│  3. Her (model, saldırı, ε) kombinasyonu için:                │
│     │   ┌─────────────────────────────────────────┐           │
│     │   │ a) Adversarial örnek üret                │           │
│     │   │ b) Robust accuracy ölç                   │           │
│     │   │ c) Gradyan istatistikleri topla           │           │
│     │   └─────────────────────────────────────────┘           │
│     │                                                         │
│     ▼                                                         │
│  4. Transferability deneyi:                                    │
│     ┌──────────────────────────────────────────────┐          │
│     │ Her kaynak model S için:                      │          │
│     │   S ile adversarial örnek üret                │          │
│     │   Her hedef model T için:                     │          │
│     │     T üzerinde saldırı başarısını ölç         │          │
│     │     → NxN Transferability matrisi oluştur     │          │
│     └──────────────────────────────────────────────┘          │
│     │                                                         │
│     ▼                                                         │
│  5. Analiz ve raporlama                                        │
└─────────────────────────────────────────────────────────────┘
```

**Kontrol Değişkenleri:**

Bilimsel bir deneyde, adil karşılaştırma için kontrol değişkenleri sabit tutulmalıdır:

| Kontrol Değişkeni | Sabit Değer | Neden? |
|-------------------|-------------|--------|
| Girdi boyutu | 224 × 224 | Tüm modeller aynı boyutta girdi alsın |
| Normalizasyon | ImageNet ortalaması/std | Aynı ön-işleme |
| Random seed | Sabit (42, 123 vb.) | Tekrarlanabilirlik |
| Donanım | Aynı GPU | Zamanlama karşılaştırması için |
| Pre-training | ImageNet-1K | Aynı veri ile ön-eğitim |

> 📌 **Makale Bağlantısı — Deney Tasarımı:**
> Makale A'nın güçlü yönlerinden biri sistematik deney tasarımıdır. Her mimari çifti
> için aynı saldırı parametreleri kullanarak adil karşılaştırma sağlar.

---

### 6.3.3 Gradyan Karakteristikleri Analizi

Bu, Makale A'nın **birinci temel katkısıdır.**

#### Gradyanın Adversarial Bağlamda Anlamı

Parça 1A'dan hatırlayın: gradyan $\nabla_x L$, kayıp fonksiyonunun girdi $x$'e göre en dik artış yönünü gösterir.

Adversarial saldırılar bu gradyanı kullanır:
- **FGSM:** Gradyanın işaretini alır → $\text{sign}(\nabla_x L)$
- **PGD:** Gradyanı iteratif olarak takip eder

Bu nedenle **gradyanın yapısı, saldırının doğasını belirler.**

#### CNN vs. ViT: Gradyan Farkları

```
     CNN Gradyanı                    ViT Gradyanı
     (Tipik Dağılım)                (Tipik Dağılım)

  ┌──────────────────┐          ┌──────────────────┐
  │  ░░░░░░░░░░░░░░  │          │  ░▒░▒░▒░▒░▒░▒░▒  │
  │  ░░░░████░░░░░░  │          │  ▒░▒░▒░▒░▒░▒░▒░  │
  │  ░░████████░░░░  │          │  ░▒░▒░▒░▒░▒░▒░▒  │
  │  ░░░░████░░░░░░  │          │  ▒░▒░▒░▒░▒░▒░▒░  │
  │  ░░░░░░░░░░░░░░  │          │  ░▒░▒░▒░▒░▒░▒░▒  │
  └──────────────────┘          └──────────────────┘
     YEREL (Sparse)                GLOBAL (Distributed)

  • Kenar ve doku bölgelerinde       • Tüm patch'lerde
    yoğunlaşır                         dağılır
  • Az sayıda pikselde                • Çok sayıda pikselde
    büyük gradyan                       küçük gradyan
  • Yüksek frekans ağırlıklı         • Karışık frekans
```

#### Gradyan İstatistikleri

Makale A, gradyan farklarını ölçmek için çeşitli istatistiksel metrikler kullanır:

**1. Gradyan Büyüklüğü (Magnitude)**

$$\text{Avg. Magnitude} = \frac{1}{d} \sum_{i=1}^{d} |\nabla_{x_i} L|$$

CNN'lerde bu değer az sayıda pikselde yüksek, çoğu pikselde düşüktür.
ViT'lerde daha uniform bir dağılım gösterir.

**2. Gradyan Seyrekliği (Sparsity)**

$$\text{Sparsity} = \frac{|\{i : |\nabla_{x_i} L| < \tau\}|}{d}$$

$\tau$ bir eşik değeridir. CNN gradyanları genellikle daha seyrek (sparse) olur.

**3. Gradyan Düzgünlüğü (Smoothness)**

Komşu pikseller arasındaki gradyan farkı:

$$\text{Smoothness} = \frac{1}{d} \sum_{i} ||\nabla_{x_i} L - \nabla_{x_{i+1}} L||$$

CNN gradyanları daha "pürüzlü" (non-smooth), ViT gradyanları daha "düzgün" (smooth) olma eğilimindedir.

**4. Gradyan Yönü Kosinüs Benzerliği**

Farklı girdiler için gradyanların yönsel tutarlılığı:

$$\text{cos\_sim}(\nabla_{x_1} L, \nabla_{x_2} L) = \frac{\nabla_{x_1} L \cdot \nabla_{x_2} L}{||\nabla_{x_1} L|| \cdot ||\nabla_{x_2} L||}$$

#### Tipik Bulgular

| Metrik | CNN | ViT | Yorumu |
|--------|-----|-----|--------|
| Gradyan seyrekliği | Yüksek | Düşük | CNN belirli bölgelere odaklanır |
| Gradyan düzgünlüğü | Düşük (pürüzlü) | Yüksek (düzgün) | ViT daha smooth gradyan üretir |
| Ortalama büyüklük | Yüksek (birkaç yerde) | Düşük (ama yaygın) | Farklı dağılım profilleri |
| Yönsel tutarlılık | Yüksek (benzer girdilerde) | Daha değişken | CNN daha öngörülebilir |

#### Bu Farkların Kaynağı

```
CNN'de Gradyan Akışı:                ViT'de Gradyan Akışı:

    Girdi → [Conv] → [Conv] → ...       Girdi → [Patch Embed]
                                                    │
    Gradyan yerel bağlantılar             ┌─────────┼──────────┐
    üzerinden geri akar.                  │  [Self-Attention]   │
    Evrişim çekirdeği boyutu              │    (Global)         │
    gradyanın yayılma alanını             │  Her patch diğer    │
    sınırlar.                             │  tüm patch'lerle    │
                                          │  bağlantılı         │
    → Gradyan YEREL kalır                 └──────────┬──────────┘
                                                     │
                                          → Gradyan GLOBAL yayılır
```

**Daha teknik açıklama:**

| Mekanizma | CNN | ViT |
|-----------|-----|-----|
| Temel katman | Convolution (3×3 veya 5×5 kernel) | Self-Attention (tüm patch'ler arası) |
| Gradyan yayılımı | Kernel boyutuyla sınırlı | Attention ağırlıkları üzerinden tüm patch'lere |
| Derin katmanlardan yüzeye | Her katmanda sadece komşu piksellere | Her katmanda tüm konumlara |
| Sonuç | Yerel, yoğun gradyan | Global, dağılmış gradyan |

> **💡 Kritik İçgörü:**
>
> CNN'in yerel gradyanı, saldırganın "az piksel değiştirerek" büyük etki yaratmasına olanak tanır.
> ViT'in global gradyanı, pertürbasyonu tüm görüntüye yayılmaya zorlar.
> Bu nedenle **ViT, yerel pertürbasyonlara doğal olarak daha dayanıklıdır.**

---

### 6.3.4 Transferability (Aktarılabilirlik) Analizi

Bu, Makale A'nın **ikinci temel katkısıdır.**

#### Transferability Nedir? (Hatırlatma)

Parça 4A'dan: Model A için üretilen adversarial örnek, Model B'yi de kandırabiliyorsa "transfer etti" deriz.

```
  Model A (Kaynak)                      Model B (Hedef)
  ┌──────────┐                          ┌──────────┐
  │          │   x_adv üretildi          │          │
  │  ResNet  │ ──── x_adv ────────────► │  ViT-B   │
  │          │   A kandırıldı            │          │
  └──────────┘                          └──────────┘
                                         B de kandırılırsa
                                         → TRANSFER BAŞARILI!
```

#### Transferability Matrisi

Makale A, tüm model çiftleri için bir NxN transferability matrisi oluşturur:

```
                         H E D E F   M O D E L
              ┌─────────┬─────────┬─────────┬─────────┬─────────┐
              │ ResNet  │  VGG    │ ViT-B   │ DeiT-B  │ Swin-T  │
 K  ┌────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
 A  │ ResNet │  100%*  │  ~65%   │  ~25%   │  ~28%   │  ~35%   │
 Y  ├────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
 N  │ VGG    │  ~60%   │  100%*  │  ~22%   │  ~25%   │  ~30%   │
 A  ├────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
 K  │ ViT-B  │  ~30%   │  ~28%   │  100%*  │  ~55%   │  ~45%   │
    ├────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
 M  │ DeiT-B │  ~32%   │  ~30%   │  ~50%   │  100%*  │  ~42%   │
 O  ├────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
 D  │ Swin-T │  ~38%   │  ~35%   │  ~48%   │  ~45%   │  100%*  │
 E  └────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
 L
    * Köşegen = beyaz kutu saldırı (kaynak = hedef)

  Renk kodu:  ░ Düşük (<30%)   ▒ Orta (30-50%)   █ Yüksek (>50%)
```

*Not: Yukarıdaki değerler tipik sonuçları göstermek için kullanılmıştır; gerçek değerler deneysel koşullara bağlıdır.*

#### Transferability Kalıpları

Matrise baktığımızda belirgin kalıplar ortaya çıkar:

**Kalıp 1: Aile İçi Transfer Daha Yüksek**

```
    CNN ──► CNN    :  ████████████████  (~60-65%)
    ViT ──► ViT    :  ████████████     (~45-55%)
    CNN ──► ViT    :  ██████           (~22-35%)
    ViT ──► CNN    :  ████████         (~28-38%)
```

**Neden?** Aynı ailedeki modeller benzer özellik temsilleri öğrenir:
- CNN'ler → doku tabanlı yerel özellikler
- ViT'ler → şekil tabanlı global özellikler

Adversarial pertürbasyon bu ortak özellikleri hedef aldığı için transfer olur.

**Kalıp 2: CNN→ViT Transfer Düşük**

CNN için üretilen adversarial pertürbasyon tipik olarak **yerel doku bozulması** içerir. ViT ise kararlarını daha çok **global şekil bilgisine** dayandırır. Bu nedenle CNN'e yönelik pertürbasyon ViT'i daha az etkiler.

**Kalıp 3: ViT→CNN Transfer Biraz Daha Yüksek (asimetri)**

ViT'in global gradyanı, pertürbasyonu tüm görüntüye yayar. Bu yaygın pertürbasyon, CNN'in doku algısını da bozabilir. Dolayısıyla ViT→CNN aktarımı, CNN→ViT aktarımından genellikle biraz daha yüksektir.

```
  Asimetri:  ViT ──► CNN  >  CNN ──► ViT
                              │
                              ▼
             Bu asimetri, iki mimarinin öğrendiği
             özellik temsillerinin farklılığından kaynaklanır
```

**Kalıp 4: Swin Transformer "Köprü" Etkisi**

Swin Transformer, shifted window attention ile yerellik ekler (CNN'e biraz benzer). Bu nedenle hem CNN hem ViT ile nispeten daha iyi transfer gösterir.

```
           CNN ailesi
            │    ▲
            │    │  Orta transfer
            ▼    │
        ┌─────────┐
        │ Swin-T  │ ◄── Hibrit özellikler
        └─────────┘     (yerel + global)
            │    ▲
            │    │  Orta transfer
            ▼    │
           ViT ailesi
```

> 📌 **Makale Bağlantısı — Transferability:**
> Makale A'nın en değerli katkılarından biri, bu sistematik transferability analizidir.
> Sonuçlar, adversarial savunma stratejilerinin mimari-spesifik olması gerektiğini gösterir.
> Ayrıca, farklı mimari ailelerinden oluşan bir ensemble'ın daha robust olabileceğini ima eder.

---

### 6.3.5 Sonuçların Yorumlanması

#### Ana Bulgular Özeti

| # | Bulgu | Önemi |
|---|-------|-------|
| 1 | CNN gradyanları yerel ve sparse; ViT gradyanları global ve distributed | Saldırı stratejileri mimari-bağımlı olmalı |
| 2 | ViT'ler CNN'lerden daha yüksek doğal robust accuracy gösterir | Shape bias, adversarial robustness'a katkı sağlar |
| 3 | Aile içi transfer oranı, aileler arası transferden anlamlı ölçüde yüksek | Mimari çeşitlilik, ensemble savunmada avantaj sağlar |
| 4 | CNN→ViT transferi asimetrik olarak düşük | ViT'ler CNN adversarial örneklerine doğal direnç gösterir |
| 5 | Saldırı gücü (ε) arttıkça transferability de artar | Güçlü saldırılar mimari farklarını aşar |
| 6 | Swin-T gibi hibrit modeller orta seviyede transfer gösterir | Hibrit tasarım, transfer dinamiklerini değiştirir |

#### Bulguların Büyük Resme Katkısı

```
┌───────────────────────────────────────────────────────────────┐
│                    BÜYÜK RESİM                                 │
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │   CNN Mimarisi   │     │   ViT Mimarisi   │                  │
│  │  • Yerel özellik │     │ • Global özellik │                  │
│  │  • Doku ağırlıklı│     │ • Şekil ağırlıklı│                  │
│  │  • Sparse gradyan│     │ • Dense gradyan   │                  │
│  └────────┬────────┘     └────────┬────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────────────────────────────┐                    │
│  │    Adversarial Pertürbasyona Tepki       │                   │
│  │  • CNN: Yerel doku bozulmasına duyarlı   │                   │
│  │  • ViT: Global şekil bozulmasına duyarlı │                   │
│  │  • Düşük çapraz transfer                 │                    │
│  └─────────────────────┬───────────────────┘                    │
│                        │                                         │
│                        ▼                                         │
│  ┌───────────────────────────────────────────────┐              │
│  │            Pratik Çıkarımlar                    │              │
│  │                                                 │              │
│  │  1. Savunma mimari-spesifik tasarlanmalı        │             │
│  │  2. CNN+ViT ensemble daha robust olabilir       │             │
│  │  3. Transferability testleri çapraz mimari       │             │
│  │     olmalı                                      │              │
│  │  4. Shape bias → daha iyi robustness            │             │
│  └───────────────────────────────────────────────┘              │
│                        │                                         │
│                        ▼                                         │
│  ┌───────────────────────────────────────────────┐              │
│  │         Makale B'ye Köprü                       │             │
│  │  "ViT'in doğal robustness'ı, onu medikal        │             │
│  │   görüntü sahtecilik tespiti için uygun kılar    │             │
│  │   → FE-ViT-ECL mimarisinin temeli"              │              │
│  └───────────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────┘
```

---

### 6.3.6 Eleştirel Değerlendirme

#### Güçlü Yönler

| # | Güçlü Yön | Açıklama |
|---|-----------|----------|
| 1 | **Sistematik karşılaştırma** | Tüm model çiftleri için tutarlı deney protokolü |
| 2 | **Çok yönlü saldırı analizi** | FGSM, PGD, C&W, AutoAttack — farklı saldırı paradigmaları |
| 3 | **Gradyan karakteristikleri** | Sadece sonuç değil, nedenleri de araştırılmış |
| 4 | **Pratik çıkarımlar** | Akademik bulguların savunma tasarımına dönüştürülmesi |
| 5 | **Transferability matrisi** | Eksiksiz N×N karşılaştırma |

#### Potansiyel Zayıf Yönler ve Sorular

| # | Soru / Eleştiri | Neden Önemli? |
|---|-----------------|---------------|
| 1 | Sadece sınıflandırma görevi test edilmiş mi? | Tespit (detection), segmentasyon gibi görevlerde farklı olabilir |
| 2 | Pre-training etkisi ayrıştırılmış mı? | ImageNet pre-trained vs. scratch: fark önemli olabilir |
| 3 | Farklı ε değerlerinde kalıplar değişiyor mu? | Küçük ε vs. büyük ε'da transfer dinamikleri farklı olabilir |
| 4 | Model boyutu etkisi izole edilmiş mi? | ResNet-18 vs. ResNet-152 arasında da fark olabilir |
| 5 | Yeni mimari aileleri dahil mi? | MLP-Mixer, ConvNeXt gibi yeni tasarımlar |
| 6 | Adversarial training sonrası gradyan değişimi? | Savunmalı modellerin gradyan karakteristikleri farklı olabilir |

#### Gelecek Çalışma Fikirleri

Bu eleştirel analizden araştırma soruları türetebiliriz:

1. **"Adversarial training, CNN ve ViT gradyan karakteristiklerini birbirine yaklaştırır mı?"**
   - Adversarial training sonrası gradyan istatistiklerini karşılaştır

2. **"Hibrit mimariler (CNN + Attention) en iyi robustness'ı sağlar mı?"**
   - ConvNeXt, CoAtNet gibi hibritleri teste dahil et

3. **"Frekans domain'de transfer kalıpları nasıl?"**
   - Pertürbasyonun frekans spektrumunu analiz et (Parça 5A bağlantısı)
   - → Bu tam olarak Makale B'nin motivasyonuna götürür!

---

## 6.3.7 Makale A'yı Kodla Keşfetme

### Mini Deney: CNN vs. ViT Gradyan Karşılaştırması

```python
"""
Makale A - Mini Replikasyon: CNN vs ViT Gradyan Analizi
Parça 6A Kod Atölyesi
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Model Yükleme
# ============================================================

def load_models():
    """CNN (ResNet-50) ve ViT (ViT-B/16) modellerini yükle."""

    # CNN: ResNet-50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.eval()

    # ViT: Vision Transformer Base/16
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.eval()

    return resnet, vit

# ============================================================
# 2. Görüntü Ön-İşleme
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_image(image_path):
    """Bir görüntüyü yükle ve ön-işle."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return img_tensor

# ============================================================
# 3. Gradyan Hesaplama
# ============================================================

def compute_input_gradient(model, x, target_class=None):
    """
    Bir model için girdi gradyanını hesapla.

    Bu fonksiyon FGSM'in temelini oluşturur:
    ∇_x L(θ, x, y)

    Args:
        model: PyTorch modeli
        x: Girdi tensörü [1, 3, H, W]
        target_class: Hedef sınıf (None ise tahmin edilen sınıf)

    Returns:
        gradient: Girdi gradyanı [1, 3, H, W]
    """
    x_input = x.clone().detach().requires_grad_(True)

    # İleri geçiş
    output = model(x_input)

    # Hedef sınıf belirleme
    if target_class is None:
        target_class = output.argmax(dim=1)

    # Kayıp hesaplama
    loss = nn.CrossEntropyLoss()(output, target_class)

    # Geri yayılım
    loss.backward()

    # Girdi gradyanı
    gradient = x_input.grad.data.clone()

    return gradient

# ============================================================
# 4. Gradyan İstatistikleri
# ============================================================

def gradient_statistics(grad, model_name="Model"):
    """
    Gradyan istatistiklerini hesapla ve raporla.

    Makale A'daki analiz metriklerini uyguluyoruz.
    """
    grad_np = grad.squeeze().cpu().numpy()  # [3, 224, 224]
    grad_abs = np.abs(grad_np)
    grad_flat = grad_abs.flatten()

    stats = {
        'model': model_name,
        # Ortalama büyüklük
        'mean_magnitude': grad_flat.mean(),
        # Maksimum büyüklük
        'max_magnitude': grad_flat.max(),
        # Standart sapma
        'std_magnitude': grad_flat.std(),
        # Seyreklik (sparsity): |gradyan| < threshold olan piksellerin oranı
        'sparsity_01': (grad_flat < 0.01 * grad_flat.max()).mean(),
        'sparsity_05': (grad_flat < 0.05 * grad_flat.max()).mean(),
        # L1 ve L2 norm
        'l1_norm': np.abs(grad_flat).sum(),
        'l2_norm': np.sqrt((grad_flat ** 2).sum()),
    }

    print(f"\n{'='*50}")
    print(f"  Gradyan İstatistikleri: {model_name}")
    print(f"{'='*50}")
    print(f"  Ortalama |∇x|        : {stats['mean_magnitude']:.6f}")
    print(f"  Maksimum |∇x|        : {stats['max_magnitude']:.6f}")
    print(f"  Std. Sapma           : {stats['std_magnitude']:.6f}")
    print(f"  Seyreklik (τ=1%)     : {stats['sparsity_01']:.2%}")
    print(f"  Seyreklik (τ=5%)     : {stats['sparsity_05']:.2%}")
    print(f"  L1 Norm              : {stats['l1_norm']:.4f}")
    print(f"  L2 Norm              : {stats['l2_norm']:.4f}")
    print(f"{'='*50}")

    return stats

# ============================================================
# 5. Gradyan Görselleştirme
# ============================================================

def visualize_gradients(grad_cnn, grad_vit, original_img=None):
    """
    CNN ve ViT gradyanlarını yan yana görselleştir.

    Makale A'daki Figure benzeri bir karşılaştırma.
    """
    # Gradyanları numpy'a çevir
    g_cnn = grad_cnn.squeeze().cpu().numpy()
    g_vit = grad_vit.squeeze().cpu().numpy()

    # Kanal bazında ortalama al (görselleştirme için)
    g_cnn_mean = np.mean(np.abs(g_cnn), axis=0)  # [224, 224]
    g_vit_mean = np.mean(np.abs(g_vit), axis=0)   # [224, 224]

    # Normalize et (aynı skala)
    max_val = max(g_cnn_mean.max(), g_vit_mean.max())
    g_cnn_norm = g_cnn_mean / max_val
    g_vit_norm = g_vit_mean / max_val

    # Çizim
    fig, axes = plt.subplots(1, 3 if original_img is not None else 2,
                              figsize=(15, 5))

    idx = 0
    if original_img is not None:
        # Orijinal görüntüyü göster
        img_np = original_img.squeeze().cpu().numpy()
        img_display = np.transpose(img_np, (1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)
        axes[idx].imshow(img_display)
        axes[idx].set_title('Orijinal Görüntü', fontsize=14)
        axes[idx].axis('off')
        idx += 1

    # CNN gradyanı
    im1 = axes[idx].imshow(g_cnn_norm, cmap='hot', vmin=0, vmax=1)
    axes[idx].set_title('CNN (ResNet-50) Gradyanı', fontsize=14)
    axes[idx].axis('off')
    plt.colorbar(im1, ax=axes[idx], fraction=0.046)
    idx += 1

    # ViT gradyanı
    im2 = axes[idx].imshow(g_vit_norm, cmap='hot', vmin=0, vmax=1)
    axes[idx].set_title('ViT-B/16 Gradyanı', fontsize=14)
    axes[idx].axis('off')
    plt.colorbar(im2, ax=axes[idx], fraction=0.046)

    plt.suptitle('Makale A: CNN vs ViT Gradyan Karşılaştırması',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 6. FGSM Saldırısı ve Transfer Testi
# ============================================================

def fgsm_attack(model, x, y, epsilon):
    """
    FGSM saldırısı uygula.
    (Parça 4A'dan hatırlayın: x_adv = x + ε * sign(∇_x L))
    """
    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    # Pertürbasyon: ε * sign(gradyan)
    perturbation = epsilon * x_adv.grad.data.sign()
    x_adversarial = x + perturbation

    # [0,1] aralığına kırp (normalize edilmiş veriye uyarlanabilir)
    x_adversarial = torch.clamp(x_adversarial, x.min(), x.max())

    return x_adversarial, perturbation

def test_transferability(models_dict, x, y, epsilon=8/255):
    """
    Transferability matrisi oluştur.

    Her kaynak model için adversarial örnek üret,
    her hedef model üzerinde test et.
    """
    model_names = list(models_dict.keys())
    n = len(model_names)
    transfer_matrix = np.zeros((n, n))

    for i, src_name in enumerate(model_names):
        src_model = models_dict[src_name]

        # Kaynak model ile adversarial örnek üret
        x_adv, _ = fgsm_attack(src_model, x, y, epsilon)

        for j, tgt_name in enumerate(model_names):
            tgt_model = models_dict[tgt_name]

            # Hedef model üzerinde test et
            with torch.no_grad():
                output = tgt_model(x_adv)
                pred = output.argmax(dim=1)
                # Saldırı başarılı mı? (tahmin != gerçek etiket)
                attack_success = (pred != y).float().mean().item()
                transfer_matrix[i, j] = attack_success

    return transfer_matrix, model_names

def plot_transfer_matrix(matrix, names):
    """Transferability matrisini ısı haritası olarak görselleştir."""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    # Etiketler
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_xlabel('Hedef Model', fontsize=12)
    ax.set_ylabel('Kaynak Model', fontsize=12)
    ax.set_title('Transferability Matrisi (FGSM, ε=8/255)',
                 fontsize=14, fontweight='bold')

    # Değerleri hücrelere yaz
    for i in range(len(names)):
        for j in range(len(names)):
            text_color = 'white' if matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{matrix[i,j]:.1%}',
                   ha='center', va='center',
                   color=text_color, fontsize=10)

    plt.colorbar(im, label='Attack Success Rate')
    plt.tight_layout()
    plt.savefig('transfer_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 7. Gradyan Dağılım Histogramı
# ============================================================

def plot_gradient_histogram(grad_cnn, grad_vit):
    """
    CNN ve ViT gradyan büyüklüklerinin dağılımını karşılaştır.

    Makale A'daki gradyan analizi görselleştirmesi.
    """
    g_cnn = np.abs(grad_cnn.squeeze().cpu().numpy()).flatten()
    g_vit = np.abs(grad_vit.squeeze().cpu().numpy()).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = np.linspace(0, max(g_cnn.max(), g_vit.max()), 100)
    ax1.hist(g_cnn, bins=bins, alpha=0.7, label='CNN (ResNet-50)',
             color='steelblue', density=True)
    ax1.hist(g_vit, bins=bins, alpha=0.7, label='ViT-B/16',
             color='coral', density=True)
    ax1.set_xlabel('|∇_x L| (Gradyan Büyüklüğü)', fontsize=12)
    ax1.set_ylabel('Yoğunluk', fontsize=12)
    ax1.set_title('Gradyan Büyüklüğü Dağılımı', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')

    # Kümülatif dağılım (CDF)
    g_cnn_sorted = np.sort(g_cnn)
    g_vit_sorted = np.sort(g_vit)
    cdf_cnn = np.arange(1, len(g_cnn_sorted)+1) / len(g_cnn_sorted)
    cdf_vit = np.arange(1, len(g_vit_sorted)+1) / len(g_vit_sorted)

    ax2.plot(g_cnn_sorted, cdf_cnn, label='CNN (ResNet-50)',
             color='steelblue', linewidth=2)
    ax2.plot(g_vit_sorted, cdf_vit, label='ViT-B/16',
             color='coral', linewidth=2)
    ax2.set_xlabel('|∇_x L| (Gradyan Büyüklüğü)', fontsize=12)
    ax2.set_ylabel('Kümülatif Olasılık', fontsize=12)
    ax2.set_title('Kümülatif Dağılım (CDF)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0, 0.96, '95. yüzdelik', fontsize=9, color='gray')

    plt.suptitle('Makale A: Gradyan Dağılımı Karşılaştırması',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 8. Ana Çalışma Akışı
# ============================================================

def main():
    """
    Tam deney akışı:
    1. Modelleri yükle
    2. Gradyanları hesapla ve karşılaştır
    3. FGSM saldırısı uygula
    4. Transfer testi yap
    """
    print("="*60)
    print("  MAKALE A — Mini Replikasyon")
    print("  CNN vs ViT Gradyan & Transferability Analizi")
    print("="*60)

    # GPU kullanımı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nKullanılan cihaz: {device}")

    # --- 1. Model Yükleme ---
    print("\n[1/5] Modeller yükleniyor...")
    resnet, vit = load_models()
    resnet = resnet.to(device)
    vit = vit.to(device)

    # --- 2. Test Görüntüsü ---
    print("[2/5] Test görüntüsü hazırlanıyor...")
    # Rastgele bir test görüntüsü (gerçek projede ImageNet'ten alınır)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224).to(device)
    # Veya gerçek görüntü: x = load_image("test_image.jpg").to(device)

    # Her iki modelin tahmini
    with torch.no_grad():
        pred_cnn = resnet(x).argmax(dim=1)
        pred_vit = vit(x).argmax(dim=1)
    print(f"  CNN tahmini: sınıf {pred_cnn.item()}")
    print(f"  ViT tahmini: sınıf {pred_vit.item()}")

    # --- 3. Gradyan Analizi ---
    print("\n[3/5] Gradyan analizi yapılıyor...")
    y = pred_cnn  # Gerçek etiket olarak CNN tahminini kullan

    grad_cnn = compute_input_gradient(resnet, x, y)
    grad_vit = compute_input_gradient(vit, x, y)

    stats_cnn = gradient_statistics(grad_cnn, "CNN (ResNet-50)")
    stats_vit = gradient_statistics(grad_vit, "ViT-B/16")

    # Karşılaştırma tablosu
    print("\n" + "="*60)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("="*60)
    print(f"  {'Metrik':<25} {'CNN':>12} {'ViT':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for key in ['mean_magnitude', 'max_magnitude', 'std_magnitude',
                'sparsity_01', 'sparsity_05', 'l1_norm', 'l2_norm']:
        val_cnn = stats_cnn[key]
        val_vit = stats_vit[key]
        if 'sparsity' in key:
            print(f"  {key:<25} {val_cnn:>11.2%} {val_vit:>11.2%}")
        else:
            print(f"  {key:<25} {val_cnn:>12.6f} {val_vit:>12.6f}")

    # --- 4. Görselleştirme ---
    print("\n[4/5] Gradyanlar görselleştiriliyor...")
    visualize_gradients(grad_cnn, grad_vit, x)
    plot_gradient_histogram(grad_cnn, grad_vit)

    # --- 5. Transferability ---
    print("\n[5/5] Transferability testi yapılıyor...")
    models_dict = {
        'ResNet-50': resnet,
        'ViT-B/16': vit,
    }

    matrix, names = test_transferability(models_dict, x, y, epsilon=8/255)
    plot_transfer_matrix(matrix, names)

    print("\n" + "="*60)
    print("  Deney tamamlandı!")
    print("  Çıktı dosyaları:")
    print("    • gradient_comparison.png")
    print("    • gradient_histogram.png")
    print("    • transfer_matrix.png")
    print("="*60)

if __name__ == "__main__":
    main()
```

### Kodu Anlama Rehberi

| Kod Bölümü | Ne Yapıyor? | Hangi Parçadan? |
|-----------|-------------|-----------------|
| `load_models()` | ResNet-50 ve ViT-B/16 yükleme | 3A (CNN), 3B (ViT) |
| `compute_input_gradient()` | $\nabla_x L$ hesaplama | 1A (gradyan), 2A (backprop) |
| `gradient_statistics()` | Seyreklik, norm, ortalama | 4A (gradyan analizi) |
| `fgsm_attack()` | $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$ | 4A (FGSM) |
| `test_transferability()` | N×N transfer matrisi | 4A (transferability) |
| Normalization | ImageNet ortalaması/std | 3A (ön-işleme) |

---

## Ara Özet — Makale A

> **💡 Özet Kutusu — Makale A Analizi**
>
> **Problem:** CNN ve ViT mimarileri adversarial pertürbasyonlara nasıl farklı tepki veriyor?
>
> **Yöntem:** Sistematik karşılaştırmalı deney — çoklu CNN ve ViT modelleri, çoklu saldırı yöntemleri, gradyan istatistikleri, transferability matrisi
>
> **Temel Bulgular:**
> 1. CNN gradyanları yerel ve sparse; ViT gradyanları global ve distributed
> 2. ViT doğal olarak daha robust (shape bias sayesinde)
> 3. Aile içi transfer > aileler arası transfer
> 4. CNN→ViT transferi asimetrik olarak düşük
>
> **Makale B'ye Köprü:** ViT'in doğal robustness'ı, onu medikal uygulamalar için uygun mimari yapar. Frekans analizi ile bu robustness daha da güçlendirilebilir.

---

## Kavram Haritası — Parça 6A

```
                    ┌──────────────────────────────┐
                    │   AKADEMİK MAKALE OKUMA       │
                    │   METODOLOJİSİ                │
                    │                                │
                    │  3 Geçiş: Kuş bakışı →        │
                    │  Yapısal → Eleştirel           │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                        MAKALE A ANALİZİ                          │
 │                                                                   │
 │   Araştırma Sorusu                                                │
 │   "CNN vs ViT: adversarial davranış farkları"                     │
 │         │                                                         │
 │         ├──────────────────┬─────────────────────┐               │
 │         ▼                  ▼                     ▼                │
 │   ┌──────────┐      ┌───────────┐        ┌───────────┐          │
 │   │ Gradyan  │      │ Transfer- │        │ Deney     │          │
 │   │ Analizi  │      │ ability   │        │ Tasarımı  │          │
 │   │          │      │           │        │           │          │
 │   │ Sparse   │      │ N×N matris│        │ Modeller  │          │
 │   │ vs Dense │      │ Aile içi  │        │ Saldırılar│          │
 │   │ Yerel    │      │ vs arası  │        │ Veri seti │          │
 │   │ vs Global│      │ Asimetri  │        │ Metrikler │          │
 │   └────┬─────┘      └─────┬─────┘        └─────┬─────┘          │
 │        │                  │                     │                │
 │        └──────────────────┼─────────────────────┘                │
 │                           │                                       │
 │                           ▼                                       │
 │                    SONUÇ: ViT daha robust,                        │
 │                    mimari çeşitlilik savunmada avantaj             │
 └──────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │  PARÇA 6B'YE KÖPRÜ:     │
                  │  Makale B (FE-ViT-ECL)  │
                  │  ViT + Frekans + ECL    │
                  │  → Medikal Uygulama     │
                  └─────────────────────────┘
```

---

## Sonraki Parçaya Köprü

Makale A bize şunu gösterdi: **ViT, adversarial pertürbasyonlara CNN'den doğal olarak daha dayanıklıdır.** Peki bu bilgiyi pratik bir uygulamaya nasıl dönüştürebiliriz?

**Makale B tam olarak bunu yapar:**
- ViT'in robustness avantajını medikal görüntü sahtecilik tespitine uygular
- Frekans analizi (DCT) ile bu robustness'ı güçlendirir
- Explanation Consistency Loss ile açıklanabilirlik ekler

**Parça 6B'de:**
1. Makale B'yi (FE-ViT-ECL) bileşen bileşen analiz edeceğiz
2. İki makale arasındaki bağlantıları çizeceğiz
3. Araştırma metodolojisi temellerini öğreneceğiz
4. Kapanış projesi ile tüm seriyi tamamlayacağız

---

*Parça 6A Sonu — Devam: Parça 6B (Makale B Analizi, Metodoloji & Kapanış)*
