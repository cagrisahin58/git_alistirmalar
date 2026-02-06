---
title: "Parça 2A: Makine Öğrenmesi & Sinir Ağlarının Temelleri"
subtitle: "Sıfırdan Adversarial Derin Öğrenme Araştırmacısı Yetiştirme Serisi"
author: "Eğitim Dizisi"
date: "2025"
geometry: "margin=2.5cm"
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{Parça 2A — Makine Öğrenmesi Temelleri}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Giriş ve Motivasyon

## Makinelerin "Öğrenmesi" Ne Demek?

**Analoji:** Bir çocuk düşün — ilk kez bir kedi görüyor. Ailesi "Bu kedi" diyor. Sonra farklı kediler görüyor: siyah kedi, beyaz kedi, tüylü kedi, tüysüz kedi... Zaman içinde hiç görmediği bir kediyi bile tanıyabiliyor. Çocuk "kediliğin özünü" öğrenmiş — bu **genelleme** (generalization).

Makine öğrenmesi de tam olarak budur: **örneklerden kalıpları öğrenmek ve yeni örneklere genellemek.**

Geleneksel programlama vs makine öğrenmesi:

```
Geleneksel Programlama:
┌──────────┐      ┌──────────┐      ┌──────────┐
│  Kurallar │  +   │   Veri   │  ──▶ │  Çıktılar │
└──────────┘      └──────────┘      └──────────┘
  (insan yazar)

Makine Öğrenmesi:
┌──────────┐      ┌──────────┐      ┌──────────┐
│   Veri   │  +   │ Çıktılar │  ──▶ │  Kurallar │
└──────────┘      └──────────┘      └──────────┘
                                    (makine öğrenir)
```

## Bu Parçanın Hedef Makalelerle Bağlantısı

> 📌 **Makale Bağlantısı:** Her iki makaledeki modeller, binlerce görüntü-etiket çiftiyle eğitilmiş **gözetimli öğrenme** sınıflandırıcılarıdır. Adversarial saldırılar, bu öğrenme sürecinin bir yan etkisini (gradyan bilgisini) istismar eder.

## Parça 1'den Köprü

Parça 1'de öğrendiğimiz araçlar şimdi hayata geçiyor:

| Parça 1 Kavramı | Parça 2'deki Rolü |
|:---|:---|
| Matris çarpımı | Sinir ağı katmanının temel işlemi: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ |
| Gradyan | Ağırlıkları güncellemek: $\theta \leftarrow \theta - \alpha \nabla L$ |
| Zincir kuralı | Backpropagation algoritması |
| Softmax | Çıktıyı olasılık dağılımına dönüştürme |

---

\newpage

# Öğrenme Paradigmaları

## Gözetimli Öğrenme (Supervised Learning)

**Analoji:** Bir öğretmen sınıfta — her soru için doğru cevabı veriyor. Öğrenci, soru-cevap çiftlerinden öğreniyor.

```
  Eğitim Verisi:
  ┌──────────────┬──────────┐
  │   Girdi (x)  │ Etiket(y)│
  ├──────────────┼──────────┤
  │  [görüntü 1] │   kedi   │
  │  [görüntü 2] │   köpek  │
  │  [görüntü 3] │   kedi   │
  │  [görüntü 4] │   kuş    │
  │     ...      │   ...    │
  └──────────────┴──────────┘

  Model öğrenir: f(görüntü) → etiket
```

İki alt türü:

| Tür | Çıktı | Örnek |
|:---|:---|:---|
| **Sınıflandırma** (classification) | Ayrık kategori | Kedi / Köpek / Kuş |
| **Regresyon** (regression) | Sürekli değer | Ev fiyatı: ₺1,250,000 |

## Gözetimsiz Öğrenme (Unsupervised Learning)

**Analoji:** Öğretmen yok — öğrenci kendi başına kitapları grupluyor: kalın olanlar, ince olanlar, renkli kapaklılar...

- **Kümeleme** (clustering): Benzer verileri gruplama
- **Boyut azaltma** (dimensionality reduction): Veriyi sıkıştırma

## Pekiştirmeli Öğrenme (Reinforcement Learning)

**Analoji:** Bir köpek eğitmek — doğru davranışta ödül, yanlışta ceza. Deneme-yanılma ile öğrenme.

## Paradigma Karşılaştırma Tablosu

| Paradigma | Veri | Sinyal | DL Örneği |
|:---|:---|:---|:---|
| Gözetimli | Etiketli çiftler | Doğru cevap | Görüntü sınıflandırma |
| Gözetimsiz | Etiketsiz veri | Kalıp keşfi | Kümeleme, GAN |
| Pekiştirmeli | Ödül/ceza | Gecikmeli sinyal | Oyun oynama |
| Öz-gözetimli | Verinin kendisi | Maskeleme/tahmin | BERT, MAE |

> 📌 **Makale Bağlantısı:** Her iki hedef makaledeki modeller **gözetimli sınıflandırma** problemi üzerinde çalışır — etiketli görüntü-sınıf çiftleriyle eğitilir.

> **Öğrendiklerimiz — Öğrenme Paradigmaları**
>
> ✅ Gözetimli öğrenme: girdi-çıktı çiftlerinden öğrenme (hedef makalelerin yöntemi)
> ✅ Sınıflandırma: ayrık kategorilere ayırma
> ✅ Genelleme: görülmemiş veriye doğru tahmin yapabilme

---

\newpage

# Perceptron'dan Çok Katmanlı Ağlara

## Biyolojik Nöron → Yapay Nöron

```
  Biyolojik Nöron:                    Yapay Nöron:

     dendritler                          girdiler
     (giriş)                             (x₁, x₂, ..., xₙ)
        │                                    │
        ▼                                    ▼
  ┌──────────┐                         ┌──────────┐
  │  Hücre   │                         │  Toplama  │ Σ = w₁x₁ + w₂x₂ + ... + b
  │  Gövdesi │                         │    +      │
  └────┬─────┘                         │ Aktivasyon│ y = σ(Σ)
       │                               └────┬─────┘
       ▼                                    │
     akson                                  ▼
    (çıkış)                              çıktı (y)
```

## Perceptron — İlk Yapay Nöron (1958)

Frank Rosenblatt'ın perceptron'u, en basit yapay nörondur:

$$y = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b) = \text{sign}\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

```
     x₁ ──── w₁ ───╲
                     ╲
     x₂ ──── w₂ ─────▶ [Σ + b] ──▶ [sign] ──▶ y ∈ {-1, +1}
                     ╱
     x₃ ──── w₃ ───╱

     Girdiler  Ağırlıklar  Toplama    Aktivasyon   Çıktı
```

### Karar Sınırı (Decision Boundary)

Perceptron, girdi uzayında bir **doğru** (2D) veya **düzlem** (3D) çizer:

$$w_1 x_1 + w_2 x_2 + b = 0$$

```
     x₂
     │    - - - - ╱ + + +
     │   - - - -╱ + + + +
     │  - - - -╱ + + + + +       ╱ = karar sınırı
     │ - - - ╱ + + + + + +
     │- - -╱ + + + + + + +       - = sınıf 0
     │- -╱ + + + + + + + +       + = sınıf 1
     └──╱───────────────── x₁
```

### Perceptron Öğrenme Algoritması

```python
import numpy as np

def perceptron_egit(X, y, ogrenme_hizi=0.1, epoch_sayisi=100):
    """Basit perceptron eğitimi."""
    n_ozellik = X.shape[1]
    w = np.zeros(n_ozellik)  # Ağırlıkları sıfırla
    b = 0.0                   # Bias

    for epoch in range(epoch_sayisi):
        hatalar = 0
        for i in range(len(X)):
            # İleri hesaplama (forward pass)
            z = np.dot(w, X[i]) + b
            tahmin = 1 if z >= 0 else 0

            # Güncelleme (hata varsa)
            hata = y[i] - tahmin
            if hata != 0:
                w += ogrenme_hizi * hata * X[i]
                b += ogrenme_hizi * hata
                hatalar += 1

        if hatalar == 0:
            print(f"Epoch {epoch}: Tüm örnekler doğru sınıflandı!")
            break

    return w, b

# AND kapısı
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

w, b = perceptron_egit(X, y)
print(f"Ağırlıklar: {w}, Bias: {b}")
# Ağırlıklar: [0.1 0.1], Bias: -0.1
```

## XOR Problemi — Perceptron'un Sınırı

```
  AND (lineer ayrılabilir ✓)       XOR (lineer AYRILAMAZ ✗)

  x₂                               x₂
  1 │ ○         ●                   1 │ ●         ○
    │     ╱                           │
    │   ╱                             │    ???
    │ ╱                               │
  0 │○──╱────── ○                   0 │ ○         ●
    └───────────── x₁                 └───────────── x₁

  Tek doğru yeterli!                Tek doğruyla AYRILAMAZ!
```

XOR problemi, tek katmanlı perceptron'un çözemediğini gösterir. Bu, derin öğrenmenin **neden "derin" olduğunun** tarihsel motivasyonudur.

> **Çözüm:** Birden fazla katman kullan → **Çok Katmanlı Algılayıcı (MLP)**

## Çok Katmanlı Algılayıcı (Multi-Layer Perceptron / MLP)

Birden fazla katman, doğrusal olmayan (non-linear) karar sınırları çizebilir:

```
  Girdi         Gizli Katman 1    Gizli Katman 2     Çıktı
  Katmanı        (hidden)          (hidden)          Katmanı

   x₁ ─────┬────── h₁⁽¹⁾ ────┬────── h₁⁽²⁾ ────┬────── ŷ₁
            ├────── h₂⁽¹⁾ ────┤────── h₂⁽²⁾ ────┤────── ŷ₂
   x₂ ─────┤────── h₃⁽¹⁾ ────┤────── h₃⁽²⁾ ────┘
            └────── h₄⁽¹⁾ ────┘
   x₃ ─────┘

   (3)           (4)               (3)               (2)
  3 girdi     4 gizli nöron     3 gizli nöron     2 çıktı sınıfı
```

### İleri Hesaplama (Forward Pass)

Her katmandaki işlem:

$$\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

burada:
- $\mathbf{W}^{(l)}$: $l$. katmanın ağırlık matrisi (Parça 1'deki matris çarpımı!)
- $\mathbf{b}^{(l)}$: bias vektörü
- $\sigma$: aktivasyon fonksiyonu (doğrusal olmama sağlar)

**Sayısal Örnek:** 2 girdili, 2 gizli nöronlu, 1 çıktılı ağ:

```python
import numpy as np

# Ağırlıklar ve biaslar
W1 = np.array([[0.2, 0.4],    # Gizli katman ağırlıkları (2×2)
               [0.6, 0.8]])
b1 = np.array([0.1, 0.1])

W2 = np.array([[0.3, 0.7]])   # Çıktı katmanı ağırlıkları (1×2)
b2 = np.array([0.1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# İleri hesaplama
x = np.array([1.0, 0.5])  # Girdi

# Katman 1
z1 = W1 @ x + b1          # [0.2*1 + 0.4*0.5 + 0.1, 0.6*1 + 0.8*0.5 + 0.1]
h1 = sigmoid(z1)           # = [0.5, 1.1] → sigmoid → [0.622, 0.750]
print(f"z1 = {z1}")        # [0.5 1.1]
print(f"h1 = {h1}")        # [0.622 0.750]

# Katman 2
z2 = W2 @ h1 + b2         # [0.3*0.622 + 0.7*0.750 + 0.1]
y_hat = sigmoid(z2)        # = [0.812] → sigmoid → [0.693]
print(f"z2 = {z2}")        # [0.812]
print(f"ŷ = {y_hat}")      # [0.693]
```

```
  Hesaplama Akışı:

  x=[1.0, 0.5] ──▶ z₁=Wx+b ──▶ h₁=σ(z₁) ──▶ z₂=Wh₁+b ──▶ ŷ=σ(z₂)
                    [0.5, 1.1]   [.622,.750]    [0.812]      [0.693]
```

> 💡 **İpucu:** Evrensel yaklaşım teoremi (Universal Approximation Theorem): Yeterince geniş bir gizli katmana sahip MLP, herhangi bir sürekli fonksiyonu istenen doğrulukta yakınsayabilir.

> **Öğrendiklerimiz — Perceptron & MLP**
>
> ✅ Perceptron: tek nöron, sadece lineer sınırlar çizebilir
> ✅ XOR problemi: çok katman ihtiyacının tarihsel kanıtı
> ✅ MLP: birden fazla katman → karmaşık karar sınırları
> ✅ İleri hesaplama: $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$

---

\newpage

# Aktivasyon Fonksiyonları

## Neden Doğrusal Olmama (Non-linearity) Gerekli?

Aktivasyon fonksiyonu olmadan, ne kadar katman eklersen ekle, sonuç **tek bir matris çarpımına** indirgenir:

$$\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} = \mathbf{W}_{etkili} \mathbf{x}$$

Aktivasyon fonksiyonu, her katman arasına **doğrusal olmayan** bir dönüşüm koyarak ağa gerçek "derinlik" kazandırır.

## Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
  1.0 │                    ────────
      │                 ╱
      │               ╱
  0.5 │- - - - - - -●- - - - - - -
      │           ╱
      │         ╱
  0.0 │────────
      └────────────┼──────────────
                   0
  Aralık: (0, 1)
```

| Özellik | Değer |
|:---|:---|
| Çıktı aralığı | $(0, 1)$ |
| Avantaj | Olasılık yorumu |
| Dezavantaj | **Kaybolan gradyan** (vanishing gradient) — uç değerlerde türev ≈ 0 |
| Türev | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ |

## Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

```
  1.0 │                    ────────
      │                 ╱
      │               ╱
  0.0 │─ ─ ─ ─ ─ ─ ●─ ─ ─ ─ ─ ─
      │           ╱
      │         ╱
 -1.0 │────────
      └────────────┼──────────────
                   0
  Aralık: (-1, 1)    Sıfır merkezli ✓
```

## ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$$

```
      │        ╱
      │       ╱
      │      ╱
      │     ╱
      │    ╱
  ────●───╱──────
      │  0
      │
  Aralık: [0, ∞)
```

| Özellik | Değer |
|:---|:---|
| Avantaj | Hesaplaması çok hızlı, gradyan kaybolmaz (pozitif bölge) |
| Dezavantaj | **Ölen ReLU** (dying ReLU) — negatif bölgede gradyan = 0 |
| Türev | $x > 0$ ise $1$, $x \leq 0$ ise $0$ |

## GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

burada $\Phi(x)$ standart normal dağılımın kümülatif dağılım fonksiyonudur.

```
      │        ╱
      │       ╱
      │      ╱
      │     ╱
      │   ╱
  ──── ╲╱─────────
     -0.17
  Aralık: [≈-0.17, ∞)   Düzgün (smooth) ReLU ✓
```

> 📌 **Makale Bağlantısı:** CNN'lerde genellikle **ReLU**, Transformer/ViT'lerde genellikle **GELU** kullanılır. Bu fark, Parça 3'te detaylı incelenecek. Parça 4'te, bu aktivasyon fonksiyonlarının adversarial gradyanlar üzerindeki etkisini göreceğiz.

## Karşılaştırma Tablosu

| Fonksiyon | Formül | Aralık | Gradyan Sorunu | Yaygın Kullanım |
|:---|:---|:---:|:---|:---|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0,1)$ | Vanishing gradient | Çıktı katmanı (ikili) |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1,1)$ | Vanishing gradient | RNN'ler |
| ReLU | $\max(0,x)$ | $[0,\infty)$ | Dying ReLU | CNN'ler |
| GELU | $x\Phi(x)$ | $[\approx-0.17,\infty)$ | Minimal | Transformer/ViT |

> **Öğrendiklerimiz — Aktivasyon Fonksiyonları**
>
> ✅ Doğrusal olmama olmadan derin ağ anlamsız (her şey tek katmana indirgenir)
> ✅ ReLU: CNN'lerin standart seçimi (hızlı, etkili)
> ✅ GELU: Transformer'ların standart seçimi (düzgün)
> ✅ Sigmoid: çıktı katmanında olasılık üretmek için

---

\newpage

# Kayıp Fonksiyonları (Loss Functions)

## Kayıp Fonksiyonu Nedir?

**Analoji:** Kayıp fonksiyonu bir **sınav notu** gibidir — modelin tahmininin gerçek cevaptan ne kadar uzak olduğunu ölçer. Düşük kayıp = iyi öğrenme, yüksek kayıp = kötü öğrenme.

$$L(\theta) = \text{Modelin ne kadar yanlış yaptığının ölçüsü}$$

**Eğitimin amacı:** $\min_\theta L(\theta)$ — Parça 1'deki gradyan inişi tam olarak bunu yapar.

## Mean Squared Error (MSE) — Regresyon İçin

$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
import numpy as np

y_gercek = np.array([3.0, 5.0, 2.5, 7.0])
y_tahmin = np.array([2.8, 5.2, 2.0, 6.5])

mse = np.mean((y_gercek - y_tahmin) ** 2)
print(f"MSE = {mse:.4f}")  # 0.1325
```

## Cross-Entropy Loss — Sınıflandırma İçin

Bu, hedef makalelerdeki modellerin kullandığı kayıp fonksiyonudur.

### İkili Cross-Entropy (Binary)

$$L_{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

### Kategorik Cross-Entropy (Multi-class)

$$L_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

Pratikte (one-hot encoding ile) bu sadeleşir:

$$L_{CE} = -\log(\hat{y}_{doğru\_sınıf})$$

**Sezgisel açıklama:**

```
  Doğru sınıf: Kedi

  İyi tahmin:              Kötü tahmin:
  P(kedi) = 0.9            P(kedi) = 0.1
  L = -log(0.9) = 0.105    L = -log(0.1) = 2.303

  ──●─────────────────▶    ──────────────────●▶
  0.0                2.5    0.0              2.5
  (düşük kayıp ✓)          (yüksek kayıp ✗)
```

```python
import numpy as np

def cross_entropy(y_true, y_pred):
    """Kategorik cross-entropy kaybı."""
    epsilon = 1e-12  # log(0) önleme
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# Doğru sınıf: kedi (index 0)
y_true = np.array([1, 0, 0])  # one-hot: kedi

# İyi tahmin
y_pred_iyi = np.array([0.9, 0.05, 0.05])
print(f"İyi tahmin kaybı:  {cross_entropy(y_true, y_pred_iyi):.4f}")  # 0.1054

# Kötü tahmin
y_pred_kotu = np.array([0.1, 0.6, 0.3])
print(f"Kötü tahmin kaybı: {cross_entropy(y_true, y_pred_kotu):.4f}")  # 2.3026
```

> 📌 **Makale Bağlantısı — KRİTİK İÇGÖRÜ:**
>
> Eğitim sırasında kayıp **minimize** edilir (model öğrenir):
> $$\min_\theta L(\theta, x, y)$$
>
> Adversarial saldırıda kayıp **maksimize** edilir (model kandırılır):
> $$\max_\delta L(\theta, x + \delta, y)$$
>
> **Aynı fonksiyon, zıt amaçlar!** Bu, Parça 4'ün temel kavramıdır.

## Kayıp Yüzeyi (Loss Landscape)

Kayıp fonksiyonu, parametre uzayında bir "yüzey" oluşturur:

```
  Kayıp
  (L)
   │
   │  ╲     ╱╲        ╱
   │   ╲   ╱  ╲      ╱
   │    ╲ ╱    ╲    ╱    ← Yerel minimum
   │     ●      ╲  ╱
   │              ╲╱
   │               ●  ← Global minimum
   │
   └───────────────────── θ (parametreler)

   Gradyan inişi: ● noktasından aşağı iner
```

> **Öğrendiklerimiz — Kayıp Fonksiyonları**
>
> ✅ Kayıp = modelin hata ölçüsü (düşük = iyi)
> ✅ MSE: regresyon için, CE: sınıflandırma için
> ✅ Cross-entropy: $L = -\log(\hat{y}_{doğru})$
> ✅ Eğitim: kaybı minimize et; Saldırı: kaybı maksimize et

---

\newpage

# Geri Yayılım (Backpropagation)

## Neden Backpropagation?

Milyonlarca parametreli bir sinir ağını eğitmek için her parametrenin kaybı nasıl etkilediğini bilmemiz gerekir: $\frac{\partial L}{\partial w_i}$ — yani her ağırlığa göre kaybın gradyanı.

**Backpropagation**, zincir kuralını (Parça 1) sistematik olarak uygulayarak bu gradyanları **verimli biçimde** hesaplayan algoritmadır.

## Analoji: Fabrikada Hata Takibi

```
  Hammadde ──▶ [İstasyon 1] ──▶ [İstasyon 2] ──▶ [İstasyon 3] ──▶ Ürün
                                                                     │
     ◀── Hata ◀── [İstasyon 1] ◀── [İstasyon 2] ◀── [İstasyon 3] ◀─┘
  "Sen ne                                                       "Ürün
  kadar katkı                                                  hatalı!"
  yaptın?"

  İleri yayılım (Forward): Hammadde → Ürün
  Geri yayılım (Backward): Hata sinyali ← Ürün'den geriye doğru
```

## Hesaplama Grafı (Computational Graph)

Her sinir ağı bir hesaplama grafı olarak temsil edilebilir:

**Örnek:** $L = (wx + b - y)^2$

```
      x ────▶ [×] ────▶ [+] ────▶ [-] ────▶ [²] ────▶ L
              ↑          ↑          ↑
      w ──────┘    b ────┘    y ────┘

  İleri yayılım: soldan sağa (değerleri hesapla)
  Geri yayılım: sağdan sola (gradyanları hesapla)
```

## Adım Adım Sayısal Örnek

Basit bir 1 katmanlı ağ: $\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$

**Başlangıç değerleri:**

| Değişken | Değer |
|:---:|:---:|
| $x_1$ | 1.0 |
| $x_2$ | 0.5 |
| $w_1$ | 0.3 |
| $w_2$ | 0.7 |
| $b$ | 0.1 |
| $y$ (gerçek) | 1.0 |

### Adım 1: İleri Yayılım (Forward Pass)

```
z = w₁x₁ + w₂x₂ + b
  = 0.3×1.0 + 0.7×0.5 + 0.1
  = 0.3 + 0.35 + 0.1 = 0.75

ŷ = σ(z) = σ(0.75) = 1/(1+e^(-0.75)) = 0.679

L = -(y·log(ŷ) + (1-y)·log(1-ŷ))
  = -(1.0·log(0.679) + 0·log(0.321))
  = -log(0.679) = 0.387
```

### Adım 2: Geri Yayılım (Backward Pass)

Zincir kuralını uygulayarak her parametrenin gradyanını hesaplıyoruz:

```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ) = -1/0.679 = -1.473

∂ŷ/∂z = σ(z)(1-σ(z)) = 0.679 × 0.321 = 0.218

∂L/∂z = ∂L/∂ŷ × ∂ŷ/∂z = -1.473 × 0.218 = -0.321
         (veya kısaca: ŷ - y = 0.679 - 1.0 = -0.321)

∂z/∂w₁ = x₁ = 1.0
∂z/∂w₂ = x₂ = 0.5
∂z/∂b  = 1.0

Sonuç gradyanları (zincir kuralı):
∂L/∂w₁ = ∂L/∂z × ∂z/∂w₁ = -0.321 × 1.0 = -0.321
∂L/∂w₂ = ∂L/∂z × ∂z/∂w₂ = -0.321 × 0.5 = -0.161
∂L/∂b  = ∂L/∂z × ∂z/∂b  = -0.321 × 1.0 = -0.321
```

### Adım 3: Parametre Güncelleme ($\alpha = 0.1$)

```
w₁_yeni = w₁ - α × ∂L/∂w₁ = 0.3 - 0.1×(-0.321) = 0.332
w₂_yeni = w₂ - α × ∂L/∂w₂ = 0.7 - 0.1×(-0.161) = 0.716
b_yeni  = b  - α × ∂L/∂b  = 0.1 - 0.1×(-0.321) = 0.132
```

**Gradyanlar negatif** → ağırlıklar **artırılıyor** → $\hat{y}$ büyüyecek → kayıp azalacak ✓

```
  İleri yayılım:

  x₁=1.0 ──w₁=0.3──▶╲
                       ╲
                        [Σ+b=0.1] ──▶ z=0.75 ──▶ [σ] ──▶ ŷ=0.679 ──▶ L=0.387
                       ╱
  x₂=0.5 ──w₂=0.7──▶╱

  Geri yayılım:
                                                   ∂L/∂ŷ    ∂ŷ/∂z
  ∂L/∂w₁=-0.321 ◀─── x₁ ◀──╲                    -1.473  × 0.218
                               ◀── ∂L/∂z=-0.321 ◀─────────────────── L
  ∂L/∂w₂=-0.161 ◀─── x₂ ◀──╱
```

```python
# Tam backpropagation kodu
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Başlangıç
x = np.array([1.0, 0.5])
y = 1.0
w = np.array([0.3, 0.7])
b = 0.1
alpha = 0.1

print("Eğitim başlangıcı:")
for adim in range(10):
    # İleri
    z = np.dot(w, x) + b
    y_hat = sigmoid(z)
    loss = -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    # Geri
    dL_dz = y_hat - y  # Cross-entropy + sigmoid birleşik gradyan
    dL_dw = dL_dz * x
    dL_db = dL_dz

    # Güncelle
    w = w - alpha * dL_dw
    b = b - alpha * dL_db

    if adim % 2 == 0:
        print(f"  Adım {adim}: ŷ={y_hat:.4f}, L={loss:.4f}, w={w}")

# Çıktı:
# Adım 0: ŷ=0.6792, L=0.3868, w=[0.3321 0.7161]
# Adım 2: ŷ=0.7302, L=0.3145, w=[0.3587 0.7293]
# Adım 4: ŷ=0.7712, L=0.2598, w=[0.3806 0.7403]
# Adım 6: ŷ=0.8038, L=0.2182, w=[0.3987 0.7494]
# Adım 8: ŷ=0.8293, L=0.1870, w=[0.4136 0.7568]
```

> 📌 **Makale Bağlantısı:** Adversarial gradyanlar tam olarak bu backpropagation mekanizmasıyla hesaplanır. Tek fark:
>
> | | Eğitim | Adversarial Saldırı |
> |:---|:---|:---|
> | Hesaplanan | $\nabla_\theta L$ (parametrelere göre) | $\nabla_x L$ (girdiye göre) |
> | Amaç | Kaybı azalt | Kaybı artır |
> | Güncellenen | Ağırlıklar ($\theta$) | Girdi ($x$) |

> **Öğrendiklerimiz — Backpropagation**
>
> ✅ Backpropagation = zincir kuralının sinir ağına sistematik uygulanması
> ✅ İleri yayılım: girdi → çıktı → kayıp
> ✅ Geri yayılım: kayıp → gradyanlar → tüm parametrelere
> ✅ Aynı mekanizma eğitimde ve adversarial saldırıda kullanılır (Parça 4)

---

# Parça 2B'ye Köprü

Bu bölümde sinir ağlarının temel bileşenlerini öğrendik: perceptron → MLP → aktivasyon → kayıp → backpropagation. Parça 2B'de:

1. **Optimizasyon algoritmaları** (SGD, Momentum, Adam) ile eğitimi hızlandıracağız
2. **Overfitting & regularization** ile genelleme sorunlarını çözeceğiz
3. **PyTorch** kütüphanesine geçerek gerçek model eğiteceğiz
4. **MNIST** üzerinde ilk sınıflandırıcımızı eğiteceğiz

---

*Devam: Parça 2B — Optimizasyon, PyTorch & Kod Atölyesi →*
