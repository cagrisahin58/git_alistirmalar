---
title: "Parça 1B: Olasılık, Python, NumPy & Kod Atölyesi"
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
  - \fancyhead[L]{Parça 1B — Olasılık, Python, NumPy}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Olasılık ve İstatistik Temelleri

## Neden Olasılık?

Bir sınıflandırma modeli, bir görüntüyü gördüğünde "Bu kesinlikle kedi" demez. Bunun yerine şöyle der:

```
┌──────────────────────────────────────┐
│  Model Çıktısı (Olasılık Dağılımı)  │
├──────────────┬───────────────────────┤
│  Kedi        │  ████████████  0.85   │
│  Köpek       │  ██           0.10   │
│  Kuş         │  █            0.04   │
│  Balık       │                0.01   │
└──────────────┴───────────────────────┘
```

Model bir **olasılık dağılımı** üretir. Bu dağılımı anlamak, adversarial saldırıların "ne yaptığını" anlamamızın anahtarıdır.

> 📌 **Makale Bağlantısı:** Adversarial saldırı bu dağılımı bozar — modeli yüksek güvenle yanlış sınıfa yönlendirir. Makale A'da "model confidansı" bu dağılımdan gelir.

## Temel Olasılık Kavramları

### Olasılık Uzayı

| Kavram | Tanım | Örnek |
|:---|:---|:---|
| **Deney** | Sonucu belirsiz olan eylem | Zar atma |
| **Örneklem uzayı** ($\Omega$) | Tüm olası sonuçlar | $\{1,2,3,4,5,6\}$ |
| **Olay** ($A$) | Örneklem uzayının alt kümesi | "Çift gelme" = $\{2,4,6\}$ |
| **Olasılık** ($P(A)$) | Olayın gerçekleşme ölçüsü | $P(\text{çift}) = 3/6 = 0.5$ |

**Olasılık aksiyomları:**

1. $0 \leq P(A) \leq 1$ (her olay 0 ile 1 arasında)
2. $P(\Omega) = 1$ (bir şey mutlaka olur)
3. $P(A \cup B) = P(A) + P(B)$ eğer $A \cap B = \emptyset$ (ayrık olaylar toplanır)

### Koşullu Olasılık

"B olduğunu biliyorsam, A'nın olasılığı nedir?"

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Örnek:** Bir torba: 3 kırmızı + 2 mavi top. İlk çekilen kırmızı ise (yerine koymadan), ikincinin de kırmızı olma olasılığı:

$$P(K_2 | K_1) = \frac{2}{4} = 0.5$$

### Bayes Teoremi

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Analoji:** Hasta olduğunu (A) test sonucundan (B) çıkarsama. Test pozitif geldi — gerçekten hasta mısın?

| Terim | Anlam | Makine öğrenmesi karşılığı |
|:---|:---|:---|
| $P(A)$ | Ön olasılık (prior) | Sınıfın genel olasılığı |
| $P(B\|A)$ | Olabilirlik (likelihood) | Veri verildiğinde modelin tahmini |
| $P(A\|B)$ | Son olasılık (posterior) | Veri gördükten sonra sınıf olasılığı |

> **Öğrendiklerimiz — Temel Olasılık**
>
> ✅ Olasılık 0-1 arası bir ölçüdür
> ✅ Koşullu olasılık: "bunu biliyorsam, şunun olasılığı ne?"
> ✅ Bayes teoremi: gözlemlerden çıkarım yapmanın temeli

---

## Olasılık Dağılımları

### Kesikli Dağılımlar

**Bernoulli Dağılımı:** Yazı-tura gibi iki sonuçlu deney.

$$P(X=1) = p, \quad P(X=0) = 1-p$$

**Kategorik Dağılım:** Birden fazla sınıf (zar atma, sınıflandırma).

### Sürekli Dağılımlar

**Uniform Dağılım:** Her değer eşit olasılıklı.

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

**Normal (Gauss) Dağılım:** Doğadaki en yaygın dağılım — "çan eğrisi":

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

```
                    μ = ortalama
                    ↓
            ┌───────●───────┐
           ╱│       │       │╲
          ╱ │       │       │ ╲
         ╱  │  %68  │       │  ╲
        ╱   │◄──────┤───────▶│  ╲
       ╱    │   σ   │   σ   │   ╲
      ╱     │       │       │    ╲
  ───╱──────┼───────┼───────┼─────╲────
   μ-3σ   μ-σ      μ     μ+σ    μ+3σ

   ├──── %68.3 ────┤ (±1σ)
   ├─────── %95.4 ─────────┤ (±2σ)
   ├────────── %99.7 ──────────────┤ (±3σ)
```

| Parametre | Sembol | Anlam |
|:---|:---:|:---|
| Ortalama | $\mu$ | Dağılımın merkezi |
| Standart sapma | $\sigma$ | Dağılımın yayılımı (genişliği) |
| Varyans | $\sigma^2$ | Standart sapmanın karesi |

> 📌 **Makale Bağlantısı:** PGD saldırısında (Parça 4) başlangıç pertürbasyonu genellikle uniform dağılımdan örneklenir: $\delta_0 \sim U(-\epsilon, \epsilon)$

## Beklenen Değer, Varyans, Standart Sapma

$$E[X] = \sum_i x_i \cdot P(x_i) \quad \text{(kesikli)}$$

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

$$\text{Std}(X) = \sigma = \sqrt{\text{Var}(X)}$$

**Analoji:** Beklenen değer = ortalama sonuç. Varyans = sonuçların ortalamadan ne kadar sapabileceği.

---

## Softmax Fonksiyonu — Sayılardan Olasılığa

Bir sinir ağının son katmanı ham sayılar (logit'ler) üretir. Bunları olasılık dağılımına dönüştüren fonksiyon **softmax**tir:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Örnek:**

```
  Logitler (ham çıktı)     Softmax             Olasılıklar
  ┌──────┐                                     ┌──────────┐
  │ 2.0  │──────▶  e^2.0 / (e^2 + e^1 + e^0.1)  = 0.659  │ Kedi  ✓
  │ 1.0  │──────▶  e^1.0 / (e^2 + e^1 + e^0.1)  = 0.242  │ Köpek
  │ 0.1  │──────▶  e^0.1 / (e^2 + e^1 + e^0.1)  = 0.099  │ Kuş
  └──────┘                                     └──────────┘
                                                Toplam = 1.0 ✓
```

**Softmax'ın özellikleri:**

- Tüm çıktılar 0 ile 1 arasında: $0 < \text{softmax}(z_i) < 1$
- Toplamları her zaman 1: $\sum_i \text{softmax}(z_i) = 1$
- En büyük logit'e en yüksek olasılık

> 📌 **Makale Bağlantısı:** Her iki makaledeki modellerin çıktısı softmax olasılıklarıdır. Adversarial saldırılar bu olasılıkları manipüle ederek yanlış sınıfın olasılığını artırır.

> **Öğrendiklerimiz — Olasılık & İstatistik**
>
> ✅ Normal dağılım: doğadaki rastgeleliğin temel modeli
> ✅ Beklenen değer ve varyans: dağılımın özetleri
> ✅ Softmax: ham sayıları olasılık dağılımına dönüştürür
> ✅ Sınıflandırma modellerinin çıktısı bir olasılık dağılımıdır

---

\newpage

# Python Programlamaya Giriş

## Neden Python?

Python, derin öğrenme dünyasının **resmi dili**dir. PyTorch, TensorFlow, NumPy — tüm temel araçlar Python'dadır.

## Kurulum

```
┌────────────────────────────────────────┐
│  Önerilen Kurulum Yolu                 │
├────────────────────────────────────────┤
│  1. Python 3.9+ indir: python.org      │
│  2. pip ile paket yönetimi              │
│  3. Jupyter Notebook: pip install       │
│     jupyter notebook                    │
│                                         │
│  Alternatif (daha kolay):               │
│  → Google Colab (colab.google) kullan   │
│    (kurulum gerektirmez, tarayıcıda)    │
└────────────────────────────────────────┘
```

> 💡 **İpucu:** Başlangıç için Google Colab en pratik seçenektir — kurulum gerektirmez, GPU bile sunar.

## Temel Veri Tipleri

```python
# Tam sayı (integer)
yas = 25
print(type(yas))        # <class 'int'>

# Ondalıklı sayı (float)
sicaklik = 36.6
print(type(sicaklik))   # <class 'float'>

# Metin (string)
isim = "Adversarial"
print(type(isim))       # <class 'str'>

# Mantıksal (boolean)
dogru_mu = True
print(type(dogru_mu))   # <class 'bool'>
```

## Temel Aritmetik İşlemler

```python
a, b = 7, 3

print(a + b)    # 10    Toplama
print(a - b)    # 4     Çıkarma
print(a * b)    # 21    Çarpma
print(a / b)    # 2.333 Bölme (float)
print(a // b)   # 2     Tam bölme (floor)
print(a % b)    # 1     Mod (kalan)
print(a ** b)   # 343   Üs alma (7³)
```

## Kontrol Yapıları

### if / elif / else

```python
skor = 85

if skor >= 90:
    print("A - Mükemmel")
elif skor >= 80:
    print("B - İyi")       # ← Bu çalışır
elif skor >= 70:
    print("C - Orta")
else:
    print("F - Yetersiz")

# Çıktı: B - İyi
```

### for Döngüsü

```python
# Liste üzerinde döngü
renkler = ["kırmızı", "yeşil", "mavi"]
for renk in renkler:
    print(f"Kanal: {renk}")

# Çıktı:
# Kanal: kırmızı
# Kanal: yeşil
# Kanal: mavi

# range ile sayısal döngü
for i in range(5):
    print(i, end=" ")
# Çıktı: 0 1 2 3 4
```

### while Döngüsü

```python
# Gradyan inişi simülasyonu (sezgisel)
x = 10.0
ogrenme_hizi = 0.1
for adim in range(20):
    gradyan = 2 * x  # f(x) = x² fonksiyonunun türevi
    x = x - ogrenme_hizi * gradyan
    if adim % 5 == 0:
        print(f"Adım {adim}: x = {x:.4f}, f(x) = {x**2:.4f}")

# Çıktı:
# Adım 0: x = 8.0000, f(x) = 64.0000
# Adım 5: x = 2.6214, f(x) = 6.8720
# Adım 10: x = 0.8590, f(x) = 0.7378
# Adım 15: x = 0.2815, f(x) = 0.0792
```

## Fonksiyonlar

```python
def sigmoid(x):
    """Sigmoid aktivasyon fonksiyonu."""
    import math
    return 1 / (1 + math.exp(-x))

# Test
print(sigmoid(0))     # 0.5
print(sigmoid(2))     # 0.8808
print(sigmoid(-2))    # 0.1192
```

```python
def oklid_normu(vektor):
    """Bir vektörün L2 (Öklid) normunu hesaplar."""
    toplam = sum(x**2 for x in vektor)
    return toplam ** 0.5

# Test
v = [3, 4]
print(oklid_normu(v))  # 5.0 (3-4-5 üçgeni!)
```

## Listeler ve Sözlükler

```python
# Liste (list) — sıralı, değiştirilebilir
piksel_degerleri = [128, 64, 255, 0, 192]
print(piksel_degerleri[0])     # 128 (ilk eleman)
print(piksel_degerleri[-1])    # 192 (son eleman)
print(piksel_degerleri[1:3])   # [64, 255] (dilimleme)

# List comprehension — kısa ve güçlü
kareler = [x**2 for x in range(10)]
print(kareler)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Sözlük (dictionary) — anahtar-değer çiftleri
model_sonuclari = {
    "clean_accuracy": 0.95,
    "robust_accuracy": 0.43,
    "attack_method": "PGD",
    "epsilon": 8/255
}
print(model_sonuclari["clean_accuracy"])  # 0.95
```

> **Öğrendiklerimiz — Python Temelleri**
>
> ✅ Temel veri tipleri: int, float, str, bool
> ✅ Kontrol yapıları: if/elif/else, for, while
> ✅ Fonksiyonlar: def ile tanımlama
> ✅ Liste ve sözlük: verileri organize etme
> ✅ List comprehension: kısa ve etkili liste oluşturma

---

\newpage

# NumPy — Sayısal Hesaplama Kütüphanesi

## NumPy Nedir?

NumPy (Numerical Python), Python'da vektör, matris ve tensör işlemlerini **hızlı** yapan kütüphanedir. Derin öğrenme kütüphaneleri (PyTorch, TensorFlow) NumPy'ın tasarım kalıplarını kullanır.

```python
import numpy as np
```

## Dizi (Array) Oluşturma

```python
# Vektör (1D array)
v = np.array([1, 2, 3, 4, 5])
print(v)          # [1 2 3 4 5]
print(v.shape)    # (5,)
print(v.ndim)     # 1

# Matris (2D array)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)    # (2, 3)
print(M.ndim)     # 2

# Tensör (3D array) — Renkli görüntü benzeri
T = np.random.randint(0, 256, size=(3, 4, 4))  # 3 kanal, 4×4
print(T.shape)    # (3, 4, 4)
print(T.ndim)     # 3
```

### Özel Diziler

```python
# Sıfır matrisi
sifirlar = np.zeros((3, 3))
print(sifirlar)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

# Bir matrisi
birler = np.ones((2, 4))

# Birim matris (identity)
I = np.eye(3)
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Aralıklı dizi
x = np.linspace(0, 2*np.pi, 100)  # 0'dan 2π'ye 100 eşit nokta

# Rastgele dizi (normal dağılım)
gurultu = np.random.randn(28, 28)  # 28×28 Gauss gürültüsü
```

## Dilimleme (Indexing & Slicing)

```python
A = np.array([[10, 20, 30, 40],
              [50, 60, 70, 80],
              [90, 100, 110, 120]])

print(A[0, 2])      # 30      (0.satır, 2.sütun)
print(A[1, :])       # [50 60 70 80]  (1. satırın tamamı)
print(A[:, 0])       # [10 50 90]     (0. sütunun tamamı)
print(A[0:2, 1:3])   # [[20 30]       (alt matris)
                      #  [60 70]]

# Koşullu indeksleme
print(A[A > 50])     # [ 60  70  80  90 100 110 120]
```

```
  A matrisi:
  ┌────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ 40 │  ← A[0, :]
  ├────┼────┼────┼────┤
  │ 50 │ 60 │ 70 │ 80 │  ← A[1, :]
  ├────┼────┼────┼────┤
  │ 90 │100 │110 │120 │  ← A[2, :]
  └────┴────┴────┴────┘
    ↑              ↑
  A[:, 0]       A[:, 3]

  A[0:2, 1:3]:
  ┌────┬────┐
  │ 20 │ 30 │
  ├────┼────┤
  │ 60 │ 70 │
  └────┴────┘
```

## Temel İşlemler

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Eleman bazlı işlemler
print(a + b)        # [5. 7. 9.]
print(a * b)        # [ 4. 10. 18.]  (Hadamard çarpım)
print(a ** 2)       # [1. 4. 9.]

# İç çarpım (dot product)
print(np.dot(a, b))          # 32.0
print(a @ b)                 # 32.0 (@ operatörü)

# Norm
print(np.linalg.norm(a))    # 3.7417 (L2 normu)
print(np.linalg.norm(a, 1)) # 6.0    (L1 normu)
print(np.linalg.norm(a, np.inf))  # 3.0 (L∞ normu)
```

## Matris İşlemleri

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matris çarpımı
print(A @ B)
# [[19 22]
#  [43 50]]

# Transpose
print(A.T)
# [[1 3]
#  [2 4]]

# Determinant
print(np.linalg.det(A))     # -2.0

# Ters matris
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Özdeğer ve özvektörler
ozdegerler, ozvektorler = np.linalg.eig(A)
print(f"Özdeğerler: {ozdegerler}")    # [-0.3723, 5.3723]
print(f"Özvektörler:\n{ozvektorler}")
```

## Broadcasting — Boyut Uyumsuzluğunu Çözme

NumPy, farklı boyutlu dizileri otomatik olarak uyumlu hale getirir:

```python
# Skaler + Matris
A = np.array([[1, 2], [3, 4]])
print(A + 10)
# [[11 12]
#  [13 14]]

# Vektör + Matris (her satıra vektör eklenir)
v = np.array([100, 200])
print(A + v)
# [[101 202]
#  [103 204]]
```

```
  Broadcasting kuralı:

  A (2×3):  ┌─┬─┬─┐    v (1×3):  ┌─┬─┬─┐
            │ │ │ │              │ │ │ │
            ├─┼─┼─┤    ───▶     ├─┼─┼─┤  (otomatik kopyalanır)
            │ │ │ │              │ │ │ │
            └─┴─┴─┘              └─┴─┴─┘

  Sonuç (2×3): eleman bazlı toplama
```

> 📌 **Makale Bağlantısı:** Adversarial pertürbasyon eklerken broadcasting kullanılır:
> ```python
> x_adv = x + epsilon * sign_gradient  # x: (B,3,224,224), epsilon: skaler
> ```

> **Öğrendiklerimiz — NumPy**
>
> ✅ np.array: vektör, matris, tensör oluşturma
> ✅ Dilimleme: alt bölge seçme (görüntü kırpma gibi)
> ✅ Eleman bazlı işlemler ve matris çarpımı (@)
> ✅ np.linalg: norm, determinant, ters matris
> ✅ Broadcasting: boyut uyumsuzluğunu otomatik çözme

---

\newpage

# Matplotlib — Veri Görselleştirme

## Temel Grafikler

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Çizgi Grafik: Aktivasyon Fonksiyonları ---
x = np.linspace(-5, 5, 200)

sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, sigmoid, 'b-', linewidth=2)
axes[0].set_title('Sigmoid')
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, relu, 'r-', linewidth=2)
axes[1].set_title('ReLU')
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x, tanh, 'g-', linewidth=2)
axes[2].set_title('Tanh')
axes[2].axhline(y=0, color='k', linewidth=0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aktivasyon_fonksiyonlari.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Beklenen çıktı:**

```
  Sigmoid                ReLU                  Tanh
  1.0 │    ──────       5│       ╱          1.0│      ──────
      │   ╱              │      ╱              │    ╱
  0.5 │──╱──             │     ╱           0.0 │───╱───
      │ ╱                │    ╱                │  ╱
  0.0 │╱                 │___╱             -1.0│╱
      └──────────        └──────────           └──────────
      -5    0    5       -5   0    5           -5    0    5
```

## Görüntü Gösterimi (imshow)

```python
# Rastgele gri tonlamalı görüntü oluştur
goruntu = np.random.randint(0, 256, size=(28, 28)).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Orijinal
axes[0].imshow(goruntu, cmap='gray')
axes[0].set_title('Orijinal')
axes[0].axis('off')

# Gürültü ekle
gurultu = np.random.randn(28, 28) * 50  # σ=50 Gauss gürültüsü
gurultulu = np.clip(goruntu + gurultu, 0, 255).astype(np.uint8)
axes[1].imshow(gurultulu, cmap='gray')
axes[1].set_title('Gürültülü (σ=50)')
axes[1].axis('off')

# Fark (pertürbasyon)
fark = np.abs(goruntu.astype(float) - gurultulu.astype(float))
axes[2].imshow(fark, cmap='hot')
axes[2].set_title('Fark (Pertürbasyon)')
axes[2].axis('off')

plt.suptitle('Görüntü + Gürültü = Adversarial\'ın Basit Versiyonu', fontsize=14)
plt.tight_layout()
plt.savefig('gurultu_ornegi.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Histogram

```python
# Piksel dağılımlarını karşılaştır
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(goruntu.flatten(), bins=50, color='blue', alpha=0.7)
axes[0].set_title('Orijinal Piksel Dağılımı')
axes[0].set_xlabel('Piksel Değeri')
axes[0].set_ylabel('Frekans')

axes[1].hist(gurultulu.flatten(), bins=50, color='red', alpha=0.7)
axes[1].set_title('Gürültülü Piksel Dağılımı')
axes[1].set_xlabel('Piksel Değeri')

plt.tight_layout()
plt.savefig('piksel_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **Öğrendiklerimiz — Matplotlib**
>
> ✅ plt.plot: çizgi grafikleri (fonksiyon görselleştirme)
> ✅ plt.imshow: görüntü gösterimi
> ✅ plt.hist: dağılım analizi
> ✅ Subplot: birden fazla grafiği yan yana

---

\newpage

# Kod Atölyesi — Her Şeyi Birleştiriyoruz

## Atölye 1: NumPy ile Lineer Cebir

```python
import numpy as np

# ─── Vektör İşlemleri ───
print("=" * 50)
print("VEKTÖR İŞLEMLERİ")
print("=" * 50)

# İki vektör tanımla
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Toplama (adversarial eklemenin temeli)
print(f"a + b = {a + b}")                    # [5. 7. 9.]

# İç çarpım (dikkat mekanizmasının temeli)
print(f"a · b = {np.dot(a, b)}")             # 32.0

# Normlar (pertürbasyon ölçümünün temeli)
print(f"||a||₁ = {np.linalg.norm(a, 1)}")    # 6.0
print(f"||a||₂ = {np.linalg.norm(a, 2):.4f}")# 3.7417
print(f"||a||∞ = {np.linalg.norm(a, np.inf)}")# 3.0

# Kosinüs benzerliği
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Kosinüs benzerliği: {cos_sim:.4f}")  # 0.9746

# ─── Matris İşlemleri ───
print("\n" + "=" * 50)
print("MATRİS İŞLEMLERİ")
print("=" * 50)

W = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # 3×2 ağırlık matrisi

x = np.array([0.5, 0.3])  # 2 boyutlu girdi
b_bias = np.array([0.1, 0.2, 0.3])  # 3 boyutlu bias

# Sinir ağı katmanı: y = Wx + b
y = W @ x + b_bias
print(f"W @ x + b = {y}")                   # [1.2 2.7 4.3]

print(f"\nW boyutu: {W.shape}")              # (3, 2)
print(f"x boyutu: {x.shape}")               # (2,)
print(f"Çıktı boyutu: {y.shape}")           # (3,)
```

---

## Atölye 2: Gradyan İnişi Simülasyonu

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Minimize edilecek fonksiyon: f(x) = x² - 4x + 4 = (x-2)²"""
    return x**2 - 4*x + 4

def f_turev(x):
    """Fonksiyonun türevi: f'(x) = 2x - 4"""
    return 2*x - 4

# ─── Gradyan İnişi ───
x_baslangic = -1.0
ogrenme_hizi = 0.1
adim_sayisi = 30

# Geçmişi kaydet
x_gecmisi = [x_baslangic]
f_gecmisi = [f(x_baslangic)]

x_simdiki = x_baslangic
for i in range(adim_sayisi):
    gradyan = f_turev(x_simdiki)
    x_simdiki = x_simdiki - ogrenme_hizi * gradyan
    x_gecmisi.append(x_simdiki)
    f_gecmisi.append(f(x_simdiki))

# ─── Görselleştirme ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Fonksiyon üzerinde adımlar
x_cizim = np.linspace(-2, 5, 200)
axes[0].plot(x_cizim, f(x_cizim), 'b-', linewidth=2, label='f(x) = (x-2)²')
axes[0].plot(x_gecmisi, f_gecmisi, 'ro-', markersize=5, alpha=0.6,
             label=f'Gradyan İnişi (α={ogrenme_hizi})')
axes[0].plot(x_gecmisi[0], f_gecmisi[0], 'g*', markersize=15,
             label='Başlangıç')
axes[0].plot(x_gecmisi[-1], f_gecmisi[-1], 'r*', markersize=15,
             label=f'Son (x={x_gecmisi[-1]:.3f})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Gradyan İnişi — Fonksiyon Üzerinde')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sağ: Kayıp eğrisi
axes[1].plot(range(adim_sayisi + 1), f_gecmisi, 'r-o', markersize=3)
axes[1].set_xlabel('Adım')
axes[1].set_ylabel('f(x) değeri')
axes[1].set_title('Kayıp (Loss) Eğrisi')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')  # Logaritmik ölçek

plt.tight_layout()
plt.savefig('gradyan_inisi_simulasyonu.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSonuç: x = {x_gecmisi[-1]:.6f} (beklenen: 2.0)")
print(f"f(x) = {f_gecmisi[-1]:.6f} (beklenen: 0.0)")
```

---

## Atölye 3: Görüntüyü Matris Olarak İşleme

```python
import numpy as np
import matplotlib.pyplot as plt

# ─── Basit bir 8×8 görüntü oluştur (satranç tahtası) ───
satranc = np.zeros((8, 8))
satranc[::2, 1::2] = 1
satranc[1::2, ::2] = 1

# ─── Rastgele bir görüntü oluştur ───
np.random.seed(42)
goruntu = np.random.randint(0, 256, (32, 32)).astype(np.float64)

# ─── Adversarial pertürbasyon simülasyonu ───
epsilon_degerleri = [0, 2, 8, 32, 64]

fig, axes = plt.subplots(2, len(epsilon_degerleri), figsize=(16, 7))

for idx, eps in enumerate(epsilon_degerleri):
    # Rastgele pertürbasyon (gerçek adversarial değil, kavramsal demo)
    if eps == 0:
        delta = np.zeros_like(goruntu)
    else:
        delta = np.random.uniform(-eps, eps, goruntu.shape)

    # Pertürbasyonu uygula ve [0, 255] aralığına kırp
    goruntu_pert = np.clip(goruntu + delta, 0, 255)

    # Üst satır: Pertürbe edilmiş görüntü
    axes[0, idx].imshow(goruntu_pert, cmap='gray', vmin=0, vmax=255)
    axes[0, idx].set_title(f'ε = {eps}')
    axes[0, idx].axis('off')

    # Alt satır: Pertürbasyonun kendisi
    if eps == 0:
        axes[1, idx].imshow(np.zeros_like(goruntu), cmap='RdBu',
                           vmin=-64, vmax=64)
    else:
        axes[1, idx].imshow(delta, cmap='RdBu', vmin=-64, vmax=64)
    axes[1, idx].set_title(f'δ (ε={eps})')
    axes[1, idx].axis('off')

axes[0, 0].set_ylabel('Görüntü', fontsize=12)
axes[1, 0].set_ylabel('Pertürbasyon', fontsize=12)

plt.suptitle('Farklı ε Değerlerinde Pertürbasyon Etkisi\n'
             '(Küçük ε → gözle görülmez, büyük ε → belirgin bozulma)',
             fontsize=14)
plt.tight_layout()
plt.savefig('epsilon_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── Pertürbasyon istatistikleri ───
print("\n" + "=" * 60)
print("PERTÜRBASİYON İSTATİSTİKLERİ")
print("=" * 60)
for eps in [2, 8, 32, 64]:
    delta = np.random.uniform(-eps, eps, goruntu.shape)
    print(f"\nε = {eps}:")
    print(f"  L∞ normu: {np.linalg.norm(delta.flatten(), np.inf):.2f}")
    print(f"  L2 normu: {np.linalg.norm(delta.flatten(), 2):.2f}")
    print(f"  Ortalama |δ|: {np.mean(np.abs(delta)):.2f}")
```

---

## Atölye 4: Softmax Uygulama ve Görselleştirme

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """Softmax fonksiyonu (sayısal kararlılık ile)."""
    z_shifted = z - np.max(z)  # Overflow önleme
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# ─── Normal sınıflandırma ───
logitler_normal = np.array([2.0, 1.0, 0.1, -0.5, -1.0])
siniflar = ['Kedi', 'Köpek', 'Kuş', 'Balık', 'Kurbağa']
olasiliklar = softmax(logitler_normal)

print("Normal Sınıflandırma:")
for sinif, p in zip(siniflar, olasiliklar):
    bar = '█' * int(p * 40)
    print(f"  {sinif:10s} {p:.3f} {bar}")
print(f"  Toplam: {sum(olasiliklar):.3f}")

# ─── Adversarial sınıflandırma (logitler manipüle edilmiş) ───
logitler_adv = np.array([-1.0, 3.0, 0.5, -0.5, -1.0])  # Köpek yükseltilmiş
olasiliklar_adv = softmax(logitler_adv)

print("\nAdversarial Sınıflandırma (manipüle edilmiş):")
for sinif, p in zip(siniflar, olasiliklar_adv):
    bar = '█' * int(p * 40)
    print(f"  {sinif:10s} {p:.3f} {bar}")

# ─── Görselleştirme ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

axes[0].barh(siniflar, olasiliklar, color=colors)
axes[0].set_xlim(0, 1)
axes[0].set_title('Normal Tahmin\n(Doğru: Kedi ✓)', fontsize=13)
axes[0].set_xlabel('Olasılık')

axes[1].barh(siniflar, olasiliklar_adv, color=colors)
axes[1].set_xlim(0, 1)
axes[1].set_title('Adversarial Tahmin\n(Yanlış: Köpek ✗)', fontsize=13)
axes[1].set_xlabel('Olasılık')

plt.suptitle('Adversarial Saldırının Softmax Çıktısına Etkisi', fontsize=14)
plt.tight_layout()
plt.savefig('softmax_adversarial.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Beklenen çıktı:**

```
Normal Sınıflandırma:
  Kedi       0.549 ██████████████████████
  Köpek      0.202 ████████
  Kuş        0.082 ███
  Balık      0.045 █
  Kurbağa    0.027 █
  Toplam: 1.000

Adversarial Sınıflandırma (manipüle edilmiş):
  Kedi       0.015
  Köpek      0.818 ████████████████████████████████
  Kuş        0.067 ██
  Balık      0.025 █
  Kurbağa    0.015
```

---

\newpage

# Parça 1 — Genel Kavram Haritası

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     PARÇA 1 — TAM KAVRAM HARİTASI                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  MATEMATİK TEMELLERİ                    PROGRAMLAMA ARAÇLARI           ║
║  ═══════════════════                    ════════════════════            ║
║                                                                         ║
║  Skaler → Vektör → Matris → Tensör ◄────── NumPy (np.array)           ║
║             │         │        │                                        ║
║             ▼         ▼        ▼                                        ║
║        İç çarpım  Çarpım   Görüntü ◄────── Matplotlib (imshow)        ║
║             │      (W·x+b)  verisi                                      ║
║             │         │                                                  ║
║        Norm ◄─────────┘                    Python                       ║
║        (L₁,L₂,L∞)                        (fonksiyonlar,                ║
║             │                              döngüler,                    ║
║             ▼                              listeler)                    ║
║      Pertürbasyon                              │                        ║
║       bütçesi (ε)                              ▼                        ║
║             │                           Softmax (olasılık)              ║
║             │        Olasılık                  │                        ║
║             │     ═══════════                  │                        ║
║             │     Normal dağılım               │                        ║
║             │     Beklenen değer               │                        ║
║             │     Bayes teoremi                │                        ║
║             │           │                      │                        ║
║             ▼           ▼                      ▼                        ║
║         Türev ──▶ Zincir Kuralı ──▶ Gradyan ──▶ Gradyan İnişi         ║
║                                        │           │       │            ║
║                                        │           ▼       ▼            ║
║                                        │       Öğrenme  Saldırı        ║
║                                        │       (−∇L)    (+∇L)          ║
║                                        │       ▼            ▼           ║
║                                        │    PARÇA 2      PARÇA 4       ║
║                                        │                                ║
║                                        └──▶ PARÇA 3 (CNN, ViT)        ║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

# Sonraki Parçaya Köprü — Parça 2'ye Hazırlık

Bu parçada **derin öğrenmenin tüm temel araçlarını** öğrendik:

| Araç | Parça 2'deki Kullanımı |
|:---|:---|
| Matris çarpımı ($\mathbf{W}\mathbf{x} + \mathbf{b}$) | Sinir ağı katmanının temel işlemi |
| Gradyan ($\nabla L$) | Ağırlıkları güncellemek (öğrenmek) için |
| Zincir kuralı | Backpropagation algoritmasının temeli |
| Softmax | Modelin çıktısını olasılığa dönüştürme |
| NumPy / Matplotlib | Veri işleme ve görselleştirme |

**Parça 2'de** bu araçları kullanarak:

1. Bir **perceptron** (tek nöron) inşa edeceğiz
2. Onu **çok katmanlı ağ**a (MLP) genişleteceğiz
3. **Backpropagation** ile eğiteceğiz
4. **PyTorch** kütüphanesine geçiş yapacağız
5. İlk gerçek sınıflandırıcımızı (MNIST) eğiteceğiz

Gradyan inişi artık bir matematiksel kavram olmaktan çıkıp, gerçek bir sinir ağının **öğrenme motoru** olacak.

---

*Devam: Parça 2A — Makine Öğrenmesi & Sinir Ağlarının Temelleri →*
