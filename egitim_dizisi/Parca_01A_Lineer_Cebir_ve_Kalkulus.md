---
title: "Parça 1A: Matematiksel Temeller — Lineer Cebir & Kalkülüs"
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
  - \fancyhead[L]{Parça 1A — Lineer Cebir \& Kalkülüs}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Giriş ve Motivasyon

## Bu Serinin Amacı

Bu eğitim serisi seni sıfırdan, iki ileri düzey araştırma makalesini anlayabilecek, uygulayabilecek ve benzer çalışmalar üretebilecek bir **adversarial derin öğrenme araştırmacısı** olarak yetiştirmek için tasarlandı.

Hedef makalelerimiz:

> **Makale A:** *"A Comparative Study of Convolutional and Transformer Architectures Under Adversarial Perturbations: Gradient Characteristics and Transferability Analysis"*
>
> **Makale B:** *"FE-ViT-ECL: Adversarially Robust Medical Image Forgery Detection via Frequency-Enhanced Vision Transformers with Explanation Consistency Loss"*

Bu makaleler ilk bakışta karmaşık görünebilir — ama endişelenme. Her kavramı sıfırdan, günlük hayattan analojilerle inşa edeceğiz.

## Neden Matematik?

Yapay zeka, özünde **matematiksel bir makine**dir. Bir sinir ağı eğitmek demek:

- **Lineer cebir** ile veri ve ağırlıkları temsil etmek (matris çarpımları)
- **Kalkülüs** ile "öğrenme" yönünü bulmak (gradyan hesabı)
- **Olasılık** ile belirsizliği modellemek (sınıflandırma çıktıları)

demektir. Bu parçada bu üç ayağın ilk ikisini sağlamlaştıracağız.

## İlk Tat: Adversarial Örnek Nedir?

Aşağıdaki ünlü örneği düşün:

```
┌─────────────┐     ┌───────────┐     ┌─────────────┐
│             │     │           │     │             │
│   🐼 Panda  │  +  │  Gürültü  │  =  │  🦍 Gibbon  │
│  (%57.7)    │     │ (×0.007)  │     │  (%99.3)    │
│             │     │           │     │             │
└─────────────┘     └───────────┘     └─────────────┘
   Orijinal          Pertürbasyon       Adversarial
   görüntü           (gözle görünmez)   görüntü
```

Bir yapay zeka modeli bu pandayı %57.7 güvenle doğru tanıyor. Ama görüntüye **insan gözüyle fark edilemeyen** minik bir gürültü eklendiğinde, aynı model bunu %99.3 güvenle **gibbon** olarak sınıflandırıyor!

Bu gürültü rastgele değil — modelin **gradyanı** kullanılarak hesaplanıyor. İşte bu yüzden gradyanı ve onu hesaplamak için gereken matematiği öğrenmemiz şart.

> 📌 **Makale Bağlantısı:** Her iki hedef makale de bu tür adversarial örneklerle ilgili. Makale A, farklı mimarilerin bu saldırılara nasıl tepki verdiğini inceler. Makale B ise medikal görüntülerde bu tür sahtecilikleri tespit eden bir sistem önerir.

## 6 Parçanın Yol Haritası

```
╔══════════════════════════════════════════════════════════════════╗
║  Parça 1  ➤  Matematiksel Temeller & Programlama    ◄── Buradasın!
║  Parça 2  ➤  Makine Öğrenmesi & Sinir Ağları                   ║
║  Parça 3  ➤  Bilgisayarlı Görü & Derin Mimariler (CNN, ViT)    ║
║  Parça 4  ➤  Adversarial Makine Öğrenmesi                      ║
║  Parça 5  ➤  Frekans Analizi & Açıklanabilirlik                 ║
║  Parça 6  ➤  Entegrasyon — Hedef Makalelerin Anatomisi          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

\newpage

# Lineer Cebir Temelleri

## Skaler, Vektör, Matris, Tensör — Veri Hiyerarşisi

Derin öğrenmede veriler farklı boyutlarda yapılarda saklanır. Bunları bir hiyerarşi olarak düşünelim:

### Analoji: Kitaplık Sistemi

| Matematiksel Yapı | Analoji | Boyut | Örnek |
|:---|:---|:---:|:---|
| **Skaler** (sayıl) | Tek bir kitap | 0D | Sıcaklık: $37.2$ |
| **Vektör** (yöney) | Bir raf dolusu kitap | 1D | Piksel satırı: $[128, 64, 255]$ |
| **Matris** (dizey) | Bir dolap dolusu raf | 2D | Gri tonlamalı görüntü: $28 \times 28$ |
| **Tensör** | Oda dolusu dolap | 3D+ | Renkli görüntü: $224 \times 224 \times 3$ |

```
  Skaler         Vektör           Matris              Tensör (3D)
                                                   ┌─────────┐
               ┌─┬─┬─┐      ┌─┬─┬─┬─┐           ┌─┤─────────┤
   [5]         │3│1│4│      │2│7│1│8│          ┌─┤ │░░░░░░░░░│
               └─┴─┴─┘      │3│6│5│0│         │  │ │░░░░░░░░░│
                             │9│2│4│7│         │  │ │░░░░░░░░░│
  0 boyut     1 boyut       └─┴─┴─┴─┘         │  └─┤─────────┤
                             2 boyut            └────┘─────────┘
                                                3 boyut
```

> 📌 **Makale Bağlantısı:** Bir renkli görüntü aslında 3 boyutlu bir tensördür: yükseklik × genişlik × renk kanalı (RGB). Hedef makalelerdeki tüm modeller bu tensörler üzerinde çalışır.

### Matematiksel Gösterim

Serinin tamamında şu notasyonu kullanacağız:

- Skaler: küçük harf, normal yazı → $x$, $\alpha$, $\epsilon$
- Vektör: küçük harf, kalın → $\mathbf{x}$, $\mathbf{w}$ (veya ok: $\vec{x}$)
- Matris: büyük harf, kalın → $\mathbf{W}$, $\mathbf{A}$
- Tensör: büyük harf, özel font → $\mathcal{X}$

---

## Vektörler

### Vektör Nedir? — Sezgisel Yaklaşım

**Analoji:** Bir vektör, bir **ok** gibidir — hem büyüklüğü (uzunluğu) hem de yönü vardır.

Günlük hayatta: "Kuzeye 5 km yürü" bir vektördür. "5 km" tek başına (büyüklük) bir skalerdir.

Matematiksel olarak, $n$ boyutlu bir vektör, sıralı $n$ sayıdan oluşur:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

**Derin öğrenme bağlamında:** Bir gri tonlamalı $28 \times 28$ görüntüyü düzleştirdiğinde (flatten), $784$ boyutlu bir vektör elde edersin.

### Vektör Toplama

İki vektör **eleman eleman** toplanır:

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Geometrik yorum:**

```
        b╱
        ╱
  a+b  ╱
──────▶╱
      ╱
     ╱
────▶─────▶
  a     b

  (Paralelkenar kuralı)
```

> 📌 **Makale Bağlantısı:** Adversarial saldırıda yapılan tam olarak bir vektör toplamasıdır:
> $$\mathbf{x}_{adv} = \mathbf{x} + \boldsymbol{\delta}$$
> Orijinal görüntü vektörüne ($\mathbf{x}$) bir pertürbasyon vektörü ($\boldsymbol{\delta}$) eklenir.

### Skaler Çarpım

Bir vektörü bir skalerle çarpmak, onu **ölçeklendirir** (uzatır/kısaltır):

$$c \cdot \mathbf{a} = c \cdot \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} c \cdot a_1 \\ c \cdot a_2 \end{bmatrix}$$

```
  c = 2:     ──▶  ────────▶    (aynı yön, 2 kat uzun)
  c = -1:    ──▶  ◀──          (ters yön, aynı uzunluk)
  c = 0.5:   ──▶  ─▶           (aynı yön, yarı uzunluk)
```

### İç Çarpım (Dot Product / Nokta Çarpım)

İki vektörün iç çarpımı bir **skaler** üretir:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**Geometrik yorumu:**

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos\theta$$

burada $\theta$ iki vektör arasındaki açıdır.

| Durum | $\cos\theta$ | İç çarpım | Anlam |
|:---|:---:|:---:|:---|
| $\theta = 0°$ | $1$ | Maksimum pozitif | Aynı yön |
| $\theta = 90°$ | $0$ | Sıfır | Dik (ortogonal) |
| $\theta = 180°$ | $-1$ | Maksimum negatif | Ters yön |

> 💡 **İpucu:** İç çarpım, iki vektörün ne kadar "benzer yöne baktığını" ölçer. Derin öğrenmede **dikkat mekanizması** (attention, Parça 3) tam olarak iç çarpım kullanarak "benzerlik" hesaplar.

**Sayısal örnek:**

$$\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$$

$$\mathbf{a} \cdot \mathbf{b} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$$

### Norm — Vektörün Büyüklüğü

Norm, bir vektörün "uzunluğunu" ölçer. Farklı normlar farklı ölçüm biçimleridir:

| Norm | Formül | Analoji |
|:---|:---|:---|
| $L_1$ normu | $\|\mathbf{x}\|_1 = \sum_i \|x_i\|$ | Manhattan mesafesi (bloklar arası yürüme) |
| $L_2$ normu | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Kuş uçuşu mesafe (Öklid) |
| $L_\infty$ normu | $\|\mathbf{x}\|_\infty = \max_i \|x_i\|$ | En büyük bileşen |

```
  L₁ normu              L₂ normu             L∞ normu
  (Manhattan)           (Öklid)              (Chebyshev)

  ┌───┬───┐            ┌───┬───┐            ┌───┬───┐
  │   │   │            │  ╱│   │            │███████│
  ├───┤   │            │╱  │   │            │███████│
  │   │   │            │   │   │            │███████│
  └───┴───┘            └───┴───┘            └───┴───┘
  Birim top: elmas     Birim top: daire     Birim top: kare
```

> 📌 **Makale Bağlantısı:** Adversarial pertürbasyonların boyutu normlarla ölçülür. $\|\boldsymbol{\delta}\|_\infty \leq \epsilon$ kısıtı, her pikseldeki değişikliğin $\epsilon$'dan küçük olmasını garanti eder. Parça 4'te bu norm kısıtlarını detaylı göreceğiz.

**Sayısal örnek:**

$$\mathbf{x} = \begin{bmatrix} 3 \\ -4 \end{bmatrix}$$

$$\|\mathbf{x}\|_1 = |3| + |-4| = 7$$
$$\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2} = \sqrt{9+16} = 5$$
$$\|\mathbf{x}\|_\infty = \max(|3|, |-4|) = 4$$

> **Öğrendiklerimiz — Vektörler**
>
> ✅ Vektör = sıralı sayı listesi, hem büyüklük hem yön taşır
> ✅ Vektör toplama: adversarial saldırının temel işlemi ($\mathbf{x}_{adv} = \mathbf{x} + \boldsymbol{\delta}$)
> ✅ İç çarpım: benzerlik ölçüsü, dikkat mekanizmasının temeli
> ✅ Norm: vektör büyüklüğü, pertürbasyon bütçesini sınırlar

---

## Matrisler

### Matris Nedir? — Sezgisel Yaklaşım

**Analoji:** Matris bir **dönüşüm makinesidir**. İçine bir vektör atarsın, dönüştürülmüş bir vektör çıkar.

Matematiksel olarak, $m \times n$ boyutlu bir matris:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Derin öğrenme bağlamında:**

- Gri görüntü → $28 \times 28$ matris
- Sinir ağı katmanının ağırlıkları → matris
- Bir katmandaki işlem → matris çarpımı

### Matris Çarpımı

$\mathbf{A}$ ($m \times n$) ile $\mathbf{B}$ ($n \times p$) çarpılırsa sonuç $\mathbf{C}$ ($m \times p$) olur:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

> ⚠️ **Dikkat:** Matris çarpımı için $\mathbf{A}$'nın sütun sayısı = $\mathbf{B}$'nin satır sayısı olmalıdır!

**Adım adım örnek:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**Görsel:**

```
         B sütunları
         ↓  ↓
A    ┌──────────┐
s  → │ 19 │ 22 │  ← A'nın 1.satırı · B'nin sütunları
a  → │ 43 │ 50 │  ← A'nın 2.satırı · B'nin sütunları
t    └──────────┘
```

> 💡 **İpucu:** Bir sinir ağı katmanının temel işlemi şudur:
> $$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
> Bu bir matris-vektör çarpımı + vektör toplamadır. Parça 2'de bunu detaylı göreceğiz.

### Matrisin Transpozu

Satır ve sütunları yer değiştirir:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \implies \mathbf{A}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

Boyut değişimi: $(m \times n) \to (n \times m)$

> 📌 **Makale Bağlantısı:** Dikkat mekanizmasında (Parça 3) $QK^T$ çarpımı yapılır — buradaki $K^T$, $K$ matrisinin transpozudur.

### Determinant ve Ters Matris

**Determinant** ($\det(\mathbf{A})$ veya $|\mathbf{A}|$): Matrisin temsil ettiği dönüşümün alanı ne kadar büyüttüğünü/küçülttüğünü ölçer.

$2 \times 2$ matris için:

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**Ters matris** ($\mathbf{A}^{-1}$): Dönüşümü geri alan matris.

$$\mathbf{A} \cdot \mathbf{A}^{-1} = \mathbf{I}$$

$\mathbf{I}$ birim matristir (köşegen 1, geri kalan 0).

$2 \times 2$ matris için:

$$\mathbf{A}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

> ⚠️ **Dikkat:** $\det(\mathbf{A}) = 0$ ise ters matris yoktur (tekil / singular matris).

### Matris Dönüşüm Olarak — Geometrik Yorum

```
Orijinal          Ölçekleme          Döndürme           Kayma
  ┌──┐          ┌────────┐            ╱╲              ┌──────┐
  │  │          │        │           ╱  ╲             │     ╱│
  │  │    →     │        │    →     ╱    ╲     →     │    ╱ │
  └──┘          │        │         ╱──────╲          │   ╱  │
                └────────┘                            └──╱───┘

[2 0]          [cos -sin]          [1  k]
[0 2]          [sin  cos]          [0  1]
```

> **Öğrendiklerimiz — Matrisler**
>
> ✅ Matris = dönüşüm makinesi (vektörü alır, dönüştürür)
> ✅ Matris çarpımı: sinir ağının temel işlemi
> ✅ Transpose: dikkat mekanizmasında kullanılır ($QK^T$)
> ✅ Determinant: dönüşümün ölçeğini belirler
> ✅ Ters matris: dönüşümü geri alır

---

## Tensörler — Derin Öğrenmenin Veri Yapısı

### Tensör Nedir?

Tensör, matrisin daha yüksek boyutlu genellemesidir:

```
Boyut   İsim          DL Örneği                    Şekil
─────   ────          ─────────                    ─────
  0D    Skaler        Kayıp değeri                 ()
  1D    Vektör        Önyargı (bias)               (512,)
  2D    Matris        Gri görüntü / Ağırlık        (28, 28)
  3D    3-Tensör      Renkli görüntü               (3, 224, 224)
  4D    4-Tensör      Görüntü batch'i              (32, 3, 224, 224)
  5D    5-Tensör      Video batch'i                (32, 10, 3, 224, 224)
```

**Renkli görüntü tensörü:**

```
        Genişlik (W)
        ───────────→
    │  ┌─────────────┐
    │  │ R R R R R R │   Kırmızı kanal
Yük.│  │ R R R R R R │
(H) │  │ R R R R R R │
    │  └─────────────┘
    ↓  ┌─────────────┐
       │ G G G G G G │   Yeşil kanal
       │ G G G G G G │
       │ G G G G G G │
       └─────────────┘
       ┌─────────────┐
       │ B B B B B B │   Mavi kanal
       │ B B B B B B │
       │ B B B B B B │
       └─────────────┘

       Şekil: (3, H, W) veya (H, W, 3)
```

> 📌 **Makale Bağlantısı:** Hedef makalelerdeki modeller, girdi olarak $(B, 3, 224, 224)$ şeklinde tensörler alır — $B$ adet renkli görüntünün bir arada işlenmesidir (batch).

### Tensör İşlemleri Özet Tablosu

| İşlem | Açıklama | Örnek |
|:---|:---|:---|
| Yeniden şekillendirme (reshape) | Boyutları değiştirir | $(28,28) \to (784,)$ |
| Dilimleme (slicing) | Alt tensör seçer | Sadece kırmızı kanal |
| Yayınlama (broadcasting) | Boyut uyumsuzluklarını otomatik çözer | Skaler + matris |
| Eleman bazlı işlemler | Her eleman bağımsız | $\mathbf{A} \odot \mathbf{B}$ (Hadamard çarpım) |

> **Öğrendiklerimiz — Tensörler**
>
> ✅ Tensör = çok boyutlu dizi, derin öğrenmenin temel veri yapısı
> ✅ Görüntüler 3D tensör, batch'ler 4D tensör
> ✅ Reshape, slicing, broadcasting temel tensör işlemleri

---

## Özdeğer ve Özvektör — Sezgisel Giriş

### Analoji: Rüzgâr ve Bayraklar

Bir dönüşüm matrisini "rüzgâr" olarak düşün. Çoğu nesne (vektör) bu rüzgârda hem yönünü hem boyutunu değiştirir. Ama bazı özel yönler vardır ki, rüzgâr sadece onları **uzatır veya kısaltır**, yönlerini değiştirmez.

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$: özvektör (eigenvector) — yönü değişmeyen vektör
- $\lambda$: özdeğer (eigenvalue) — ne kadar uzatıldığı/kısaltıldığı

```
Genel vektör:                     Özvektör:

  ──▶   ─A─▶   ────▶↗            ──▶   ─A─▶   ──────▶
  Girdi        Çıktı              Girdi        Çıktı
              (yön değişti)                   (sadece uzadı, yön aynı)
```

Bu konuyu şimdilik sezgisel düzeyde bırakıyoruz. İleri konularda (özellikle PCA ve bazı optimizasyon analizlerinde) karşımıza tekrar çıkacak.

---

\newpage

# Kalkülüs Temelleri

## Türev Nedir? — Anlık Değişim Hızı

### Analoji: Arabanın Hız Göstergesi

Bir araba düşün. Yolda toplam 100 km gittiysen ve 2 saat sürdüyse, **ortalama hızın** 50 km/s'tir. Ama hız göstergesine baktığında, o **anda** 73 km/s gösterebilir. İşte türev, bu anlık hızı veren matematiksel araçtır.

### Matematiksel Tanım

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Bu formül şunu söyler: "$h$ sonsuz küçülürken, fonksiyondaki değişimin $h$'ye oranı nedir?"

**Geometrik yorum:** Türev, fonksiyon eğrisine çizilen **teğet doğrunun eğimidir**.

```
    f(x)
    │         ╱·  teğet doğru (eğimi = f'(x₀))
    │        ╱
    │      ·╱
    │     ╱·
    │    ╱  ·
    │   ╱    ·
    │  ╱      ···
    │ ╱          ···
    │╱               ···
    ┼───────────────────── x
              x₀
```

### Temel Türev Kuralları

| Fonksiyon | Türev | Örnek |
|:---|:---|:---|
| $f(x) = c$ (sabit) | $f'(x) = 0$ | $(5)' = 0$ |
| $f(x) = x^n$ (kuvvet) | $f'(x) = nx^{n-1}$ | $(x^3)' = 3x^2$ |
| $f(x) = e^x$ (üstel) | $f'(x) = e^x$ | $(e^x)' = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = \frac{1}{x}$ | $(\ln x)' = \frac{1}{x}$ |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ | $(\sin x)' = \cos x$ |

**Toplama kuralı:** $(f + g)' = f' + g'$

**Çarpım kuralı:** $(f \cdot g)' = f' \cdot g + f \cdot g'$

**Bölüm kuralı:** $\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}$

> **Öğrendiklerimiz — Türev**
>
> ✅ Türev = anlık değişim hızı = teğet doğrunun eğimi
> ✅ Temel kuralları bilmek yeterli (kuvvet, üstel, toplama, çarpım)

---

## Zincir Kuralı (Chain Rule) — En Kritik Kural

### Neden Bu Kadar Önemli?

Zincir kuralı, **derin öğrenmenin motorudur**. Sinir ağları iç içe geçmiş fonksiyonlardan oluşur ve geri yayılım (backpropagation, Parça 2) tamamen zincir kuralına dayanır.

### Analoji: Domino Etkisi

Bir fabrikada 3 makine düşün:

```
  Hammadde ──▶ [Makine A] ──▶ [Makine B] ──▶ [Makine C] ──▶ Ürün
     x              g(x)          f(g(x))       h(f(g(x)))
```

Hammadde $x$'te küçük bir değişiklik, Makine A'nın çıktısını değiştirir. Bu değişiklik Makine B'ye yayılır, oradan Makine C'ye... **Her aşamadaki değişim oranları çarpılır.**

### Formül

İç içe fonksiyon $h(x) = f(g(x))$ için:

$$h'(x) = f'(g(x)) \cdot g'(x)$$

Genel zincir ($y = f_n(f_{n-1}(\cdots f_2(f_1(x)) \cdots))$):

$$\frac{dy}{dx} = \frac{df_n}{df_{n-1}} \cdot \frac{df_{n-1}}{df_{n-2}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

### Örnek

$h(x) = (3x + 2)^5$ bul $h'(x)$:

- Dış fonksiyon: $f(u) = u^5 \implies f'(u) = 5u^4$
- İç fonksiyon: $g(x) = 3x + 2 \implies g'(x) = 3$

$$h'(x) = 5(3x+2)^4 \cdot 3 = 15(3x+2)^4$$

> 📌 **Makale Bağlantısı:** Parça 2'de öğreneceğimiz backpropagation, zincir kuralının bir sinir ağının tüm katmanları boyunca uygulanmasıdır. Parça 4'te göreceğimiz adversarial gradyanlar da aynı zincir kuralıyla hesaplanır.

> **Öğrendiklerimiz — Zincir Kuralı**
>
> ✅ İç içe fonksiyonlarda türev = dış türev × iç türev
> ✅ Backpropagation'ın (Parça 2) ve adversarial gradyanların (Parça 4) matematiksel temeli
> ✅ Derin ağlarda her katmandaki türevler çarpılarak girişe kadar taşınır

---

## Kısmi Türev ve Gradyan Vektörü

### Birden Fazla Değişkenli Fonksiyonlar

Gerçek hayatta fonksiyonlar genellikle birden fazla değişkene bağlıdır. Örneğin bir sinir ağının kaybı binlerce ağırlık parametresine bağlıdır.

### Kısmi Türev (Partial Derivative)

Çok değişkenli bir fonksiyonda **bir değişkene göre** türev alıp diğerlerini sabit tutmaktır.

$$f(x, y) = x^2 + 3xy + y^2$$

$$\frac{\partial f}{\partial x} = 2x + 3y \qquad \text{(y sabit tutulur)}$$

$$\frac{\partial f}{\partial y} = 3x + 2y \qquad \text{(x sabit tutulur)}$$

### Gradyan Vektörü — "En Dik Yokuş Yönü"

**Analoji:** Bir dağda duruyorsun ve sis var. Hangi yöne adım atarsan en hızlı yükselirsin? **Gradyan**, tam olarak bu yönü gösteren oktur.

Gradyan, tüm kısmi türevlerin bir vektörde toplanmasıdır:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Önemli özellikler:**

| Özellik | Açıklama |
|:---|:---|
| **Yönü** | Fonksiyonun en hızlı **arttığı** yön |
| **Büyüklüğü** | Artış hızının ne kadar dik olduğu |
| **Tersi** ($-\nabla f$) | En hızlı **azaldığı** yön |

```
         Eş yükselti eğrileri (contour plot)

    y    ╲  ╲  ╲   ╱  ╱  ╱
    │     ╲  ╲  ╲ ╱  ╱  ╱
    │      ╲  ╲  ●──▶ ╱         ← ∇f (en dik yokuş yönü)
    │       ╲  ╱ ╲╱  ╱
    │        ╱╲   ╲  ╱
    │       ╱  ╲   ╲╱
    └──────────────────── x

    ● = mevcut konum
    ──▶ = gradyan yönü (en hızlı artış)
```

### Örnek

$$f(x, y) = x^2 + y^2$$

$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

$(1, 2)$ noktasında: $\nabla f(1,2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

Bu, $(1,2)$'den $(2,4)$ yönüne gidilirse fonksiyonun en hızlı artacağını söyler.

> 📌 **Makale Bağlantısı:** Adversarial saldırılar (Parça 4) gradyanı şöyle kullanır:
>
> - **Eğitim:** $-\nabla_\theta L$ yönünde git (kaybı **azalt** = öğren)
> - **Saldırı:** $+\nabla_x L$ yönünde git (kaybı **artır** = kandır)
>
> Aynı araç, zıt amaçlarla kullanılır!

> **Öğrendiklerimiz — Kısmi Türev & Gradyan**
>
> ✅ Kısmi türev = bir değişkene göre türev, diğerleri sabit
> ✅ Gradyan = tüm kısmi türevlerin vektörü
> ✅ Gradyan yönü = en dik artış; tersi = en dik azalış
> ✅ Eğitim: $-\nabla L$ (kaybı azalt); Saldırı: $+\nabla L$ (kaybı artır)

---

## Gradyan İnişi (Gradient Descent) — Öğrenmenin Motoru

### Analoji: Sisli Dağda İniş

Bir dağın zirvesinden en alçak noktaya (vadi) inmek istiyorsun, ama yoğun sis var — etrafı göremiyorsun. Tek yapabileceğin: **ayağının altındaki zeminin eğimine bakmak ve en dik iniş yönüne doğru bir adım atmak**. Bunu tekrar tekrar yaparsan vadiye ulaşırsın.

Bu tam olarak **gradyan inişi**dir.

### Formül

$$\theta_{yeni} = \theta_{eski} - \alpha \nabla_\theta L(\theta)$$

| Sembol | Anlam | Analoji |
|:---|:---|:---|
| $\theta$ | Parametreler (ağırlıklar) | Dağdaki konumun |
| $\alpha$ | Öğrenme hızı (learning rate) | Adım büyüklüğün |
| $\nabla_\theta L$ | Gradyan | Zeminin eğimi |
| $L(\theta)$ | Kayıp fonksiyonu | Yükseklik |

### Öğrenme Hızının Etkisi

```
  α çok büyük:           α uygun:             α çok küçük:

  ╲    ╱╲   ╱            ╲                     ╲
   ╲  ╱  ╲ ╱              ╲  ·                  ╲ ·
    ╲╱    ╳                 ╲  ·                  ╲ ·
    ╱╲   ╱ ╲                 ╲  ·                  ╲  ·
   ╱  ╲╱    ╲                 ·──●                  ╲  ·
                                                     ╲  · (çok yavaş)
  Iraksar (diverge)     Yakınsar (converge)    Yakınsar ama çok yavaş
```

> ⚠️ **Dikkat:** Öğrenme hızı ($\alpha$) çok kritiktir. Çok büyükse model öğrenemez (zıplar), çok küçükse çok yavaş öğrenir. Parça 2'de Adam gibi adaptif yöntemler bunu otomatikleştiren teknikler göreceğiz.

### Sayısal Örnek

$f(x) = x^2 - 4x + 4$ fonksiyonunu minimize edelim ($\alpha = 0.1$):

$$f'(x) = 2x - 4$$

| Adım | $x$ | $f(x)$ | $f'(x)$ | $x_{yeni} = x - 0.1 \cdot f'(x)$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0 | 4.0 | -4.0 | 0.4 |
| 1 | 0.4 | 2.56 | -3.2 | 0.72 |
| 2 | 0.72 | 1.64 | -2.56 | 0.976 |
| 3 | 0.976 | 1.05 | -2.05 | 1.181 |
| ... | ... | ... | ... | ... |
| 20 | ~2.0 | ~0.0 | ~0.0 | ~2.0 |

Fonksiyon $x=2$'de minimum değerine ($f(2)=0$) ulaşır ve gradyan inişi oraya yakınsar.

```
  f(x)
  4 │·
    │  ·
  3 │    ·
    │      ·
  2 │        ·
    │   ①     ·
  1 │    ②     ·
    │     ③     ·
  0 │──────●─────── x
    0  1   2   3  4
         minimum

  ① ② ③ = gradyan inişi adımları
```

> 📌 **Makale Bağlantısı:** Parça 2'de sinir ağlarını eğitirken tam olarak bu yöntemi kullanacağız. Parça 4'te ise adversarial saldırıların gradyan **yükselişi** (gradient ascent) kullandığını göreceğiz — aynı formül, artı işaret:
> $$x_{adv} = x + \alpha \nabla_x L$$

> **Öğrendiklerimiz — Gradyan İnişi**
>
> ✅ Gradyan inişi = en dik iniş yönünde iteratif adımlar
> ✅ $\theta_{yeni} = \theta_{eski} - \alpha \nabla L$ (öğrenme)
> ✅ $x_{adv} = x + \alpha \nabla L$ (saldırı — tam tersi!)
> ✅ Öğrenme hızı ($\alpha$) seçimi kritik

---

\newpage

# Parça 1A — Kavram Haritası

```
╔════════════════════════════════════════════════════════════════════════╗
║                    PARÇA 1A — KAVRAM HARİTASI                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  LİNEER CEBİR                           KALKÜLÜS                    ║
║  ───────────                            ─────────                    ║
║                                                                      ║
║  Skaler ──▶ Vektör ──▶ Matris ──▶ Tensör    Türev (anlık değişim)   ║
║               │           │          │          │                     ║
║               ▼           ▼          ▼          ▼                     ║
║           İç çarpım   Matris      Görüntü   Zincir Kuralı           ║
║              │        çarpımı     verisi      │                      ║
║              ▼           │                    ▼                       ║
║           Norm ◄─────────┘              Kısmi Türev                  ║
║              │                              │                        ║
║              ▼                              ▼                        ║
║        Pertürbasyon                     Gradyan ◄──── "En dik yön"  ║
║         bütçesi (ε)                        │                         ║
║              │                              ▼                        ║
║              └──────────────────▶ Gradyan İnişi                      ║
║                                       │        │                     ║
║                                       ▼        ▼                     ║
║                                   Öğrenme   Saldırı                  ║
║                                  (−∇L)     (+∇L)                    ║
║                                  Parça 2    Parça 4                  ║
║                                                                      ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

# Parça 1B'ye Köprü

Bu ilk bölümde lineer cebir ve kalkülüsün temellerini öğrendik. Şimdi bildiklerimizi özetleyelim ve bir sonraki bölümde ne beklediğimize bakalım:

**Parça 1B'de öğreneceklerimiz:**

1. **Olasılık & İstatistik** — Modellerin "ne kadar emin olduğunu" anlamamız için
2. **Python Programlama** — Fikirleri koda dönüştürebilmek için
3. **NumPy & Matplotlib** — Vektör/matris işlemlerini bilgisayarda yapabilmek ve sonuçları görselleştirebilmek için
4. **Kod Atölyesi** — Öğrendiğimiz her şeyi birleştiren pratik çalışmalar

Tüm bu matematiksel araçlar, Parça 2'de bir "öğrenen sistem" — yani sinir ağı — inşa etmemizin temelini oluşturacak. Gradyan inişi öğrenmenin motoru olacak, matris çarpımları her katmanda çalışacak ve olasılık dağılımları modelin çıktısını yorumlamamızı sağlayacak.

---

*Devam: Parça 1B — Olasılık, Python, NumPy & Kod Atölyesi →*
