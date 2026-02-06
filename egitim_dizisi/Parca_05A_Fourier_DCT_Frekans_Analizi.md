---
title: "Parça 5A: Frekans Domain Analizi — Fourier Dönüşümü & DCT"
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
  - \fancyhead[L]{Parça 5A — Fourier, DCT \& Frekans}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Giriş ve Motivasyon

## Yeni Bir Perspektif: Frekans Uzayı

Parça 4'te adversarial pertürbasyonları **piksel uzayında** inceledik: her piksele ne kadar eklendi, hangi yöne? Şimdi aynı pertürbasyona **frekans gözlüğüyle** bakacağız.

**Analoji:** Bir müzik parçasını düşün. Dalga formu olarak bakabilirsin (zamanda ne oluyor?) ya da frekans spektrumu olarak (hangi notalar, hangi enstrümanlar çalıyor?). İkisi de aynı müziği tanımlar, ama farklı bilgiler verir.

```
  Piksel uzayı:              Frekans uzayı:
  "Bu pikseller değişti"     "Yüksek frekanslar bozuldu"

  ┌────────────────┐         ┌────────────────┐
  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒ │         │ Düşük │ Yüksek│
  │ ▒▒▓▒▒▒▒▒▒▒▒▒▒ │   FFT   │ frek. │ frek. │
  │ ▒▒▒▒▒▒▒▒▒▓▒▒▒ │  ────▶  │  ●    │ ●●●●● │
  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒ │         │       │ ●●●●● │
  └────────────────┘         └───────┴───────┘
  Pertürbasyon pikselleri    Pertürbasyon çoğunlukla
                             yüksek frekanslarda!
```

> 📌 **Makale Bağlantısı:** Makale B'deki **FE (Frequency Enhancement)** modülü, tam olarak frekans domain bilgisini kullanarak adversarial pertürbasyonları ve sahtecilik izlerini tespit eder.

---

\newpage

# Sinyal İşleme Temelleri

## Frekans Nedir? — Görüntülerde

**Seste:**
- **Düşük frekans** = bas ses (davul, kontrbas) → yavaş titreşim
- **Yüksek frekans** = tiz ses (zil, flüt) → hızlı titreşim

**Görüntüde:**
- **Düşük frekans** = yavaş değişen bölgeler (gökyüzü, düz alanlar, genel şekiller)
- **Yüksek frekans** = hızlı değişen bölgeler (kenarlar, dokular, gürültü, ince detaylar)

```
  Düşük frekans (görüntüde):       Yüksek frekans (görüntüde):

  ┌──────────────────┐              ┌──────────────────┐
  │ ░░░░░░░░░░░░░░░░ │              │ ┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼ │
  │ ░░░░░░░░░░░░░░░░ │              │ ┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼ │
  │ ░░░░▓▓▓▓▓▓░░░░░░ │              │ ──────────────── │
  │ ░░░░▓▓▓▓▓▓░░░░░░ │              │ ┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼ │
  │ ░░░░░░░░░░░░░░░░ │              │ ──────────────── │
  │ ░░░░░░░░░░░░░░░░ │              │ ┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼ │
  └──────────────────┘              └──────────────────┘
  Yumuşak geçişler                  Keskin kenarlar, tekrar eden desen
  (genel şekil bilgisi)            (doku, detay, gürültü)
```

**Uzamsal domain vs. frekans domain:**

> 💡 **İpucu — Pasta Analojisi:** Bir pasta düşün. **Uzamsal domain** = pastanın kendisi (gördüğün şey). **Frekans domain** = tarifesi (hangi malzeme ne kadar?). İkisi de aynı pastayı tanımlar, sadece bakış açısı farklıdır. Fourier dönüşümü pastadan tarife, ters Fourier dönüşümü tariften pastaya gider.

---

\newpage

# Fourier Dönüşümü (Fourier Transform)

## Joseph Fourier'in Dahice Fikri (1807)

**Her periyodik sinyal, farklı frekanslardaki sinüs ve kosinüs dalgalarının toplamı olarak yazılabilir.**

```
  Orijinal sinyal = Düşük frekans + Orta frekans + Yüksek frekans

       ╱╲              ╱╲╱╲           ╱╲╱╲╱╲╱╲╱╲
      ╱  ╲            ╱    ╲         ╱            ╲
  ───╱────╲───   +   ╱──────╲───  +  ╱──────────────╲──
     Temel            Harmonik 2      Harmonik 3
     (düşük f)        (orta f)        (yüksek f)
```

## 1D Ayrık Fourier Dönüşümü (DFT)

$$F(u) = \sum_{x=0}^{N-1} f(x) \cdot e^{-j2\pi ux/N}$$

**Euler formülü** ile açarsak:

$$e^{-j\theta} = \cos(\theta) - j\sin(\theta)$$

Her $F(u)$ karmaşık (complex) bir sayıdır: gerçel kısım + sanal kısım.

## 2D DFT — Görüntüler İçin

Görüntüler 2 boyutlu olduğundan, hem yatay hem dikey frekansları dönüştürürüz:

$$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

## Genlik ve Faz Spektrumu

$$|F(u,v)| = \sqrt{\text{Re}(F)^2 + \text{Im}(F)^2} \quad \text{(Genlik / Magnitude)}$$

$$\phi(u,v) = \arctan\left(\frac{\text{Im}(F)}{\text{Re}(F)}\right) \quad \text{(Faz / Phase)}$$

```
  Orijinal Görüntü      Genlik Spektrumu       Faz Spektrumu
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │              │      │      ●       │      │ ╱╲  ╱╲╱╲    │
  │   [Görüntü]  │ FFT  │    ●●●●     │      │╱  ╲╱     ╲╱  │
  │              │ ──▶  │  ●●●●●●●    │      │   ╱╲╱╲  ╱╲  │
  │              │      │    ●●●●     │      │  ╱    ╲╱  ╲ │
  │              │      │      ●       │      │ ╱          ╲│
  └──────────────┘      └──────────────┘      └──────────────┘

  Genlik: "Her frekanstan ne kadar var?"
  Faz: "Bu frekanslar uzayda nerede?"
```

**Önemli:** Genlik spektrumunda **merkez = düşük frekans**, **kenarlar = yüksek frekans**.

## Hızlı Fourier Dönüşümü (FFT)

DFT hesaplaması $O(N^2)$ — yavaş. FFT aynı sonucu $O(N \log N)$'de hesaplar.

## Frekans Filtreleme

```
  Alçak geçiren (low-pass):     Yüksek geçiren (high-pass):

  ┌──────────────┐              ┌──────────────┐
  │ ░░░░░░░░░░░░ │              │ ████████████ │
  │ ░░░░████░░░░ │              │ ████░░░░████ │
  │ ░░██████░░░░ │   Geçir:    │ ██░░░░░░░░██ │   Geçir:
  │ ░░██████░░░░ │   Merkez    │ ██░░░░░░░░██ │   Kenarlar
  │ ░░░░████░░░░ │   (düşük f) │ ████░░░░████ │   (yüksek f)
  │ ░░░░░░░░░░░░ │              │ ████████████ │
  └──────────────┘              └──────────────┘

  Sonuç: bulanık görüntü          Sonuç: kenarlar / detaylar
```

## Kod: FFT ile Görüntü Analizi

```python
import numpy as np
import matplotlib.pyplot as plt

# ─── Görüntü yükle (basit örnek) ───
from PIL import Image
# Gri tonlamalı bir görüntü kullanıyoruz
img = np.random.rand(128, 128)  # Veya gerçek görüntü yükle

# ─── 2D FFT ───
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)  # Düşük frekansı merkeze taşı
magnitude = np.log(1 + np.abs(F_shifted))  # Log ölçek (görselleştirme için)
phase = np.angle(F_shifted)

# ─── Görselleştirme ───
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Orijinal Görüntü')
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title('Genlik Spektrumu (log)')
axes[1].axis('off')

axes[2].imshow(phase, cmap='twilight')
axes[2].set_title('Faz Spektrumu')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('fft_analizi.png', dpi=150)
plt.show()

# ─── Frekans filtreleme ───
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
radius = 20  # Kesim frekansı

# Alçak geçiren filtre (low-pass)
mask_low = np.zeros((rows, cols))
Y, X = np.ogrid[:rows, :cols]
mask_low[((X - ccol)**2 + (Y - crow)**2) <= radius**2] = 1

F_low = F_shifted * mask_low
img_low = np.real(np.fft.ifft2(np.fft.ifftshift(F_low)))

# Yüksek geçiren filtre (high-pass)
mask_high = 1 - mask_low
F_high = F_shifted * mask_high
img_high = np.real(np.fft.ifft2(np.fft.ifftshift(F_high)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Orijinal')
axes[1].imshow(img_low, cmap='gray')
axes[1].set_title(f'Alçak Geçiren (r={radius})\n(düşük frekans = genel şekil)')
axes[2].imshow(img_high, cmap='gray')
axes[2].set_title(f'Yüksek Geçiren (r={radius})\n(yüksek frekans = kenarlar)')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('frekans_filtreleme.png', dpi=150)
plt.show()
```

> **Öğrendiklerimiz — Fourier Dönüşümü**
>
> ✅ FFT: görüntüyü frekans bileşenlerine ayırır
> ✅ Genlik spektrumu: her frekanstan ne kadar var
> ✅ Düşük frekans = genel şekil, yüksek frekans = kenarlar/detay/gürültü
> ✅ Filtreleme: belirli frekans bantlarını seçme/kaldırma

---

\newpage

# Ayrık Kosinüs Dönüşümü (DCT)

## DCT vs. FFT

DCT, FFT'ye benzer ama sadece **kosinüs** baz fonksiyonları kullanır → **reel değerli** çıktı (karmaşık sayı yok).

$$C(u) = \alpha(u) \sum_{x=0}^{N-1} f(x) \cos\left(\frac{(2x+1)u\pi}{2N}\right)$$

## JPEG Sıkıştırmadaki Rolü

JPEG formatı DCT'yi doğrudan kullanır:

```
  JPEG Sıkıştırma Hattı:

  Görüntü → 8×8 blok → DCT → Kuantizasyon → Entropi kodlama → .jpg
            ──┴──
              │
  ┌──────────────────────────────────────┐
  │  8×8 DCT Katsayı Matrisi:           │
  │  ┌─────────────────────────┐         │
  │  │ DC │ ↗  ↗  ↗  ↗  ↗  ↗ │         │
  │  │ ↗  │                    │ Düşük f │
  │  │ ↗  │         ↗         │    ↗    │
  │  │ ↗  │      ↗            │  Yüksek │
  │  │ ↗  │   ↗               │    f    │
  │  │ ↗  │↗                  │         │
  │  └─────────────────────────┘         │
  │  Sol üst = düşük frekans (önemli)    │
  │  Sağ alt = yüksek frekans (atılabilir)│
  └──────────────────────────────────────┘
```

> 📌 **Makale Bağlantısı:** Makale B'deki **FE (Frequency Enhancement)** modülü, DCT tabanlı frekans zenginleştirme kullanır. Görüntüyü bloklara ayırıp DCT uygular, belirli frekans bantlarını güçlendirir ve geri dönüştürür.

## FFT vs. DCT Karşılaştırması

| Özellik | FFT | DCT |
|:---|:---|:---|
| Baz fonksiyonları | Sinüs + kosinüs | Sadece kosinüs |
| Çıktı tipi | Karmaşık sayı | Reel sayı |
| Enerji yoğunlaşması | Dağınık | Sol üst köşede yoğun |
| Yaygın kullanım | Sinyal analizi | JPEG, video sıkıştırma |
| Makale bağlantısı | Genel analiz | Makale B'nin FE modülü |

## Kod: DCT Uygulaması

```python
from scipy.fft import dctn, idctn
import numpy as np
import matplotlib.pyplot as plt

# ─── 8×8 blok DCT (JPEG tarzı) ───
img = np.random.rand(64, 64)

# Blok bazlı DCT
block_size = 8
dct_img = np.zeros_like(img)
for i in range(0, img.shape[0], block_size):
    for j in range(0, img.shape[1], block_size):
        block = img[i:i+block_size, j:j+block_size]
        dct_img[i:i+block_size, j:j+block_size] = dctn(block, norm='ortho')

# ─── Frekans bantlarını seçme ───
# Düşük frekans: sol üst köşe
# Yüksek frekans: sağ alt köşe

low_freq = np.zeros_like(dct_img)
high_freq = np.zeros_like(dct_img)

for i in range(0, img.shape[0], block_size):
    for j in range(0, img.shape[1], block_size):
        block = dct_img[i:i+block_size, j:j+block_size]
        # Düşük frekans (sol üst 3×3)
        low_block = np.zeros((block_size, block_size))
        low_block[:3, :3] = block[:3, :3]
        low_freq[i:i+block_size, j:j+block_size] = low_block
        # Yüksek frekans (geri kalan)
        high_block = block.copy()
        high_block[:3, :3] = 0
        high_freq[i:i+block_size, j:j+block_size] = high_block

# Geri dönüşüm
img_low = np.zeros_like(img)
img_high = np.zeros_like(img)
for i in range(0, img.shape[0], block_size):
    for j in range(0, img.shape[1], block_size):
        img_low[i:i+block_size, j:j+block_size] = idctn(
            low_freq[i:i+block_size, j:j+block_size], norm='ortho')
        img_high[i:i+block_size, j:j+block_size] = idctn(
            high_freq[i:i+block_size, j:j+block_size], norm='ortho')

print("DCT frekans ayrıştırması tamamlandı.")
```

---

\newpage

# Frekans Domain'de Adversarial Pertürbasyon Analizi

## Temel Soru: Pertürbasyonlar Frekans Uzayında Nerede?

```
  FGSM Pertürbasyonu        PGD Pertürbasyonu         C&W Pertürbasyonu
  (Frekans spektrumu)       (Frekans spektrumu)       (Frekans spektrumu)

  Düşük │ Yüksek            Düşük │ Yüksek            Düşük │ Yüksek
  ──────┼──────             ──────┼──────             ──────┼──────
   ●    │ ●●●●●              ●●   │ ●●●●●              ●●  │ ●●●
        │ ●●●●●              ●●   │ ●●●●               ●●  │ ●●
        │ ●●●●                    │ ●●●                     │ ●●
                                  │ ●●
  Ağırlıklı yüksek f       Dağınık ama yüksek f      Daha hedefli
  (sign fonksiyonu etkisi)   dominant                  (optimizasyon etkisi)
```

## CNN vs. ViT Frekans Duyarlılığı

| Mimari | Duyarlı Olduğu Frekans | Nedeni | Kaynak |
|:---|:---|:---|:---|
| **CNN** | Yüksek frekans | Texture bias — dokusal kalıpları tercih eder | Geirhos et al. (2019) |
| **ViT** | Düşük-orta frekans | Shape bias — genel şekilleri tercih eder | Naseer et al. (2021) |

```
  CNN'in frekans profili:         ViT'in frekans profili:

  Duyarlılık                      Duyarlılık
  █████│                          │
  ████ │                          │    ████
  ███  │                          │  ██████
  ██   │                          │████████
  █    │█████████████             │██████████
  ─────┼──────────── f            ─────┼──────────── f
    düşük    yüksek                 düşük    yüksek

  CNN yüksek frekansa duyarlı    ViT düşük-orta frekansa duyarlı
```

> 📌 **Makale Bağlantısı:** Bu farklılık, Makale A'nın gradyan analizi bulgularıyla uyumludur ve Makale B'nin FE modülünün neden frekans bilgisini kullandığını açıklar.

## Frekans Tabanlı Savunma Fikri

Adversarial pertürbasyonlar çoğunlukla yüksek frekanstaysa, **alçak geçiren filtre** bir savunma olabilir mi?

```
  x_adv ──▶ [Alçak Geçiren Filtre] ──▶ x_temizlenmiş ──▶ [Model] ──▶ tahmin
```

**Sorun:** Kenarlar ve dokular da yüksek frekanstadır! Filtre pertürbasyonu kaldırırken görüntü bilgisini de bozar.

**Daha akıllı yaklaşım (Makale B):** Filtrelemek yerine, frekans bilgisini **ek girdi olarak** modele ver:

```
  x ──────────────────────────────▶ [ViT] ──▶ tahmin
  x ──▶ [DCT] ──▶ [Frekans Seçimi] ──▶ ↑ (birleştir)
                   FE modülü
```

## Kod: Adversarial Pertürbasyonun Frekans Analizi

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_frequency(image, title=""):
    """Bir görüntünün frekans spektrumunu analiz et."""
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    magnitude = np.log(1 + np.abs(F_shifted))
    return magnitude

# Simülasyon: orijinal, pertürbasyon, adversarial
np.random.seed(42)
img = np.random.rand(64, 64) * 0.5 + 0.25  # Orijinal
eps = 8/255
delta_fgsm = eps * np.sign(np.random.randn(64, 64))  # FGSM benzeri
img_adv = np.clip(img + delta_fgsm, 0, 1)

# Frekans analizi
mag_orig = analyze_frequency(img)
mag_delta = analyze_frequency(delta_fgsm)
mag_adv = analyze_frequency(img_adv)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# Üst: uzamsal domain
axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title('Orijinal')
axes[0,1].imshow(delta_fgsm, cmap='RdBu'); axes[0,1].set_title('Pertürbasyon (δ)')
axes[0,2].imshow(img_adv, cmap='gray'); axes[0,2].set_title('Adversarial')
# Alt: frekans domain
axes[1,0].imshow(mag_orig, cmap='hot'); axes[1,0].set_title('Orijinal Spektrum')
axes[1,1].imshow(mag_delta, cmap='hot'); axes[1,1].set_title('Pertürbasyon Spektrumu')
axes[1,2].imshow(mag_adv, cmap='hot'); axes[1,2].set_title('Adversarial Spektrum')
for ax in axes.flat: ax.axis('off')
plt.suptitle('Uzamsal vs. Frekans Domain Karşılaştırması', fontsize=14)
plt.tight_layout()
plt.savefig('frekans_adversarial_analiz.png', dpi=150)
plt.show()
```

> **Öğrendiklerimiz — Frekans ve Adversarial**
>
> ✅ Adversarial pertürbasyonlar genellikle yüksek frekans bölgesinde yoğunlaşır
> ✅ CNN yüksek frekansa, ViT düşük-orta frekansa duyarlıdır
> ✅ Basit filtreleme yetersiz — Makale B, frekansı ek girdi olarak kullanır
> ✅ DCT tabanlı FE modülü bu anlayışın uygulamasıdır

---

# Kavram Haritası & Parça 5B'ye Köprü

```
╔════════════════════════════════════════════════════════════╗
║  PARÇA 5A — KAVRAM HARİTASI                               ║
║                                                            ║
║  Sinyal İşleme                                             ║
║  ├── Düşük frekans: şekiller, düz alanlar                  ║
║  └── Yüksek frekans: kenarlar, doku, gürültü               ║
║       │                                                    ║
║       ├── Fourier Transform (FFT)                          ║
║       │   ├── Genlik spektrumu                             ║
║       │   └── Frekans filtreleme                           ║
║       │                                                    ║
║       └── DCT (Ayrık Kosinüs Dönüşümü)                    ║
║           ├── JPEG sıkıştırma                              ║
║           └── Makale B: FE modülü ★                        ║
║                                                            ║
║  Adversarial + Frekans:                                    ║
║  ├── Pertürbasyonlar → yüksek frekans                      ║
║  ├── CNN: yüksek f duyarlı (texture bias)                  ║
║  └── ViT: düşük-orta f duyarlı (shape bias)                ║
║                                                            ║
║  → PARÇA 5B: Açıklanabilirlik (Grad-CAM, Attention Maps)  ║
╚════════════════════════════════════════════════════════════╝
```

**Parça 5B'de:** Model neden böyle karar veriyor? Grad-CAM, dikkat haritaları ve açıklama tutarlılığı (ECL) kavramlarını öğreneceğiz.

---

*Devam: Parça 5B — Açıklanabilirlik, Medikal Görüntü Analizi & Kod Atölyesi →*
