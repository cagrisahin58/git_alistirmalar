---
title: "Parça 4A: Adversarial Makine Öğrenmesi — Temeller & Saldırı Yöntemleri"
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
  - \fancyhead[L]{Parça 4A — Adversarial Saldırılar}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Giriş ve Motivasyon

## "Aynı Görünen Ama Farklı Yorumlanan" Görüntüler

2013 yılında Szegedy ve arkadaşları şaşırtıcı bir keşif yaptılar: son derece başarılı derin sinir ağları, görüntülere **insan gözüyle fark edilemeyen** küçük pertürbasyonlar eklenerek kolayca kandırılabilir.

```
  ┌──────────────┐        ┌────────────┐        ┌──────────────┐
  │              │        │            │        │              │
  │   🐼 Panda   │   +    │  Gürültü   │   =    │  🦍 Gibbon   │
  │   %57.7      │   ε×   │  (×0.007)  │        │   %99.3      │
  │              │        │            │        │              │
  └──────────────┘        └────────────┘        └──────────────┘
      Orijinal          sign(∇ₓL)×ε              Adversarial

  İnsan: "İkisi de aynı panda"
  Model: "Bu kesinlikle gibbon!"
```

## Bu Parçanın Önemi

> 📌 **Makale Bağlantısı:** Bu parça, **her iki hedef makalenin doğrudan konusudur**:
> - Makale A: CNN ve ViT'in adversarial saldırılara farklı tepkilerini inceler
> - Makale B: Adversarial saldırılara dayanıklı bir medikal görüntü analiz sistemi önerir

## Parça 1-3'ten Köprü

| Önceki Kavram | Adversarial Kullanımı |
|:---|:---|
| Gradyan (Parça 1) | Saldırı yönünü belirler: $\nabla_x L$ |
| Backpropagation (Parça 2A) | Girdiye göre gradyan hesaplar |
| Kayıp fonksiyonu (Parça 2A) | Eğitimde minimize, saldırıda **MAKSİMİZE** edilir |
| CNN ve ViT (Parça 3) | Farklı mimariler, farklı zayıflıklar |

**Temel kavrayış:** Backpropagation'ı **öğrenme** için kullandık (parametrelere göre gradyan). Şimdi aynı aracı **saldırı** için kullanacağız (girdiye göre gradyan).

```
  Eğitim:   ∂L/∂θ  →  θ güncelle  →  kayıp AZALIR
  Saldırı:  ∂L/∂x  →  x güncelle  →  kayıp ARTAR (model kandırılır)
```

---

\newpage

# Adversarial Örnekler: Tanım ve Formalizasyon

## Pertürbasyon Kavramı

**Analoji:** Bir tabloya gözle görülmeyecek kadar ince toz serpme. Tablo insan gözüne aynı görünür, ama özel bir algılayıcı (sinir ağı) artık tabloyu tamamen farklı yorumlar.

### Matematiksel Tanım

Orijinal girdi $\mathbf{x}$, gerçek etiket $y$, model $f_\theta$:

$$\mathbf{x}_{adv} = \mathbf{x} + \boldsymbol{\delta}$$

burada pertürbasyon $\boldsymbol{\delta}$ şu kısıtı sağlar:

$$\|\boldsymbol{\delta}\|_p \leq \varepsilon$$

**Amaç:** $f_\theta(\mathbf{x}_{adv}) \neq y$ (model yanlış sınıflandırır)

## Norm Kısıtları — Pertürbasyon Bütçesi

Farklı $L_p$ normları (Parça 1) farklı pertürbasyon türlerini tanımlar:

| Norm | Formül | Anlam | Görsel |
|:---|:---|:---|:---|
| $L_0$ | $\|\boldsymbol{\delta}\|_0 = $ sıfır olmayan eleman sayısı | Kaç piksel değişti? | Az ama güçlü değişiklikler |
| $L_2$ | $\|\boldsymbol{\delta}\|_2 = \sqrt{\sum \delta_i^2}$ | Toplam Öklid mesafesi | Yaygın, hafif pertürbasyon |
| $L_\infty$ | $\|\boldsymbol{\delta}\|_\infty = \max_i |\delta_i|$ | Herhangi bir pikseldeki max değişim | **En yaygın**, her piksel az değişir |

```
  L₀ normu:               L₂ normu:              L∞ normu:
  (az piksel, büyük       (tüm pikseller,        (tüm pikseller,
   değişiklik)             toplam kısıtlı)        her biri kısıtlı)

  ┌────────────┐          ┌────────────┐          ┌────────────┐
  │  ·  ·  ·  ·│          │ ░░░░░░░░░░ │          │ ▒▒▒▒▒▒▒▒▒▒ │
  │  ·  ·  ·  ·│          │ ░░░░░░░░░░ │          │ ▒▒▒▒▒▒▒▒▒▒ │
  │  ·  █  ·  ·│          │ ░░░░░░░░░░ │          │ ▒▒▒▒▒▒▒▒▒▒ │
  │  ·  ·  ·  ·│          │ ░░░░░░░░░░ │          │ ▒▒▒▒▒▒▒▒▒▒ │
  └────────────┘          └────────────┘          └────────────┘
  █ = büyük değişiklik    ░ = küçük değişiklik    ▒ = eşit küçük değişiklik
```

## Yaygın $\varepsilon$ Değerleri

| $\varepsilon$ | Piksel aralığı [0,1] | Piksel aralığı [0,255] | Görünürlük |
|:---:|:---:|:---:|:---|
| 1/255 | 0.004 | 1 | Gözle kesinlikle görülmez |
| 2/255 | 0.008 | 2 | Görülmez |
| 4/255 | 0.016 | 4 | Neredeyse görülmez |
| **8/255** | **0.031** | **8** | **Standart benchmark** |
| 16/255 | 0.063 | 16 | Dikkatli bakılırsa fark edilebilir |

> 📌 **Makale Bağlantısı:** Her iki makalede de $\varepsilon = 8/255$ ($L_\infty$) standart pertürbasyon bütçesi olarak kullanılır.

## Hedefli vs. Hedefsiz Saldırılar

| Tür | Amaç | Formülasyon |
|:---|:---|:---|
| **Hedefsiz** (untargeted) | Herhangi bir yanlış sınıf | $f_\theta(\mathbf{x}_{adv}) \neq y$ |
| **Hedefli** (targeted) | Belirli bir hedef sınıf | $f_\theta(\mathbf{x}_{adv}) = y_{hedef}$ |

> **Öğrendiklerimiz — Adversarial Temeller**
>
> ✅ Adversarial örnek = orijinal + algılanamaz pertürbasyon
> ✅ $L_\infty$ normu en yaygın kısıt: her pikseldeki max değişim ≤ ε
> ✅ ε = 8/255 standart benchmark değeri
> ✅ Hedefsiz: "herhangi bir yanlış" vs. hedefli: "belirli bir yanlış"

---

\newpage

# Tehdit Modelleri (Threat Models)

| Model | Saldırganın bilgisi | Güç | Gerçekçilik |
|:---|:---|:---:|:---:|
| **Beyaz kutu** (white-box) | Mimari + ağırlıklar + eğitim verisi | ★★★ | Düşük |
| **Gri kutu** (gray-box) | Mimari bilgisi, ağırlıklar yok | ★★ | Orta |
| **Siyah kutu** (black-box) | Sadece girdi → çıktı | ★ | Yüksek |

**Siyah kutu saldırı yöntemleri:**

1. **Sorgu tabanlı** (query-based): Modele binlerce sorgu atarak gradyan tahmin et
2. **Transfer tabanlı** (transfer-based): Başka bir model üzerinde üretilen adversarial örneği hedef modele uygula

> 📌 **Makale Bağlantısı:** Makale A hem beyaz kutu saldırı hem de transfer tabanlı siyah kutu analizi yapar. Transferability (aktarılabilirlik) makalenin temel katkılarından biridir.

---

\newpage

# FGSM — Fast Gradient Sign Method

## Goodfellow et al. (2014)

FGSM, ilk ve en basit gradyan tabanlı saldırıdır. Tek bir adımda adversarial örnek üretir.

## Matematiksel Türetme

**Başlangıç noktası:** Kaybı $\boldsymbol{\delta}$'ya göre maksimize etmek istiyoruz:

$$\max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} L(\theta, \mathbf{x} + \boldsymbol{\delta}, y)$$

**Adım 1:** Kayıp fonksiyonunun birinci derece Taylor açılımı:

$$L(\mathbf{x} + \boldsymbol{\delta}) \approx L(\mathbf{x}) + \nabla_{\mathbf{x}} L \cdot \boldsymbol{\delta}$$

**Adım 2:** $\nabla_{\mathbf{x}} L \cdot \boldsymbol{\delta}$'yı $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$ altında maksimize et.

$L_\infty$ kısıtı altında iç çarpımı (Parça 1) maksimize eden $\boldsymbol{\delta}$:

$$\boldsymbol{\delta}^* = \varepsilon \cdot \text{sign}(\nabla_{\mathbf{x}} L)$$

**Sonuç — FGSM formülü:**

$$\boxed{\mathbf{x}_{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_{\mathbf{x}} L(\theta, \mathbf{x}, y))}$$

## Adım Adım Süreç

```
  ① Orijinal görüntü x'i modele ver            (ileri yayılım)
  ② Kayıp L(θ, x, y) hesapla                    (cross-entropy)
  ③ Girdiye göre gradyan hesapla: ∇ₓL           (geri yayılım)
  ④ Gradyanın işaretini al: sign(∇ₓL)           (her piksel: +1 veya -1)
  ⑤ ε ile çarp ve ekle: x_adv = x + ε·sign(∇ₓL)
  ⑥ [0,1] aralığına kırp (clamp)

  x ──▶ [Model] ──▶ L ──▶ [Backprop] ──▶ ∇ₓL ──▶ sign(·) ──▶ ×ε ──▶ (+x) ──▶ x_adv
```

## PyTorch Uygulaması

```python
import torch
import torch.nn as nn

def fgsm_attack(model, x, y, epsilon):
    """FGSM adversarial saldırısı.

    Args:
        model: Hedef sınıflandırma modeli
        x: Orijinal görüntü tensörü (B, C, H, W)
        y: Gerçek etiketler (B,)
        epsilon: Pertürbasyon bütçesi (ör. 8/255)

    Returns:
        x_adv: Adversarial görüntü
    """
    x_adv = x.clone().detach().requires_grad_(True)

    # ① İleri yayılım
    output = model(x_adv)

    # ② Kayıp hesapla
    loss = nn.CrossEntropyLoss()(output, y)

    # ③ Geri yayılım — girdiye göre gradyan
    loss.backward()

    # ④⑤ Gradyan işareti × ε ve ekle
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = x + perturbation

    # ⑥ Geçerli piksel aralığına kırp
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

# Kullanım
# x_adv = fgsm_attack(model, images, labels, epsilon=8/255)
```

## FGSM'nin Güçlü ve Zayıf Yanları

| | Açıklama |
|:---|:---|
| ✅ Hızlı | Tek ileri + geri yayılım yeterli |
| ✅ Basit | Anlaşılması ve uygulaması kolay |
| ❌ Kaba | Tek adımda optimal pertürbasyon bulamayabilir |
| ❌ Zayıf | Güçlü savunmalara karşı başarısız olabilir |

> **Öğrendiklerimiz — FGSM**
>
> ✅ $\mathbf{x}_{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_{\mathbf{x}} L)$
> ✅ Tek adımlık, hızlı saldırı
> ✅ Gradyan işareti: her piksel +ε veya -ε değişir

---

\newpage

# PGD — Projected Gradient Descent

## Madry et al. (2018)

PGD, FGSM'nin **iteratif** versiyonudur. Küçük adımlarla ilerleyerek ve her adımda $\varepsilon$-topunun içine geri projeksiyon yaparak çok daha güçlü bir saldırı üretir.

## Formülasyon

$$\mathbf{x}^{(0)}_{adv} = \mathbf{x} + \text{Uniform}(-\varepsilon, \varepsilon)$$

$$\mathbf{x}^{(t+1)}_{adv} = \Pi_{\mathbf{x}+\mathcal{S}}\left(\mathbf{x}^{(t)}_{adv} + \alpha \cdot \text{sign}\left(\nabla_{\mathbf{x}} L(\theta, \mathbf{x}^{(t)}_{adv}, y)\right)\right)$$

burada:
- $\alpha$: adım boyutu (genellikle $\varepsilon/4$ veya $2/255$)
- $\Pi$: $\varepsilon$-topu içine projeksiyon operatörü
- $\mathcal{S} = \{\boldsymbol{\delta} : \|\boldsymbol{\delta}\|_\infty \leq \varepsilon\}$

```
  PGD adımları (2D kavramsal görselleştirme):

         ε-topu
        ╱──────╲
       ╱   ③    ╲
      │  ②  ↗   │         ① Rastgele başlangıç
      │ ↗       │         ② Gradyan yönünde adım
      │①        │         ③ Yine gradyan yönünde
       ╲    ④  ╱          ④ Topun dışına çıktı →
        ╲──↗──╱              Geri projeksiyon (Π)
            ↓
          ④'(projeksiyon sonrası: topun kenarında)
```

## PyTorch Uygulaması

```python
def pgd_attack(model, x, y, epsilon, alpha, num_steps):
    """PGD adversarial saldırısı.

    Args:
        model: Hedef model
        x: Orijinal görüntüler (B, C, H, W)
        y: Gerçek etiketler (B,)
        epsilon: Pertürbasyon bütçesi (ör. 8/255)
        alpha: Adım boyutu (ör. 2/255)
        num_steps: İterasyon sayısı (ör. 10-50)
    """
    # Rastgele başlangıç
    x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for step in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # İleri + kayıp
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)

        # Geri yayılım
        loss.backward()

        # Gradyan adımı
        x_adv = x_adv + alpha * x_adv.grad.sign()

        # ε-topu içine projeksiyon
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()

# Kullanım
# x_adv = pgd_attack(model, images, labels,
#                     epsilon=8/255, alpha=2/255, num_steps=20)
```

## PGD vs. FGSM Karşılaştırması

| Özellik | FGSM | PGD |
|:---|:---|:---|
| Adım sayısı | 1 | 10-50+ |
| Güç | Zayıf-orta | **Güçlü** |
| Hız | Çok hızlı | Yavaş |
| Yaklaşım | Tek büyük adım | Çok küçük adım + projeksiyon |
| Kullanım | Hızlı test, adversarial training | **Robustness değerlendirme standardı** |

> 📌 **Makale Bağlantısı:** Her iki hedef makale de PGD'yi temel saldırı yöntemi olarak kullanır. PGD, "en güçlü birinci derece saldırgan" (strongest first-order adversary) olarak kabul edilir.

> **Öğrendiklerimiz — PGD**
>
> ✅ FGSM'nin iteratif versiyonu: küçük adımlar + projeksiyon
> ✅ Her adımda ε-topunun içinde kalır
> ✅ Daha güçlü ama daha yavaş
> ✅ Robustness değerlendirmenin altın standardı

---

\newpage

# C&W — Carlini & Wagner Attack

## Carlini & Wagner (2017)

Farklı bir yaklaşım: pertürbasyonu **minimize** ederken yanlış sınıflandırmayı **garanti** et.

## Formülasyon

$$\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_2 + c \cdot g(\mathbf{x} + \boldsymbol{\delta})$$

burada $g(\cdot)$ sınıflandırma hedefini kodlar:

$$g(\mathbf{x}') = \max\left(Z(\mathbf{x}')_y - \max_{j \neq y} Z(\mathbf{x}')_j, \; -\kappa\right)$$

- $Z(\mathbf{x}')$: modelin logit çıktıları (softmax öncesi)
- $\kappa$: güven marjı (confidence margin)

**Değişken değiştirme hilesi:** $\boldsymbol{\delta} = \frac{1}{2}(\tanh(\mathbf{w}) + 1) - \mathbf{x}$ — bu, pertürbasyonun otomatik olarak $[0,1]$ aralığında kalmasını sağlar.

## FGSM/PGD ile Karşılaştırma

| Özellik | FGSM/PGD | C&W |
|:---|:---|:---|
| Yaklaşım | Kaybı maksimize et | Pertürbasyonu minimize et |
| Kısıt | Sabit ε bütçesi | Minimum gerekli pertürbasyon |
| Norm | Genellikle $L_\infty$ | Genellikle $L_2$ |
| Güç | PGD güçlü | **Daha küçük pertürbasyon** bulur |
| Hız | PGD: orta | C&W: yavaş (optimizasyon tabanlı) |

> **Öğrendiklerimiz — C&W**
>
> ✅ Optimizasyon tabanlı: minimum pertürbasyonla kandırma
> ✅ PGD'den daha küçük pertürbasyon bulabilir ama daha yavaş
> ✅ Savunma değerlendirmede önemli: "gerçekten ne kadar az pertürbasyon yeterli?"

---

# AutoAttack — Ensemble Değerlendirme Standardı

## Croce & Hein (2020)

AutoAttack, 4 farklı saldırıyı birleştiren **parametresiz** bir değerlendirme aracıdır:

```
  ┌─────────────────────────────────────────────┐
  │              AutoAttack                      │
  │                                              │
  │  ┌────────────┐  ┌────────────┐              │
  │  │  APGD-CE   │  │  APGD-DLR  │  Beyaz kutu │
  │  │ (adaptif   │  │ (farklı    │  gradyan     │
  │  │  PGD + CE) │  │  kayıp fn.)│  tabanlı     │
  │  └────────────┘  └────────────┘              │
  │                                              │
  │  ┌────────────┐  ┌────────────┐              │
  │  │    FAB      │  │  Square    │  Siyah kutu  │
  │  │ (sınır     │  │ (sorgu     │  (gradyan    │
  │  │  tabanlı)  │  │  tabanlı)  │  gerektirmez)│
  │  └────────────┘  └────────────┘              │
  │                                              │
  │  Sırayla çalışır: biri başarısızsa diğeri    │
  │  dener. Hiperparametre ayarı gerektirmez.    │
  └─────────────────────────────────────────────┘
```

**Neden önemli?** Birçok savunma yöntemi, belirli bir saldırıya karşı güçlü görünür ama aslında "gradient masking" (gradyan gizleme) yapıyordur. AutoAttack farklı saldırı türlerini birleştirerek bu tuzağı aşar.

> 📌 **Makale Bağlantısı:** Makale A, robustness değerlendirmede AutoAttack kullanır. Bu, sonuçların güvenilirliğini artırır.

---

\newpage

# Gradyan Karakteristikleri Analizi

## CNN ve ViT'te Gradyanlar Nasıl Farklı?

> 📌 **Makale Bağlantısı:** Bu bölüm, Makale A'nın **birinci temel katkısıdır**.

CNN ve ViT'in mimari farkları (Parça 3B), adversarial gradyanların karakteristiklerini doğrudan etkiler:

| Özellik | CNN Gradyanları | ViT Gradyanları |
|:---|:---|:---|
| **Uzamsal yapı** | Yerel, yapılandırılmış | Daha dağınık, global |
| **Korelasyon** | Komşu pikseller yüksek korelasyon | Daha düşük yerel korelasyon |
| **Büyüklük dağılımı** | Belirli bölgelerde yoğun | Daha uniform dağılmış |
| **Düzgünlük** | Daha pürüzlü (sharp) | Daha düzgün (smooth) |

```
  CNN gradyanı (kavramsal):         ViT gradyanı (kavramsal):

  ┌────────────────┐                ┌────────────────┐
  │ ░░░░░░░░░░░░░░ │                │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
  │ ░░░████░░░░░░░ │                │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
  │ ░░░████░░░░░░░ │                │ ▒▒▓▒▒▒▓▒▒▒▒▓▒▒ │
  │ ░░░░░░░░░████░ │                │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
  │ ░░░░░░░░░████░ │                │ ▒▓▒▒▒▒▒▒▒▓▒▒▒▒ │
  │ ░░░░░░░░░░░░░░ │                │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
  └────────────────┘                └────────────────┘
  Belirli bölgelerde yoğun          Daha homojen dağılım
  (yerellik önyargısı)              (küresel dikkat etkisi)
```

**Bu fark neden önemli?**

1. FGSM/PGD gibi gradyan tabanlı saldırıların etkinliği gradyan kalitesine bağlıdır
2. Farklı gradyan yapıları → farklı pertürbasyon kalıpları → farklı transferability

---

# Transferability (Aktarılabilirlik)

## Tanım

Model A için üretilen adversarial örnek ($\mathbf{x}_{adv}^A$), **Model B'yi de kandırabilir mi?**

$$f_A(\mathbf{x}_{adv}^A) \neq y \quad \text{VE} \quad f_B(\mathbf{x}_{adv}^A) \neq y \quad ?$$

## Transferability Matrisi

```
              Hedef Model
              ───────────────────────────────
              │ ResNet │ EffNet │  ViT  │ DeiT │
  ────────────┼────────┼────────┼───────┼──────┤
  Kaynak  Res │  100%  │  ~65%  │ ~35%  │ ~38% │
  Model   Eff │  ~60%  │  100%  │ ~30%  │ ~33% │
          ViT │  ~40%  │  ~38%  │ 100%  │ ~55% │
         DeiT │  ~42%  │  ~40%  │ ~50%  │ 100% │

  (Hipotetik değerler — genel kalıpları göstermek için)
```

## Genel Kalıplar

| Transfer Yönü | Başarı | Neden |
|:---|:---:|:---|
| CNN → CNN | **Yüksek** | Benzer yerellik önyargısı |
| ViT → ViT | **Orta-Yüksek** | Benzer küresel dikkat yapısı |
| CNN → ViT | **Düşük** | Farklı özellik işleme |
| ViT → CNN | **Düşük** | Farklı gradyan yapıları |

> 📌 **Makale Bağlantısı:** Makale A'nın **ikinci temel katkısı**: CNN-Transformer arası transferability'nin sistematik analizi. Aynı mimari ailesi içinde transfer yüksek, aileler arası düşüktür.

> **Öğrendiklerimiz — Gradyan Analizi & Transferability**
>
> ✅ CNN: yerel, yapılandırılmış gradyanlar; ViT: global, dağınık gradyanlar
> ✅ Transferability: aynı aile içinde yüksek, farklı aileler arası düşük
> ✅ Bu fark savunma stratejilerini doğrudan etkiler

---

# Parça 4A Kavram Haritası & Parça 4B'ye Köprü

```
╔═══════════════════════════════════════════════════════════════╗
║  Parça 1: Gradyan ──▶ Parça 2: Backprop ──▶ PARÇA 4A       ║
║                                                               ║
║  Adversarial Örnek: x_adv = x + δ, ||δ||∞ ≤ ε               ║
║       │                                                       ║
║       ├── FGSM: tek adım, sign(∇ₓL)                          ║
║       ├── PGD: çok adım + projeksiyon                         ║
║       ├── C&W: optimizasyon tabanlı, min pertürbasyon         ║
║       └── AutoAttack: ensemble, altın standart                ║
║                                                               ║
║  Gradyan Analizi: CNN vs ViT (Makale A Katkı 1)             ║
║  Transferability: Aileler arası düşük (Makale A Katkı 2)     ║
║                                                               ║
║  → PARÇA 4B: Savunmalar, Metrikler, Kod Atölyesi            ║
╚═══════════════════════════════════════════════════════════════╝
```

**Parça 4B'de:** Adversarial eğitim, giriş dönüşümleri, sertifikalı savunmalar, robustness metrikleri ve kapsamlı bir kod atölyesi göreceğiz.

---

*Devam: Parça 4B — Adversarial Savunmalar, Metrikler & Kod Atölyesi →*
