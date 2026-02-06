---
title: "Parça 3B: Dikkat Mekanizması, Transformer & Vision Transformer"
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
  - \fancyhead[L]{Parça 3B — Attention, Transformer \& ViT}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Dikkat Mekanizması (Attention Mechanism)

## Analoji: Kütüphanede Kitap Arama

Bir kütüphaneye girdiğini düşün. Elinde bir **soru** var (Query). Kütüphanedeki her kitabın bir **başlık/indeks** bilgisi var (Key). Sen sorunla en alakalı başlıkları buluyorsun, sonra o kitapların **içeriğini** (Value) okuyorsun.

```
  Sen (Sorgu/Query):  "Adversarial saldırılar nedir?"

  Kitaplık (Key-Value çiftleri):
  ┌──────────────────────────────────────────────────────┐
  │  Key (Başlık)              │  Value (İçerik)         │
  ├────────────────────────────┼─────────────────────────┤
  │  "Derin Öğrenme Temelleri" │  [temel bilgiler...]    │  Benzerlik: 0.3
  │  "Adversarial ML"          │  [saldırı yöntemleri...]│  Benzerlik: 0.9 ★
  │  "Doğal Dil İşleme"       │  [metin analizi...]     │  Benzerlik: 0.1
  │  "Robustness Analizi"      │  [savunma yöntemleri...]│  Benzerlik: 0.7
  └────────────────────────────┴─────────────────────────┘

  Dikkat (Attention) = Benzerlik ağırlıklı Value toplamı
  Sonuç ≈ 0.9 × [saldırı yöntemleri] + 0.7 × [savunma yöntemleri] + ...
```

## Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Her adımı açıklayalım:

```
  Adım 1:  QKᵀ         → Benzerlik skorları (hangi key'ler query'ye yakın?)
  Adım 2:  / √dₖ       → Ölçekleme (değerler çok büyümesin)
  Adım 3:  softmax(·)   → Olasılık dağılımına dönüştür (Parça 1B'deki softmax)
  Adım 4:  × V          → Ağırlıklı değer toplamı
```

### Sayısal Örnek

$d_k = 2$ boyutunda basit bir örnek:

```python
import numpy as np

# Q, K, V matrisleri (3 pozisyon, 2 boyut)
Q = np.array([[1.0, 0.0],    # Query 1
              [0.0, 1.0],    # Query 2
              [1.0, 1.0]])   # Query 3

K = np.array([[1.0, 0.0],    # Key 1
              [0.0, 1.0],    # Key 2
              [0.5, 0.5]])   # Key 3

V = np.array([[10, 0],       # Value 1
              [0, 10],       # Value 2
              [5, 5]])       # Value 3

d_k = K.shape[-1]  # 2

# Adım 1-2: Ölçekli benzerlik
scores = Q @ K.T / np.sqrt(d_k)
print("Skorlar:\n", scores.round(2))
# [[0.71, 0.00, 0.35],    Q1, K1'e en benzer
#  [0.00, 0.71, 0.35],    Q2, K2'ye en benzer
#  [0.50, 0.50, 0.71]]    Q3, K3'e en benzer

# Adım 3: Softmax
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)
print("Ağırlıklar:\n", weights.round(3))
# [[0.44, 0.22, 0.34],    Q1: V1'e en çok dikkat
#  [0.22, 0.44, 0.34],    Q2: V2'ye en çok dikkat
#  [0.30, 0.30, 0.40]]    Q3: dengeli dikkat

# Adım 4: Ağırlıklı toplam
output = weights @ V
print("Çıktı:\n", output.round(2))
# [[5.7, 3.9],    Q1: V1 ağırlıklı
#  [3.9, 5.7],    Q2: V2 ağırlıklı
#  [4.5, 4.5]]    Q3: dengeli karışım
```

### Neden $\sqrt{d_k}$ ile Ölçekleme?

$d_k$ büyüdüğünde $QK^T$ değerleri çok büyür. Softmax'ta büyük değerler → neredeyse one-hot dağılım → gradyanlar çok küçülür.

$\sqrt{d_k}$ ile bölmek, değerleri makul aralıkta tutar.

> **Öğrendiklerimiz — Dikkat Mekanizması**
>
> ✅ Query = ne arıyorum, Key = indeks bilgisi, Value = içerik
> ✅ Dikkat = benzerlik ağırlıklı değer toplamı
> ✅ $\sqrt{d_k}$ ölçekleme: sayısal kararlılık için

---

## Çok Başlı Dikkat (Multi-Head Attention)

Tek bir dikkat başlığı sadece bir "bakış açısı" sunar. Çok başlı dikkat, **farklı perspektiflerden** eş zamanlı dikkat sağlar:

```
  Başlık 1: "Renklere dikkat et"
  Başlık 2: "Şekillere dikkat et"
  Başlık 3: "Dokuya dikkat et"
  Başlık 4: "Konumsal ilişkilere dikkat et"
  ...

  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │  Head 1  │   │  Head 2  │   │  Head 3  │   │  Head h  │
  │ Att(Q,K,V│   │ Att(Q,K,V│   │ Att(Q,K,V│   │ Att(Q,K,V│
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                           │
                     [Birleştir (Concat)]
                           │
                     [Lineer projeksiyon Wᴼ]
                           │
                        Çıktı
```

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Her başlık kendi $W^Q_i$, $W^K_i$, $W^V_i$ projeksiyonlarını kullanır:

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

> 📌 **Makale Bağlantısı:** ViT'lerde dikkat başlıklarının sayısı ve boyutu modelin ifade gücünü belirler. Parça 5B'de bu dikkat ağırlıklarını **açıklanabilirlik** (explainability) aracı olarak kullanacağız.

---

## Öz-Dikkat (Self-Attention)

Öz-dikkat'te Q, K ve V aynı girdiden türetilir — her pozisyon diğer **tüm** pozisyonlara dikkat eder:

```
  Girdi:  [Yama₁, Yama₂, Yama₃, Yama₄]

         Yama₁  Yama₂  Yama₃  Yama₄
  Yama₁  [0.5    0.2    0.1    0.2 ]   Her yama tüm
  Yama₂  [0.1    0.4    0.3    0.2 ]   yamalara
  Yama₃  [0.1    0.3    0.4    0.2 ]   dikkat eder
  Yama₄  [0.2    0.2    0.2    0.4 ]

  Bu KÜRESEL (global) bir işlemdir!
  CNN'den farklı olarak yerellik kısıtı YOKTUR.
```

> **Öğrendiklerimiz — Multi-Head & Self-Attention**
>
> ✅ Multi-Head: farklı perspektiflerden paralel dikkat
> ✅ Self-Attention: Q=K=V aynı girdiden, her pozisyon herkese bakabilir
> ✅ CNN'den farkı: küresel bağlam, ilk katmandan itibaren

---

\newpage

# Transformer Mimarisi

## "Attention Is All You Need" (Vaswani et al., 2017)

Orijinal Transformer doğal dil işleme (NLP) için tasarlandı ama temel mimarisi görüye (vision) de uyarlandı.

## Transformer Encoder Bloğu

```
  ┌──────────────────────────────────────┐
  │          Transformer Encoder         │
  │                                      │
  │  Girdi (x)                           │
  │     │                                │
  │     ▼                                │
  │  ┌─────────────────────────┐         │
  │  │  Multi-Head Self-Attention │       │
  │  └────────────┬────────────┘         │
  │               │                      │
  │          ┌────▼────┐                 │
  │          │ Add & LN │◄─── x (skip)  │
  │          └────┬─────┘                │
  │               │                      │
  │  ┌────────────▼────────────┐         │
  │  │  Feed-Forward Network   │         │
  │  │  (2 lineer katman + GELU)│        │
  │  └────────────┬────────────┘         │
  │               │                      │
  │          ┌────▼────┐                 │
  │          │ Add & LN │◄─── (skip)    │
  │          └────┬─────┘                │
  │               │                      │
  │            Çıktı                     │
  └──────────────────────────────────────┘

  LN = Layer Normalization
  Add = Artık bağlantı (ResNet'teki skip connection gibi, Parça 3A)
```

## Konumsal Kodlama (Positional Encoding)

Dikkat mekanizması sıra bilgisi taşımaz (permutation invariant). Konumsal kodlama, her pozisyona bir "adres" ekler:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```
  Pozisyon 0: [sin(0), cos(0), sin(0), cos(0), ...]
  Pozisyon 1: [sin(1), cos(1), sin(0.01), cos(0.01), ...]
  Pozisyon 2: [sin(2), cos(2), sin(0.02), cos(0.02), ...]
  ...

  Her pozisyon benzersiz bir "parmak izi" alır.
  Girdi = Token Embedding + Positional Encoding
```

> **Öğrendiklerimiz — Transformer**
>
> ✅ Encoder bloğu: Self-Attention → Add&Norm → FFN → Add&Norm
> ✅ Positional Encoding: sinüzoidal fonksiyonlarla konum bilgisi
> ✅ GELU aktivasyonu (Parça 2A'daki aktivasyon karşılaştırması)
> ✅ Artık bağlantılar (ResNet'tekine benzer, Parça 3A)

---

\newpage

# Vision Transformer (ViT)

## Temel Fikir: Görüntüyü "Kelimeler"e Böl

**"An Image is Worth 16×16 Words"** — Dosovitskiy et al. (2020)

NLP'de cümleler **kelimeler**e (token) ayrılır. ViT'de görüntüler **yamalara** (patches) ayrılır:

```
  Orijinal Görüntü (224×224):          Yamalar (14×14 = 196 adet yama):

  ┌──────────────────────────┐         ┌──┬──┬──┬──┬──┬──┬──┐
  │                          │         │P₁│P₂│P₃│P₄│P₅│P₆│..│
  │                          │         ├──┼──┼──┼──┼──┼──┼──┤
  │                          │    ──▶  │P₈│P₉│..│  │  │  │  │
  │       224 × 224          │         ├──┼──┼──┼──┼──┼──┼──┤
  │                          │         │  │  │  │  │  │  │  │
  │                          │         ├──┼──┼──┼──┼──┼──┼──┤
  │                          │         │  │  │  │  │  │  │  │
  └──────────────────────────┘         └──┴──┴──┴──┴──┴──┴──┘

  Her yama: 16×16 piksel               196 adet "kelime" (token)
  3 kanal → 16×16×3 = 768 boyut
```

## ViT Mimarisi — Tam Akış

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                       Vision Transformer (ViT)                  │
  │                                                                 │
  │  Görüntü (224×224×3)                                            │
  │       │                                                         │
  │       ▼                                                         │
  │  ┌──────────────────┐                                           │
  │  │ Yama Bölme       │  224/16 = 14 → 14×14 = 196 yama          │
  │  │ (Patch Split)    │  Her yama: 16×16×3 = 768 piksel           │
  │  └────────┬─────────┘                                           │
  │           │                                                     │
  │           ▼                                                     │
  │  ┌──────────────────┐                                           │
  │  │ Lineer Projeksiyon│  768 → D boyutlu gömme (embedding)       │
  │  │ (Patch Embedding) │  Parametre: E ∈ ℝ^(768×D)               │
  │  └────────┬─────────┘                                           │
  │           │                                                     │
  │    [CLS] ─┤  ← Sınıflandırma tokeni (öğrenilebilir)            │
  │           │                                                     │
  │    (+) ───┤  ← Konumsal kodlama (positional embedding)          │
  │           │                                                     │
  │           ▼                                                     │
  │  ┌──────────────────┐                                           │
  │  │ Transformer       │  × L katman (ör. L=12 for ViT-Base)     │
  │  │ Encoder Bloğu     │  Her blok: MHSA → Add&LN → FFN → Add&LN│
  │  │ × L               │                                          │
  │  └────────┬─────────┘                                           │
  │           │                                                     │
  │     [CLS] token çıktısı                                         │
  │           │                                                     │
  │           ▼                                                     │
  │  ┌──────────────────┐                                           │
  │  │ MLP Başlık        │  [CLS] → Sınıf olasılıkları              │
  │  │ (Classification   │  D → num_classes                          │
  │  │  Head)            │                                           │
  │  └──────────────────┘                                           │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

## [CLS] Token

Girdi dizisinin başına eklenen özel bir öğrenilebilir vektör. Transformer katmanlarından geçtikten sonra, tüm yamaların bilgisini toplamış olur. Sınıflandırma bu token'ın son çıktısı üzerinden yapılır.

## ViT Varyantları

| Model | Katman (L) | Gizli Boyut (D) | Başlık Sayısı | Parametre |
|:---|:---:|:---:|:---:|:---:|
| ViT-Tiny | 12 | 192 | 3 | 5.7M |
| ViT-Small | 12 | 384 | 6 | 22M |
| **ViT-Base** | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

## Veri İhtiyacı ve Ön Eğitim

> ⚠️ **Dikkat:** ViT, CNN'lere kıyasla çok daha fazla veri ister. Bunun nedeni CNN'lerin sahip olduğu **tümevarımsal önyargıların** (yerellik, öteleme eşdeğişkenliği) ViT'de olmamasıdır. ViT bu kalıpları veriden öğrenmek zorundadır.
>
> - Küçük veri setlerinde (CIFAR-10 gibi): CNN genellikle daha iyi
> - Büyük veri setlerinde (ImageNet-21K, JFT-300M): ViT CNN'leri geçer

## Kod: ViT ile timm Kütüphanesi

```python
import timm
import torch

# Önceden eğitilmiş ViT-Base yükle
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Rastgele girdi
x = torch.randn(1, 3, 224, 224)

# İleri hesaplama
with torch.no_grad():
    output = model(x)
print(f"Çıktı boyutu: {output.shape}")  # [1, 1000] (ImageNet sınıfları)

# Model yapısını incele
print(f"Yama boyutu: {model.patch_embed.patch_size}")  # (16, 16)
print(f"Gömme boyutu: {model.embed_dim}")  # 768
print(f"Encoder katman sayısı: {len(model.blocks)}")  # 12
print(f"Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")
```

> 📌 **Makale Bağlantısı:** Makale B, **FE-ViT-ECL** mimarisinin omurgası olarak ViT kullanır. Makale A ise ViT-B/16 ve DeiT gibi modelleri adversarial saldırılara karşı test eder.

> **Öğrendiklerimiz — ViT**
>
> ✅ Görüntü → 16×16 yamalar → lineer projeksiyon → Transformer encoder
> ✅ [CLS] token: sınıflandırma için bilgi toplar
> ✅ CNN'den farklı: yerellik önyargısı yok, küresel dikkat
> ✅ Büyük veri ile CNN'leri geçer, küçük veri ile geride kalabilir

---

\newpage

# CNN vs. Transformer Karşılaştırması

## Tümevarımsal Önyargı (Inductive Bias)

| Özellik | CNN | ViT |
|:---|:---|:---|
| **Yerellik** | ✓ Evrişim küçük pencere kullanır | ✗ İlk katmandan küresel dikkat |
| **Öteleme eşdeğişkenliği** | ✓ Aynı kernel her yerde paylaşılır | ✗ Pozisyon gömmeleriyle öğrenilir |
| **Hiyerarşik yapı** | ✓ Katmanlarla giderek büyüyen RF | ✗ Tüm katmanlarda aynı çözünürlük |

```
  CNN'in "görüş alanı" (Receptive Field):

  Katman 1:     Katman 3:       Katman 5:
  ┌─┐           ┌───────┐       ┌─────────────┐
  │●│           │  ●    │       │      ●      │
  └─┘           │       │       │             │
  3×3           │       │       │             │
  (yerel)       └───────┘       └─────────────┘
                ~11×11          ~tüm görüntü
                (büyüyen)       (çok derin sonra)

  ViT'in "görüş alanı":

  Katman 1:                 (HER katmanda aynı)
  ┌─────────────────────┐
  │  ●  ←→  her yamaya  │
  │     dikkat eder     │
  │     (küresel)       │
  └─────────────────────┘
```

## Karşılaştırma Tablosu

| Kriter | CNN | ViT |
|:---|:---|:---|
| Yerel özellikler | ✓ Güçlü (kernel tasarımı) | △ Öğrenmeli |
| Küresel bağlam | △ Derin katmanlarda | ✓ İlk katmandan |
| Hesaplama (küçük görüntü) | ✓ Verimli | △ Daha maliyetli |
| Hesaplama (büyük görüntü) | ✓ Doğrusal artış | ✗ Karesel artış ($O(n^2)$) |
| Küçük veri performansı | ✓ Daha iyi | ✗ Daha kötü |
| Büyük veri performansı | △ Doyar | ✓ Ölçeklenir |
| Açıklanabilirlik | Grad-CAM (Parça 5) | Dikkat haritaları (Parça 5) |
| Adversarial robustness | ? (Parça 4) | ? (Parça 4) |

> 📌 **Makale Bağlantısı:** Makale A'nın **temel araştırma sorusu** tam olarak bu tablonun son satırıdır: "CNN ve ViT adversarial saldırılara nasıl farklı tepki verir?" Parça 4'te bu soruyu derinlemesine inceleyeceğiz.

---

## Kod Atölyesi: CNN vs. ViT Özellik Görselleştirme

```python
import torch
import torchvision.models as models
import timm

# ─── CNN: ResNet-50 özellik haritaları ───
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Hook ile ara katman çıktısını yakala
features = {}
def hook_fn(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

resnet.layer1.register_forward_hook(hook_fn('layer1'))
resnet.layer4.register_forward_hook(hook_fn('layer4'))

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    _ = resnet(x)

print(f"Layer1 özellik haritası: {features['layer1'].shape}")  # [1, 256, 56, 56]
print(f"Layer4 özellik haritası: {features['layer4'].shape}")  # [1, 2048, 7, 7]

# ─── ViT: Dikkat ağırlıkları ───
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.eval()

# Dikkat ağırlıklarını yakala
attn_weights = []
for blk in vit.blocks:
    blk.attn.fused_attn = False  # Dikkat ağırlıklarını kaydet
    original_forward = blk.attn.forward
    # timm modelinde dikkat ağırlıkları doğrudan erişilebilir

with torch.no_grad():
    output = vit(x)

print("ViT çıktı:", output.shape)  # [1, 1000]

# Görselleştirme için Matplotlib kullanılır
# (Parça 5B'de detaylı Grad-CAM ve Attention Map görselleştirmesi)
```

---

# Parça 3 — Genel Kavram Haritası

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                   PARÇA 3 — TAM KAVRAM HARİTASI                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  PARÇA 1-2'DEN: Tensörler, Matris çarpımı, Backprop, PyTorch            ║
║       │                                                                  ║
║       ▼                                                                  ║
║  Dijital Görüntü ──▶ (H×W×C) Tensör ──▶ Normalizasyon                  ║
║       │                                                                  ║
║       ├──────────────────────┐                                           ║
║       ▼                      ▼                                           ║
║  CNN YOLU                  TRANSFORMER YOLU                              ║
║  ═════════                 ═══════════════                                ║
║  Evrişim (Convolution)     Dikkat (Attention)                            ║
║       │                        │                                         ║
║  Özellik Haritaları        Q, K, V                                       ║
║       │                    Scaled Dot-Product                             ║
║  Pooling                   Multi-Head                                    ║
║       │                        │                                         ║
║  CNN Mimarileri:           Positional Encoding                           ║
║  LeNet→AlexNet→VGG             │                                         ║
║  →ResNet→EfficientNet      Transformer Encoder                           ║
║       │                        │                                         ║
║       │                    Vision Transformer                            ║
║       │                    (Yama→Token→Encoder)                          ║
║       │                        │                                         ║
║       ├────────────────────────┤                                         ║
║       ▼                        ▼                                         ║
║  ┌─────────────────────────────────────┐                                 ║
║  │    CNN vs. ViT Karşılaştırması      │                                 ║
║  │  Yerellik vs. Küresel Bağlam       │                                 ║
║  │  Makale A'nın temel sorusu!        │                                 ║
║  └─────────────────┬───────────────────┘                                 ║
║                    │                                                     ║
║           ┌────────┼─────────┐                                           ║
║           ▼        ▼         ▼                                           ║
║        PARÇA 4   PARÇA 5   PARÇA 6                                      ║
║       Adversarial Frekans  Entegrasyon                                   ║
║        Saldırılar  & XAI   & Makaleler                                  ║
║                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

# Sonraki Parçaya Köprü — Parça 4'e Hazırlık

Bu parçada CNN ve Transformer/ViT mimarilerini derinlemesine öğrendik. Bunlar güçlü modeller — ResNet ImageNet'te %76+, ViT %85+ doğruluk elde edebilir.

Ama bu güçlü modellerin **ciddi bir zayıflığı** var: **adversarial örnekler**.

İnsan gözüyle fark edilemeyen minik pertürbasyonlar, bu modelleri yüksek güvenle **yanlış sınıflandırmaya** zorlayabilir. Ve bu pertürbasyonlar rastgele değil — Parça 1'de öğrendiğimiz **gradyan** ve Parça 2'de öğrendiğimiz **backpropagation** kullanılarak hesaplanır.

**Parça 4'te:**

1. Adversarial örneklerin ne olduğunu matematiksel olarak formalize edeceğiz
2. 4 temel saldırı yöntemini (FGSM, PGD, C&W, AutoAttack) öğreneceğiz
3. CNN ve ViT'in adversarial saldırılara **neden farklı tepki verdiğini** anlayacağız
4. Savunma mekanizmalarını inceleyeceğiz

Bu, **her iki hedef makalenin doğrudan konusu** olan parçadır.

---

*Devam: Parça 4A — Adversarial Makine Öğrenmesi: Temeller & Saldırı Yöntemleri →*
