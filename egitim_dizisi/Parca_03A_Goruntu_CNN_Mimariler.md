---
title: "Parça 3A: Bilgisayarlı Görü — Görüntü Temsili, Evrişim & CNN Mimarileri"
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
  - \fancyhead[L]{Parça 3A — Görüntü, Evrişim \& CNN}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Giriş ve Motivasyon

## Bilgisayar Bir Görüntüyü Nasıl "Görür"?

Biz insanlar bir fotoğrafa baktığımızda anında "kedi", "araba", "yüz" gibi kavramları algılarız. Peki bir bilgisayar ne görür? **Sadece sayılar.**

```
  İnsan gözüyle:          Bilgisayar gözüyle:

  ┌───────────┐           ┌──────────────────────┐
  │           │           │ 142  138  135  130 ...│
  │   🐱      │    ──▶    │ 145  140  137  132 ...│
  │  (kedi!)  │           │ 150  148  143  139 ...│
  │           │           │ 158  155  150  145 ...│
  └───────────┘           └──────────────────────┘
  Anlam algılar           Piksel değerleri (sayılar) görür
```

Parça 2'de MLP ile MNIST rakamlarını sınıflandırdık. Ama MLP'nin önemli bir sınırlaması vardı: her pikseli bağımsız bir girdi olarak alıyordu. Bir görüntüde **komşuluk ilişkileri** kritik bilgi taşır — kenarlar, dokular, şekiller hep yerel piksel kalıplarıdır.

> 📌 **Makale Bağlantısı:** Her iki hedef makale de görüntü sınıflandırma modelleri üzerine çalışır. Makale A, CNN ve ViT'yi karşılaştırır. Makale B, ViT omurgası üzerine inşa edilmiştir. Bu parçada her iki mimari ailesini öğreneceğiz.

---

\newpage

# Dijital Görüntü Temsili

## Gri Tonlamalı Görüntüler

Bir gri tonlamalı görüntü, 2 boyutlu bir **matristir** (Parça 1). Her eleman 0–255 arası bir piksel değeridir:

```
  0 = siyah                    255 = beyaz

  ┌─────────────────────┐
  │   0   50  100  150  │     0 ██████   (koyu)
  │  50  100  150  200  │    128 ▓▓▓▓▓▓  (orta)
  │ 100  150  200  255  │    255 ░░░░░░  (açık)
  │ 150  200  255  255  │
  └─────────────────────┘
  Boyut: Yükseklik (H) × Genişlik (W)
```

## Renkli Görüntüler (RGB)

Renkli görüntüler **3 kanallı** tensörlerdir (Parça 1'deki tensör kavramı):

```
  Kırmızı (R)        Yeşil (G)         Mavi (B)
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ R  R  R  │      │ G  G  G  │      │ B  B  B  │
  │ R  R  R  │  +   │ G  G  G  │  +   │ B  B  B  │  = Renkli Görüntü
  │ R  R  R  │      │ G  G  G  │      │ B  B  B  │
  └──────────┘      └──────────┘      └──────────┘

  Tensör boyutu: (3, H, W)  veya  (H, W, 3)
                  ↑                      ↑
              PyTorch sırası       NumPy/PIL sırası
```

## Normalizasyon

Model eğitiminde piksel değerleri genellikle normalize edilir:

| Yöntem | Dönüşüm | Aralık |
|:---|:---|:---:|
| [0,1] ölçekleme | piksel / 255.0 | $[0, 1]$ |
| Standardizasyon | (piksel - μ) / σ | $\approx[-3, 3]$ |
| ImageNet normalizasyonu | μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] | Standart |

## Yaygın Veri Setleri

| Veri Seti | Boyut | Sınıf | Eğitim | Test | Kanal |
|:---|:---:|:---:|:---:|:---:|:---:|
| MNIST | 28×28 | 10 | 60K | 10K | Gri |
| CIFAR-10 | 32×32 | 10 | 50K | 10K | RGB |
| CIFAR-100 | 32×32 | 100 | 50K | 10K | RGB |
| ImageNet | 224×224 | 1000 | 1.28M | 50K | RGB |

> 📌 **Makale Bağlantısı:** Makale A, ImageNet veya CIFAR ölçekli veri setleri kullanır. Makale B, medikal görüntü veri setleri (Parça 5B'de detay) kullanır.

## Kod: Görüntü Yükleme

```python
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# CIFAR-10 yükleme
transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W,C) uint8 → (C,H,W) float [0,1]
])
dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
sinif_isimleri = ['uçak','araba','kuş','kedi','geyik',
                  'köpek','kurbağa','at','gemi','kamyon']

# Bir görüntü incele
goruntu, etiket = dataset[0]
print(f"Tensör boyutu: {goruntu.shape}")   # torch.Size([3, 32, 32])
print(f"Değer aralığı: [{goruntu.min():.2f}, {goruntu.max():.2f}]")
print(f"Sınıf: {sinif_isimleri[etiket]}")

# Görselleştir
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    img, lbl = dataset[i]
    ax = axes[i//8, i%8]
    ax.imshow(img.permute(1,2,0))  # (C,H,W) → (H,W,C)
    ax.set_title(sinif_isimleri[lbl], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('cifar10_ornekler.png', dpi=150)
plt.show()
```

> **Öğrendiklerimiz — Görüntü Temsili**
>
> ✅ Gri görüntü = 2D matris, renkli görüntü = 3D tensör (C×H×W)
> ✅ Piksel değerleri 0–255, eğitimde [0,1] veya standart normalize edilir
> ✅ ToTensor(): otomatik [0,1] dönüşümü + kanal sırası değişimi

---

\newpage

# Evrişim (Convolution) İşlemi

## Analoji: Büyüteçle Resmi Taramak

MLP tüm pikselleri bir anda alıp tek bir büyük matris çarpımı yapar. Bu, tüm resme bir defada bakmak gibidir — verimsiz ve yerel kalıpları yakalamada zayıf.

**Evrişim** ise bir büyüteçle resmi sistematik olarak taramak gibidir: küçük bir pencere (çekirdek / kernel) görüntü üzerinde kayarak her bölgedeki yerel kalıbı yakalar.

```
  MLP yaklaşımı:                 CNN yaklaşımı:

  ┌────────────────┐            ┌────────────────┐
  │ Tüm pikselleri │            │  🔍 Küçük      │
  │ tek seferde    │            │  pencereyle    │
  │ düzleştir ve   │            │  tara, yerel   │
  │ matris çarp    │            │  kalıpları     │
  │                │            │  yakala        │
  └────────────────┘            └────────────────┘
  784 girdi → büyük W           3×3 kernel → paylaşılan W
```

## 2D Evrişim — Adım Adım

Bir $3 \times 3$ **çekirdek** (kernel/filter) görüntü üzerinde kayar:

$$\text{Çıktı}(i,j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Girdi}(i+m,\; j+n) \cdot \text{Kernel}(m,n)$$

### Sayısal Örnek

```
  Girdi (5×5):                    Kernel (3×3):
  ┌───┬───┬───┬───┬───┐          ┌───┬───┬───┐
  │ 1 │ 0 │ 1 │ 0 │ 1 │          │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┤          ├───┼───┼───┤
  │ 0 │ 1 │ 0 │ 1 │ 0 │          │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┤          ├───┼───┼───┤
  │ 1 │ 0 │ 1 │ 0 │ 1 │          │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┤          └───┴───┴───┘
  │ 0 │ 1 │ 0 │ 1 │ 0 │          (dikey kenar algılayıcı)
  ├───┼───┼───┼───┼───┤
  │ 1 │ 0 │ 1 │ 0 │ 1 │
  └───┴───┴───┴───┴───┘

  Adım 1 (sol üst köşe):
  ┌───┬───┬───┐
  │ 1 │ 0 │ 1 │  × Kernel = 1×1 + 0×0 + 1×(-1)
  │ 0 │ 1 │ 0 │             + 0×1 + 1×0 + 0×(-1)
  │ 1 │ 0 │ 1 │             + 1×1 + 0×0 + 1×(-1) = 0
  └───┴───┴───┘

  Çıktı (3×3):
  ┌───┬───┬───┐
  │ 0 │ 0 │ 0 │
  ├───┼───┼───┤
  │ 0 │ 0 │ 0 │    (Bu girdi simetriktir, bu yüzden
  ├───┼───┼───┤     dikey kenar algılayıcı sıfır verir)
  │ 0 │ 0 │ 0 │
  └───┴───┴───┘
```

### Stride ve Padding

```
  Stride = 1 (varsayılan):       Stride = 2 (atlayarak):

  ┌─●─●─●─●─●─┐                 ┌─●─·─●─·─●─┐
  │ ●─●─●─●─● │                 │ ·─·─·─·─· │
  │ ●─●─●─●─● │                 │ ●─·─●─·─● │
  │ ●─●─●─●─● │                 │ ·─·─·─·─· │
  └─●─●─●─●─●─┘                 └─●─·─●─·─●─┘
  Her pozisyonda hesap            Her 2 pozisyonda hesap
  → daha büyük çıktı              → daha küçük çıktı

  Padding = 0:                   Padding = 1 (sıfırla çevrele):
  5×5 girdi, 3×3 kernel           ┌─0─0─0─0─0─0─0─┐
  → 3×3 çıktı (küçüldü)          │ 0 ┌─────────┐0 │
                                   │ 0 │  5×5    │0 │
  Padding = 1:                    │ 0 │  girdi  │0 │
  → 5×5 çıktı (boyut korundu)    │ 0 └─────────┘0 │
                                   └─0─0─0─0─0─0─0─┘
```

**Çıktı boyutu formülü:**

$$O = \frac{W - K + 2P}{S} + 1$$

| Sembol | Anlam | Örnek |
|:---:|:---|:---:|
| $W$ | Girdi boyutu | 32 |
| $K$ | Kernel boyutu | 3 |
| $P$ | Padding | 1 |
| $S$ | Stride | 1 |
| $O$ | Çıktı boyutu | $\frac{32-3+2}{1}+1 = 32$ |

## Özellik Haritaları ve Özellik Hiyerarşisi

Her kernel farklı bir **özellik haritası** (feature map) üretir. Birden fazla kernel = birden fazla özellik haritası:

```
                         Kernel 1 ──▶ Özellik Haritası 1 (dikey kenarlar)
  Girdi Görüntü ──────── Kernel 2 ──▶ Özellik Haritası 2 (yatay kenarlar)
  (3 × 32 × 32)         Kernel 3 ──▶ Özellik Haritası 3 (çapraz kenarlar)
                         ...     ──▶ ...
                         Kernel N ──▶ Özellik Haritası N

  Çıktı: (N × 30 × 30)  → N = filtre sayısı
```

**Katman derinliğiyle özellik hiyerarşisi:**

```
  Erken katmanlar        Orta katmanlar         Derin katmanlar
  (Katman 1-2)          (Katman 3-5)           (Katman 6+)

  ┌─────────┐           ┌─────────┐            ┌─────────┐
  │ ── │ │  │           │  ◡  ◠   │            │  🐱 🐕  │
  │ ╱  ╲ ·  │           │  ◇  □   │            │  🚗 ✈   │
  │ ╲  ╱ ○  │           │  ◎  ⊞   │            │  🏠 🌳  │
  └─────────┘           └─────────┘            └─────────┘
  Kenarlar, çizgiler    Dokular, parçalar      Nesneler, kavramlar
  (düşük seviye)        (orta seviye)          (yüksek seviye)
```

## Havuzlama (Pooling)

Özellik haritalarının boyutunu azaltır, hesaplama maliyetini düşürür:

```
  Max Pooling (2×2, stride=2):

  Girdi (4×4):              Çıktı (2×2):
  ┌───┬───┬───┬───┐        ┌───┬───┐
  │ 1 │ 3 │ 2 │ 4 │        │ 6 │ 8 │  max(1,3,5,6)=6
  ├───┼───┼───┼───┤   ──▶  ├───┼───┤  max(2,4,7,8)=8
  │ 5 │ 6 │ 7 │ 8 │        │ 3 │ 4 │  max(2,1,3,2)=3
  ├───┼───┼───┼───┤        └───┴───┘  max(1,3,4,1)=4
  │ 2 │ 1 │ 1 │ 3 │
  ├───┼───┼───┼───┤
  │ 3 │ 2 │ 4 │ 1 │
  └───┴───┴───┴───┘
```

## Kod: Evrişim ile PyTorch

```python
import torch
import torch.nn as nn

# Tek bir evrişim katmanı
conv = nn.Conv2d(in_channels=3,   # RGB girdi
                 out_channels=16,  # 16 filtre
                 kernel_size=3,    # 3×3 kernel
                 stride=1,
                 padding=1)        # Boyut koruma

# Rastgele girdi (1 görüntü, 3 kanal, 32×32)
x = torch.randn(1, 3, 32, 32)
out = conv(x)
print(f"Girdi:  {x.shape}")     # [1, 3, 32, 32]
print(f"Çıktı:  {out.shape}")   # [1, 16, 32, 32]
print(f"Kernel: {conv.weight.shape}")  # [16, 3, 3, 3]
# 16 filtre × 3 girdi kanalı × 3×3 kernel boyutu

# Max pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)
out_pooled = pool(out)
print(f"Pooling sonrası: {out_pooled.shape}")  # [1, 16, 16, 16]
```

> **Öğrendiklerimiz — Evrişim**
>
> ✅ Evrişim = küçük kernel'ı görüntü üzerinde kaydırarak yerel kalıp yakalama
> ✅ Stride: adım büyüklüğü, padding: kenarlara sıfır ekleme
> ✅ Özellik hiyerarşisi: kenarlar → dokular → nesneler
> ✅ Pooling: boyut azaltma, öteleme değişmezliği

---

\newpage

# CNN Mimarileri — Tarihsel Gelişim

## LeNet-5 (1998) — Başlangıç

Yann LeCun'un el yazısı rakam tanıma için tasarladığı ilk başarılı CNN:

```
  Girdi      Conv1    Pool1    Conv2    Pool2     FC1    FC2   Çıktı
  32×32  →  28×28  → 14×14 → 10×10 →  5×5   → 120  → 84  →  10
  (1 ch)   (6 ch)   (6 ch)  (16 ch) (16 ch)

  ┌──────┐  ┌────┐  ┌──┐  ┌────┐  ┌─┐  ┌───┐ ┌──┐ ┌──┐
  │      │→ │    │→ │  │→ │    │→ │ │→ │   │→│  │→│10│
  │32×32 │  │28² │  │14²│  │10² │  │5²│  │120│ │84│ │  │
  └──────┘  └────┘  └──┘  └────┘  └─┘  └───┘ └──┘ └──┘
```

**Tarihsel önemi:** Evrişimin görüntü tanımada çalıştığını kanıtladı.

## AlexNet (2012) — Derin Öğrenme Devrimi

ImageNet yarışmasını (ILSVRC 2012) ezici farkla kazandı ve derin öğrenme çağını başlattı.

| Yenilik | Açıklama |
|:---|:---|
| **ReLU** | Sigmoid yerine — daha hızlı eğitim |
| **Dropout** | Aşırı öğrenmeyi önleme (Parça 2B) |
| **GPU eğitimi** | 2 GPU üzerinde paralel eğitim |
| **Data augmentation** | Veri artırma teknikleri |

## VGGNet (2014) — Derinliğin Gücü

Temel fikir: **3×3 küçük filtreler** kullanarak çok derin ağlar inşa et.

Neden 3×3? İki adet 3×3 katman, bir adet 5×5 katmanla aynı alıcı alana (receptive field) sahiptir, ama daha az parametre kullanır:

```
  5×5 tek katman:          3×3 iki katman:
  5×5 = 25 parametre       3×3 + 3×3 = 9+9 = 18 parametre
                            + 2 doğrusal olmama (non-linearity)
  Aynı receptive field, daha verimli! ✓
```

## ResNet (2015) — Artık Bağlantılar ⭐

Bu mimari derin öğrenmeyi köklü şekilde değiştirdi. Derin ağlardaki **kaybolan gradyan** sorununa zarif bir çözüm getirdi.

### Problem: Kaybolan Gradyan

```
  20 katmanlı ağ:  Eğitilebilir ✓
  56 katmanlı ağ:  Eğitim kaybı DAHA YÜKSEK! (daha derin = daha kötü?!)

  Neden? Backpropagation'da (Parça 2A) gradyanlar her katmanda çarpılır.
  Çok katman = gradyanlar sıfıra yaklaşır = ilk katmanlar öğrenemez.
```

### Çözüm: Artık Bağlantılar (Skip Connections)

```
  Normal blok:                     Artık (Residual) blok:

  x ──▶ [Katman] ──▶ [Katman] ──▶ F(x)     x ──▶ [Katman] ──▶ [Katman] ──┐
                                                                             │ (+)──▶ F(x) + x
                                              x ────────────────────────────┘
                                              (kısayol / skip connection)
```

**Matematiksel formülasyon:**

Normal blok: $\mathbf{y} = F(\mathbf{x})$ öğrenir

Artık blok: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$ → ağ **artığı** (residual) $F(\mathbf{x}) = \mathbf{y} - \mathbf{x}$ öğrenir

**Neden çalışır?**

1. **Gradyan otoyolu:** Geri yayılımda gradyan doğrudan skip connection üzerinden akabilir
2. **Kimlik öğrenme:** En kötü durumda $F(\mathbf{x}) = 0$ öğrenerek katman "atlanabilir"
3. **Daha derin ağlar:** 152+ katmanlı ağlar artık eğitilebilir

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                     # Skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity                   # F(x) + x
        out = self.relu(out)
        return out
```

> 📌 **Makale Bağlantısı:** Makale A, ResNet-50'yi adversarial saldırılara karşı test eder. ResNet, CNN ailesinin en yaygın temsilcisidir.

## EfficientNet (2019) — Akıllı Ölçekleme

Temel fikir: Bir modeli büyütürken genişlik, derinlik ve çözünürlüğü **birlikte** ölçekle.

```
  Sadece derinlik ↑     Sadece genişlik ↑     Bileşik ölçekleme
  ┌──┐                  ┌──────────┐           ┌────────┐
  │  │                  │          │           │        │
  │  │                  │          │           │        │
  │  │                  │          │           │        │
  │  │                  │          │           │        │
  │  │                  └──────────┘           │        │
  │  │                                         │        │
  │  │                                         └────────┘
  │  │                                      Genişlik + Derinlik
  └──┘                                      + Çözünürlük birlikte ↑
  (yetersiz)              (yetersiz)          (en etkili) ✓
```

> 📌 **Makale Bağlantısı:** Makale A, EfficientNet'i de deney modellerinden biri olarak kullanır.

## Mimari Karşılaştırma Tablosu

| Mimari | Yıl | Katman | Parametre | Top-1 Acc. | Temel Yenilik |
|:---|:---:|:---:|:---:|:---:|:---|
| LeNet-5 | 1998 | 7 | 60K | — | İlk CNN |
| AlexNet | 2012 | 8 | 61M | 63.3% | ReLU, GPU, Dropout |
| VGG-16 | 2014 | 16 | 138M | 74.4% | 3×3 filtrelerde derinlik |
| ResNet-50 | 2015 | 50 | 25.6M | 76.1% | Skip connections |
| ResNet-152 | 2015 | 152 | 60.2M | 78.3% | Çok derin artık ağ |
| EfficientNet-B0 | 2019 | — | 5.3M | 77.1% | Bileşik ölçekleme |
| EfficientNet-B7 | 2019 | — | 66M | 84.3% | Ölçeklenmiş B0 |

> **Öğrendiklerimiz — CNN Mimarileri**
>
> ✅ LeNet → AlexNet → VGG → ResNet → EfficientNet: giderek daha derin ve verimli
> ✅ ResNet'in skip connection'ları kaybolan gradyan sorununu çözdü
> ✅ EfficientNet bileşik ölçeklemeyle verimlilik sağladı
> ✅ Makale A'da ResNet ve EfficientNet adversarial testlerde kullanılır

---

\newpage

# Transfer Öğrenme

## Neden Sıfırdan Eğitmiyoruz?

ImageNet üzerinde eğitilmiş bir ResNet-50 zaten "kenarları, dokuları, şekilleri, nesneleri" tanımayı öğrenmiştir. Bu bilgiyi yeni bir göreve **transfer** edebiliriz.

```
  ImageNet'te eğitilmiş ResNet:

  [Erken katmanlar]  [Orta katmanlar]  [Son katmanlar]   [Sınıflandırıcı]
   Kenarlar, dokular  Parçalar, şekiller  Nesneler         1000 sınıf
   (genel bilgi)      (genel bilgi)      (yarı özel)      (ImageNet'e özel)
         │                  │                 │                  │
         ▼                  ▼                 ▼                  ▼
   ──── KORU ──────── KORU ──────── KORU veya ────── DEĞİŞTİR ─
                                    İNCE AYAR         (yeni görev)
```

```python
import torchvision.models as models
import torch.nn as nn

# Önceden eğitilmiş ResNet-50 yükle
model = models.resnet50(pretrained=True)

# Son sınıflandırma katmanını değiştir (1000 → 2 sınıf)
model.fc = nn.Linear(model.fc.in_features, 2)

# Opsiyonel: Omurga katmanlarını dondur
for param in model.parameters():
    param.requires_grad = False
# Sadece son katmanı eğitilebilir yap
for param in model.fc.parameters():
    param.requires_grad = True

print(f"Eğitilebilir parametre: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# Sadece ~4K parametre (son katman), tüm model değil
```

> **Öğrendiklerimiz — Transfer Öğrenme**
>
> ✅ Önceden eğitilmiş ağırlıklar: sıfırdan başlamak yerine bilgiyi transfer et
> ✅ Feature extraction: omurgayı dondur, sadece sınıflandırıcıyı eğit
> ✅ Fine-tuning: tüm ağı küçük öğrenme hızıyla ince ayar yap

---

# Parça 3A Kavram Haritası & Parça 3B'ye Köprü

```
╔════════════════════════════════════════════════════════════════════╗
║                  PARÇA 3A — KAVRAM HARİTASI                       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Piksel → Görüntü Tensörü (Parça 1) → Normalizasyon               ║
║               │                                                    ║
║               ▼                                                    ║
║  Evrişim (Convolution) ──── Kernel, Stride, Padding               ║
║       │                                                            ║
║       ▼                                                            ║
║  Özellik Haritaları ──── Pooling → Boyut Azaltma                  ║
║       │                                                            ║
║       ▼                                                            ║
║  CNN Mimarileri:                                                   ║
║  LeNet → AlexNet → VGG → ResNet (skip conn.) → EfficientNet       ║
║                              │                      │              ║
║                              │                      │              ║
║                          Makale A ◄─────────────────┘              ║
║                              │                                     ║
║               Transfer Öğrenme (pre-trained ağırlıklar)           ║
║                              │                                     ║
║                              ▼                                     ║
║                     PARÇA 3B: Transformer & ViT ──▶               ║
║                     PARÇA 4: Adversarial ML ──▶                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

## Parça 3B'ye Köprü

CNN'ler güçlüdür ama bir temel sınırlama taşır: **yerellik önyargısı** (locality bias). Evrişim çekirdeği sadece küçük bir pencereyi görür. Uzak pikseller arasındaki ilişkileri yakalamak için çok sayıda katmana (derin ağ) ihtiyaç vardır.

2017'de tamamen farklı bir yaklaşım ortaya çıktı: **Dikkat mekanizması** (Attention). Bu mekanizma, bir pozisyonun görüntüdeki **tüm** diğer pozisyonlarla doğrudan ilişki kurmasını sağlar — yerellik kısıtı olmadan.

**Parça 3B'de:**

1. Dikkat mekanizmasını (Q, K, V) öğreneceğiz
2. Transformer mimarisini inceleyeceğiz
3. **Vision Transformer (ViT)** ile tanışacağız — Makale B'nin omurgası
4. CNN vs. ViT karşılaştırmasını yapacağız — Makale A'nın temel sorusu

---

*Devam: Parça 3B — Dikkat Mekanizması, Transformer & Vision Transformer →*
