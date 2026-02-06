---
title: "Parça 2B: Optimizasyon, PyTorch & Kod Atölyesi"
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
  - \fancyhead[L]{Parça 2B — Optimizasyon, PyTorch}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Optimizasyon Algoritmaları

## SGD — Stokastik Gradyan İnişi

Parça 1'de öğrendiğimiz gradyan inişinin pratikte kullanılan versiyonu.

### Batch, Mini-batch, Stokastik

| Yöntem | Kullanılan veri | Avantaj | Dezavantaj |
|:---|:---|:---|:---|
| Batch GD | Tüm veri seti | Kararlı gradyan | Yavaş, bellek sorunlu |
| **Mini-batch SGD** | Küçük alt küme (32-256) | Dengeli | En yaygın seçim ✓ |
| Stokastik GD | Tek örnek | Çok hızlı | Çok gürültülü |

```
  Batch GD:          Mini-batch SGD:       Stokastik GD:

  ╲                  ╲                     ╲ ╱
   ╲                  ╲╱                    ╲╱ ╲
    ╲                  ╲                      ╲
     ╲                  ╲╱                   ╱ ╲╱
      ●                  ●                    ●

  Düz ama yavaş      Hafif salınımlı       Çok salınımlı
                      (en yaygın)
```

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t; \mathbf{x}_{batch}, \mathbf{y}_{batch})$$

## Momentum

**Analoji:** Tepenin aşağı yuvarlanan bir top düşün. Top, önceki hızını koruyarak küçük tümsekleri aşabilir.

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta L$$
$$\theta_{t+1} = \theta_t - \alpha \mathbf{v}_t$$

$\beta$ = momentum katsayısı (genellikle 0.9)

```
  Momentum olmadan:            Momentum ile:

       ╱╲    ╱╲                    ──────▶
      ╱  ╲  ╱  ╲                 (tümsekleri aşar)
     ╱    ╲╱    ╲
    (vadide takılabilir)
```

## Adam (Adaptive Moment Estimation)

Pratikte en çok kullanılan optimizer. Her parametre için **ayrı öğrenme hızı** ayarlar.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1. moment: ortalama gradyan)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2. moment: gradyan varyansı)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias düzeltme)}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

| Hiperparametre | Varsayılan | Anlam |
|:---|:---:|:---|
| $\alpha$ | $10^{-3}$ | Öğrenme hızı |
| $\beta_1$ | 0.9 | 1. moment üstel bozunum |
| $\beta_2$ | 0.999 | 2. moment üstel bozunum |
| $\epsilon$ | $10^{-8}$ | Sıfıra bölme önleme |

> 💡 **İpucu:** Şüpheye düşersen Adam kullan. Çoğu derin öğrenme çalışmasında varsayılan seçimdir.

## Optimizer Karşılaştırma Tablosu

| Optimizer | Adaptif LR | Momentum | Yaygın Kullanım |
|:---|:---:|:---:|:---|
| SGD | ✗ | ✗ | Basit problemler |
| SGD+Momentum | ✗ | ✓ | CNN eğitimi |
| Adam | ✓ | ✓ | Genel amaçlı, Transformer |
| AdamW | ✓ | ✓ | ViT, modern modeller |

## Learning Rate Scheduling

Eğitim boyunca öğrenme hızını azaltmak genellikle performansı artırır:

```
  Sabit LR:              Step Decay:           Cosine Annealing:

  α│────────────         α│────┐                α│╲
   │                      │    └───┐              │  ╲
   │                      │        └───┐          │    ╲
   │                      │            └──        │     ╲___
   └──────── epoch        └──────── epoch         └──────── epoch
```

> **Öğrendiklerimiz — Optimizasyon**
>
> ✅ Mini-batch SGD: gradyan inişinin pratik versiyonu
> ✅ Momentum: önceki hızı koruyarak daha kararlı iniş
> ✅ Adam: adaptif öğrenme hızı, en yaygın seçim
> ✅ LR scheduling: eğitim boyunca öğrenme hızını ayarlama

---

\newpage

# Aşırı Öğrenme ve Düzenlileştirme

## Eğitim / Doğrulama / Test Ayrımı

```
  Tüm Veri
  ┌──────────────────────────────────────────────┐
  │  Eğitim (%70)    │ Doğrulama (%15) │ Test (%15) │
  │  (train)         │ (validation)     │ (test)     │
  │                  │                  │            │
  │  Model bu        │  Model ayarı     │  Son       │
  │  veriyle         │  (hiperparametre │  performans│
  │  öğrenir         │  seçimi)         │  raporu    │
  └──────────────────┴──────────────────┴────────────┘
```

> ⚠️ **Dikkat:** Test seti hiçbir zaman eğitim kararlarında kullanılmamalı!

## Aşırı Öğrenme (Overfitting) ve Eksik Öğrenme (Underfitting)

```
  Kayıp                         Kayıp
   │╲   Eğitim                   │    Eğitim ───────
   │ ╲  ─────────                │    Doğrulama ──────
   │  ╲                          │
   │   ╲ Doğrulama               │
   │    ╲─────╱                  │
   │         ╱ (overfitting!)    │
   └──────────── epoch           └──────────── epoch
   Aşırı Öğrenme                 Eksik Öğrenme
   (eğitim ↓, doğrulama ↑)      (ikisi de yüksek)
```

```
  Model Karmaşıklığı Spektrumu:

  Underfitting ◄──────────────────────▶ Overfitting
  (çok basit)         İdeal           (çok karmaşık)

  ───────            ╱╲╱╲            ╱╲╱╲╱╲╱╲╱╲
  (düz çizgi)      (dengeli)        (her noktaya fit)
```

## Düzenlileştirme (Regularization) Yöntemleri

### L2 Regularization (Weight Decay)

Ağırlıkları küçük tutmaya zorlar:

$$L_{toplam} = L_{veri} + \lambda \|\theta\|_2^2 = L_{veri} + \lambda \sum_i \theta_i^2$$

### Dropout

Eğitim sırasında rastgele nöronları "kapatır":

```
  Normal İleri Yayılım:        Dropout (p=0.5):

  ○──○──○──○──○                ○──●──○──●──○
  │╲ │╲ │╲ │╲ │                │  ╳  │  ╳  │
  ○──○──○──○──○                ○──○──●──○──○
  │╲ │╲ │╲ │╲ │                │╲ │   ╲ │╲ │
  ○──○──○──○──○                ○──○──○──○──●

  ○ = aktif nöron              ● = kapatılmış nöron
```

### Early Stopping

Doğrulama kaybı artmaya başladığında eğitimi durdur.

### Data Augmentation (Veri Artırma)

Eğitim verisini çoğaltarak genellemeyi artırır (detay Parça 3'te):

```
  [Orijinal] → [Döndür] → [Çevir] → [Kırp] → [Renk değiştir]
```

> 📌 **Makale Bağlantısı:** Adversarial training (adversarial eğitim) aslında bir düzenlileştirme tekniği olarak da görülebilir — eğitim setine adversarial örnekler ekleyerek modelin daha dayanıklı hale gelmesini sağlar (Parça 4'te detay).

> **Öğrendiklerimiz — Overfitting & Regularization**
>
> ✅ Overfitting: eğitimi ezberler, genelleyemez
> ✅ L2 regularization: ağırlıkları küçük tutar
> ✅ Dropout: rastgele nöron kapatma
> ✅ Early stopping: doğrulama kaybı artınca dur
> ✅ Adversarial training da bir düzenlileştirmedir (Parça 4)

---

\newpage

# PyTorch'a Giriş

## Neden PyTorch?

- Dinamik hesaplama grafı (debug kolaylığı)
- Pythonic API (NumPy'a benzer)
- Güçlü otomatik türev (autograd) sistemi
- Akademi ve endüstride en yaygın framework

## Tensörler — NumPy Dizilerine Benzer

```python
import torch

# Oluşturma
x = torch.tensor([1.0, 2.0, 3.0])
print(x)              # tensor([1., 2., 3.])
print(x.shape)        # torch.Size([3])
print(x.dtype)        # torch.float32

# NumPy benzeri işlemler
M = torch.randn(3, 4)            # 3×4 normal dağılımlı
Z = torch.zeros(2, 3)            # 2×3 sıfır
I = torch.eye(3)                 # 3×3 birim matris

# NumPy ↔ PyTorch dönüşümü
import numpy as np
np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)
geri_np = torch_tensor.numpy()

# GPU desteği (varsa)
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(f"GPU'da: {x_gpu.device}")
```

## Autograd — Otomatik Türev

PyTorch, hesaplama grafını otomatik oluşturur ve gradyanları hesaplar:

```python
import torch

# requires_grad=True: "bu tensörün gradyanını takip et"
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# İleri hesaplama (otomatik graf oluşturulur)
y = w * x + b        # y = 2×3 + 1 = 7
loss = (y - 5) ** 2  # L = (7-5)² = 4

# Geri yayılım (tek komut!)
loss.backward()

# Gradyanları oku
print(f"y = {y.item()}")           # 7.0
print(f"loss = {loss.item()}")     # 4.0
print(f"∂L/∂w = {w.grad.item()}")  # 2*(7-5)*x = 2*2*3 = 12.0
print(f"∂L/∂x = {x.grad.item()}")  # 2*(7-5)*w = 2*2*2 = 8.0
print(f"∂L/∂b = {b.grad.item()}")  # 2*(7-5)*1 = 4.0
```

> 📌 **Makale Bağlantısı:** Adversarial saldırılarda `x.grad` kullanılarak girdiye göre gradyan hesaplanır. FGSM saldırısı (Parça 4) tam olarak bu mekanizmayı kullanır:
> ```python
> x.requires_grad = True
> loss = criterion(model(x), y)
> loss.backward()
> x_adv = x + epsilon * x.grad.sign()  # FGSM!
> ```

## nn.Module ile Model Tanımlama

```python
import torch
import torch.nn as nn

class BasitMLP(nn.Module):
    def __init__(self, girdi_boyut, gizli_boyut, sinif_sayisi):
        super().__init__()
        self.katman1 = nn.Linear(girdi_boyut, gizli_boyut)
        self.relu = nn.ReLU()
        self.katman2 = nn.Linear(gizli_boyut, gizli_boyut)
        self.katman3 = nn.Linear(gizli_boyut, sinif_sayisi)

    def forward(self, x):
        x = self.relu(self.katman1(x))  # Katman 1 + ReLU
        x = self.relu(self.katman2(x))  # Katman 2 + ReLU
        x = self.katman3(x)             # Çıktı katmanı (logitler)
        return x

# Model oluştur
model = BasitMLP(girdi_boyut=784, gizli_boyut=256, sinif_sayisi=10)
print(model)

# Parametre sayısı
toplam_param = sum(p.numel() for p in model.parameters())
print(f"Toplam parametre: {toplam_param:,}")
# 784×256 + 256 + 256×256 + 256 + 256×10 + 10 = 269,322
```

## Eğitim Döngüsü Anatomisi

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Bileşenler
model = BasitMLP(784, 256, 10)
criterion = nn.CrossEntropyLoss()        # Kayıp fonksiyonu
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Eğitim döngüsü (pseudo-code)
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 1. Gradyanları sıfırla
        optimizer.zero_grad()

        # 2. İleri yayılım
        tahminler = model(batch_x)

        # 3. Kayıp hesapla
        loss = criterion(tahminler, batch_y)

        # 4. Geri yayılım
        loss.backward()

        # 5. Parametre güncelleme
        optimizer.step()
```

```
  Eğitim Döngüsü Akış Şeması:

  ┌─────────────────────────────────────────────────────┐
  │  Her epoch için:                                    │
  │  ┌─────────────────────────────────────────────┐    │
  │  │  Her batch için:                            │    │
  │  │                                             │    │
  │  │  ① optimizer.zero_grad()  ← Gradyanları temizle │
  │  │         │                                   │    │
  │  │         ▼                                   │    │
  │  │  ② output = model(x)     ← İleri yayılım   │    │
  │  │         │                                   │    │
  │  │         ▼                                   │    │
  │  │  ③ loss = criterion(output, y) ← Kayıp     │    │
  │  │         │                                   │    │
  │  │         ▼                                   │    │
  │  │  ④ loss.backward()       ← Geri yayılım    │    │
  │  │         │                                   │    │
  │  │         ▼                                   │    │
  │  │  ⑤ optimizer.step()      ← Güncelleme      │    │
  │  │                                             │    │
  │  └─────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────┘
```

## Dataset ve DataLoader

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.ToTensor(),           # PIL → Tensor, [0,1] aralığına normalize
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST ortalaması ve std'si
])

# MNIST veri seti
train_dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)

# DataLoader: veriyi batch'ler halinde sunar
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Bir batch incele
images, labels = next(iter(train_loader))
print(f"Batch görüntü boyutu: {images.shape}")  # torch.Size([64, 1, 28, 28])
print(f"Batch etiket boyutu:  {labels.shape}")   # torch.Size([64])
```

> **Öğrendiklerimiz — PyTorch**
>
> ✅ torch.tensor: GPU destekli tensörler
> ✅ autograd: otomatik türev hesaplama (loss.backward())
> ✅ nn.Module: model tanımlama çerçevesi
> ✅ Eğitim döngüsü: zero_grad → forward → loss → backward → step
> ✅ DataLoader: veriyi batch'ler halinde sunma

---

\newpage

# Kod Atölyesi — MNIST Sınıflandırıcı

## Tam Çalışan MNIST MLP Eğitimi

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ─── 1. Ayarlar ───
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {DEVICE}")

# ─── 2. Veri Hazırlama ───
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

# ─── 3. Model Tanımlama ───
class MNISTKlasifiye(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 28×28 → 784
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)      # Regularization
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 10)        # 10 sınıf (0-9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)                     # Logitler (softmax yok!)
        return x

model = MNISTKlasifiye().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")

# ─── 4. Eğitim Fonksiyonu ───
def egit(model, loader, criterion, optimizer, device):
    model.train()
    toplam_kayip = 0
    dogru = 0
    toplam = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        toplam_kayip += loss.item() * images.size(0)
        dogru += (output.argmax(1) == labels).sum().item()
        toplam += labels.size(0)
    return toplam_kayip / toplam, dogru / toplam

# ─── 5. Test Fonksiyonu ───
def test(model, loader, criterion, device):
    model.eval()
    toplam_kayip = 0
    dogru = 0
    toplam = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            toplam_kayip += loss.item() * images.size(0)
            dogru += (output.argmax(1) == labels).sum().item()
            toplam += labels.size(0)
    return toplam_kayip / toplam, dogru / toplam

# ─── 6. Eğitim Döngüsü ───
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = egit(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = test(model, test_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} │ "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

# Beklenen çıktı:
# Epoch  1/10 │ Train Loss: 0.3021 Acc: 0.9134 │ Test Loss: 0.1254 Acc: 0.9612
# Epoch  5/10 │ Train Loss: 0.0832 Acc: 0.9742 │ Test Loss: 0.0734 Acc: 0.9778
# Epoch 10/10 │ Train Loss: 0.0498 Acc: 0.9847 │ Test Loss: 0.0712 Acc: 0.9803
```

## Eğitim Eğrilerini Görselleştirme

```python
# ─── 7. Görselleştirme ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Kayıp eğrileri
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Eğitim')
axes[0].plot(range(1, EPOCHS+1), test_losses, 'r-o', label='Test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Kayıp (Loss)')
axes[0].set_title('Kayıp Eğrisi')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Doğruluk eğrileri
axes[1].plot(range(1, EPOCHS+1), train_accs, 'b-o', label='Eğitim')
axes[1].plot(range(1, EPOCHS+1), test_accs, 'r-o', label='Test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Doğruluk (Accuracy)')
axes[1].set_title('Doğruluk Eğrisi')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_egitim_egrileri.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Tahminleri Görselleştirme

```python
# ─── 8. Tahminleri Görselleştir ───
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)

with torch.no_grad():
    output = model(images)
    probs = torch.softmax(output, dim=1)
    preds = output.argmax(1)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))

for i in range(8):
    # Doğru tahminler
    axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
    renk = 'green' if preds[i] == labels[i] else 'red'
    axes[0, i].set_title(f"T:{preds[i].item()} ({probs[i, preds[i]]:.2f})",
                         color=renk, fontsize=10)
    axes[0, i].axis('off')

# Yanlış tahminleri bul ve göster
yanlis_idx = (preds != labels).nonzero(as_tuple=True)[0]
for i, idx in enumerate(yanlis_idx[:8]):
    axes[1, i].imshow(images[idx].cpu().squeeze(), cmap='gray')
    axes[1, i].set_title(f"T:{preds[idx].item()} G:{labels[idx].item()}",
                         color='red', fontsize=10)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Doğru', fontsize=12)
axes[1, 0].set_ylabel('Yanlış', fontsize=12)
plt.suptitle('Model Tahminleri (T=Tahmin, G=Gerçek)', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_tahminler.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

\newpage

# Parça 2 — Genel Kavram Haritası

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    PARÇA 2 — TAM KAVRAM HARİTASI                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PARÇA 1'DEN:                                                        ║
║  Matris çarpımı, Gradyan, Zincir kuralı, Softmax, NumPy             ║
║       │              │           │                                    ║
║       ▼              ▼           ▼                                    ║
║  ┌─────────┐    ┌─────────┐ ┌────────────┐                          ║
║  │Perceptron│───▶│  MLP    │ │Backpropag. │                          ║
║  └─────────┘    └────┬────┘ └──────┬─────┘                          ║
║                      │             │                                  ║
║              ┌───────┼─────────────┤                                  ║
║              ▼       ▼             ▼                                  ║
║  ┌───────────┐ ┌──────────┐ ┌──────────────┐                        ║
║  │Aktivasyon │ │  Kayıp   │ │ Optimizasyon │                        ║
║  │(ReLU,GELU)│ │(CE Loss) │ │ (SGD, Adam)  │                        ║
║  └───────────┘ └──────────┘ └──────────────┘                        ║
║       │              │             │                                  ║
║       └──────────────┼─────────────┘                                  ║
║                      ▼                                                ║
║              ┌──────────────┐        ┌────────────────┐              ║
║              │   PyTorch    │        │ Regularization │              ║
║              │ Eğitim Döngüsü│       │(Dropout, L2)   │              ║
║              └──────┬───────┘        └────────────────┘              ║
║                     ▼                                                 ║
║              ┌──────────────┐                                        ║
║              │MNIST Klasifiye│ ← İlk gerçek modelimiz!               ║
║              │ %98 doğruluk │                                        ║
║              └──────┬───────┘                                        ║
║                     │                                                 ║
║          ┌──────────┼──────────┐                                     ║
║          ▼          ▼          ▼                                      ║
║      PARÇA 3     PARÇA 4    PARÇA 5                                  ║
║     (CNN, ViT)  (Saldırı)  (Frekans)                                ║
║                                                                      ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

# Sonraki Parçaya Köprü — Parça 3'e Hazırlık

Bu parçada sinir ağlarını inşa ettik, eğittik ve test ettik. Ancak MLP'nin önemli bir sınırlaması var: **her piksele bağımsız davranır**. Bir görüntüde pikselin komşularıyla ilişkisi kritik bilgi taşır — kenarlar, dokular, şekiller...

**Parça 3'te** iki temel mimariyi öğreneceğiz:

| Mimari | Temel Fikir | Analoji |
|:---|:---|:---|
| **CNN** | Yerel kalıpları yakala (evrişim) | Resmi büyüteçle taramak |
| **Transformer/ViT** | Küresel ilişkileri yakala (dikkat) | Tüm resme kuş bakışı bakmak |

Bu iki mimari, **Makale A'nın doğrudan karşılaştırma konusu** ve **Makale B'nin omurgası**dır.

---

*Devam: Parça 3A — Bilgisayarlı Görü, Evrişim & CNN Mimarileri →*
