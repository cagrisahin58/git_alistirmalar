---
title: "Parça 4B: Adversarial Savunmalar, Metrikler & Kod Atölyesi"
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
  - \fancyhead[L]{Parça 4B — Savunmalar \& Kod Atölyesi}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Adversarial Savunma Mekanizmaları

## Adversarial Eğitim (Adversarial Training)

En etkili ve en yaygın savunma yöntemi. Temel fikir: eğitim sırasında adversarial örnekler üretip bunları da eğitim verisine dahil et.

### Min-Max Formülasyonu

$$\min_\theta \; \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} L(\theta, \mathbf{x} + \boldsymbol{\delta}, y) \right]$$

```
  İç döngü (max):  En güçlü saldırıyı bul (PGD ile)
  Dış döngü (min):  Bu saldırıya karşı dayanıklı ol (gradyan inişi ile)

  ┌──────────────────────────────────────────────────────┐
  │  Her eğitim adımında:                                │
  │                                                      │
  │  ① x_batch, y_batch al                               │
  │  ② PGD ile x_adv üret (iç max)                      │
  │  ③ L(θ, x_adv, y) hesapla                           │
  │  ④ ∇θ L hesapla (geri yayılım)                      │
  │  ⑤ θ güncelle (dış min)                              │
  │                                                      │
  │  Normal eğitimden ~3-10× yavaş (PGD adımları yüzünden)│
  └──────────────────────────────────────────────────────┘
```

### Clean Accuracy vs. Robust Accuracy Trade-off

```
  Doğruluk
  100%│── Normal eğitim (clean)
     │───────────────────────────── 95%
  90%│
     │── Adversarial eğitim (clean)
  85%│───────────────────────────── 83%  ← Clean accuracy düşer
     │
  60%│── Adversarial eğitim (robust)
     │───────────────────────────── 55%  ← Ama robust accuracy artar
     │
   0%│── Normal eğitim (robust)
     │───────────────────────────── ~0%  ← Savunmasız!
     └─────────────────────────────
```

> 📌 **Makale Bağlantısı:** Makale B, adversarial training ile modeli sağlamlaştırır. Clean accuracy'den bir miktar fedakarlık yapılarak robust accuracy kazanılır.

> ⚠️ **Dikkat:** Clean accuracy ile robust accuracy arasında bir **trade-off** vardır. Tamamen robust bir model genellikle daha düşük clean accuracy'ye sahiptir.

### PyTorch ile Adversarial Training

```python
def adversarial_train_epoch(model, loader, optimizer, epsilon, alpha, pgd_steps, device):
    """Bir epoch adversarial eğitim."""
    model.train()
    toplam_kayip = 0
    dogru_clean, dogru_robust, toplam = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # ─── Clean doğruluk ───
        with torch.no_grad():
            clean_out = model(images)
            dogru_clean += (clean_out.argmax(1) == labels).sum().item()

        # ─── PGD ile adversarial örnek üret ───
        x_adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(pgd_steps):
            x_adv.requires_grad_(True)
            loss_inner = nn.CrossEntropyLoss()(model(x_adv), labels)
            loss_inner.backward()
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - images, -epsilon, epsilon)
            x_adv = torch.clamp(images + delta, 0, 1).detach()

        # ─── Adversarial örnek üzerinde eğit ───
        optimizer.zero_grad()
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()

        toplam_kayip += loss.item() * images.size(0)
        dogru_robust += (output.argmax(1) == labels).sum().item()
        toplam += labels.size(0)

    return (toplam_kayip/toplam,
            dogru_clean/toplam,
            dogru_robust/toplam)
```

## Giriş Dönüşümü Savunmaları (Input Transformation)

Adversarial pertürbasyonları girdiye dönüşüm uygulayarak "temizleme" yaklaşımı:

| Yöntem | Açıklama | Etkinlik |
|:---|:---|:---:|
| JPEG sıkıştırma | Yüksek frekans bileşenlerini kaldırır | Orta |
| Bit derinliği azaltma | Piksel hassasiyetini düşürür | Düşük-Orta |
| Rastgele yeniden boyutlandırma | Pertürbasyonu dağıtır | Düşük |
| Gauss bulanıklaştırma | Yüksek frekansları yumuşatır | Orta |

> 💡 **İpucu:** Bu yöntemler tek başına yeterli değildir ama adversarial training ile birlikte kullanılabilir.

## Sertifikalı Savunmalar (Certified Defenses)

Matematiksel **garanti** veren savunmalar:

**Randomized Smoothing** (Cohen et al., 2019):

Modelin tahminini gürültülü girdiler üzerinde ortalayarak, belirli bir $L_2$ yarıçapı içinde **kanıtlanmış** robustness sağlar:

$$g(\mathbf{x}) = \arg\max_c \; \Pr[f(\mathbf{x} + \boldsymbol{\epsilon}) = c], \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)$$

## Savunma Değerlendirme Tuzakları

> ⚠️ **Dikkat — Gradient Masking:**
>
> Bazı savunmalar gradyanları "gizler" — gradyan tabanlı saldırılar başarısız görünür ama aslında model robust değildir!
>
> **Belirtiler:**
> - Beyaz kutu saldırı siyah kutudan daha başarısız (mantıksız!)
> - Bir adımlık saldırı iteratif saldırıdan daha başarılı
> - Unbounded saldırı başarısız (ε=1.0 bile kandıramıyor)
>
> **Çözüm:** AutoAttack gibi çeşitli saldırılar kullan.

> **Öğrendiklerimiz — Savunmalar**
>
> ✅ Adversarial training: en etkili savunma, min-max optimizasyon
> ✅ Clean vs robust accuracy trade-off'u kaçınılmaz
> ✅ Giriş dönüşümleri: yardımcı ama tek başına yetersiz
> ✅ Gradient masking: yanlış güvenlik hissi — dikkat!

---

\newpage

# Robustness Değerlendirme Metrikleri

| Metrik | Formül | Açıklama |
|:---|:---|:---|
| **Clean Accuracy** | $\frac{1}{N}\sum \mathbb{1}[f(\mathbf{x}_i) = y_i]$ | Temiz veri doğruluğu |
| **Robust Accuracy** | $\frac{1}{N}\sum \mathbb{1}[f(\mathbf{x}_{adv,i}) = y_i]$ | Saldırı altında doğruluk |
| **Attack Success Rate (ASR)** | $\frac{\text{Başarılı saldırı sayısı}}{\text{Toplam saldırı}}$ | Saldırı başarı oranı |
| **Robustness Gap** | Clean Acc − Robust Acc | Saldırıdan kaynaklanan düşüş |

### Robustness Eğrisi

```
  Doğruluk
  95%│●
     │  ●
  80%│    ●
     │      ●
  60%│        ●
     │          ●
  40%│            ●
     │              ●
  20%│                ●
     │
   0%│                  ●
     └──┬──┬──┬──┬──┬──┬── ε (pertürbasyon bütçesi)
        0  2  4  8  12 16
           /255

  ε arttıkça doğruluk düşer.
  Robust model: eğri yavaş düşer.
  Kırılgan model: eğri hızla düşer.
```

---

\newpage

# Kod Atölyesi — Tam Çalışan Adversarial ML Deneyleri

## Atölye 1: MNIST Üzerinde FGSM Saldırısı

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ─── Model (Parça 2B'den) ───
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# ─── Veri ───
transform = transforms.ToTensor()
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# ─── FGSM Saldırısı ───
def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    x_adv = x + epsilon * x_adv.grad.sign()
    return torch.clamp(x_adv, 0, 1).detach()

# ─── Farklı ε değerlerinde test ───
def evaluate_robustness(model, loader, epsilon, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x_adv = fgsm_attack(model, images, labels, epsilon)
        preds = model(x_adv).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

device = torch.device('cpu')
model = MNISTModel().to(device)

# Eğitilmiş model varsayarak (Parça 2B'de eğittik):
# model.load_state_dict(torch.load('mnist_model.pth'))

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []
for eps in epsilons:
    acc = evaluate_robustness(model, test_loader, eps, device)
    accuracies.append(acc)
    print(f"ε = {eps:.2f} → Doğruluk: {acc:.4f}")

# Beklenen çıktı (yaklaşık):
# ε = 0.00 → Doğruluk: 0.9803
# ε = 0.05 → Doğruluk: 0.9541
# ε = 0.10 → Doğruluk: 0.8823
# ε = 0.15 → Doğruluk: 0.7364
# ε = 0.20 → Doğruluk: 0.5012
# ε = 0.25 → Doğruluk: 0.2856
# ε = 0.30 → Doğruluk: 0.1432

# ─── Robustness eğrisi ───
plt.figure(figsize=(8, 5))
plt.plot(epsilons, accuracies, 'ro-', linewidth=2, markersize=8)
plt.xlabel('ε (Pertürbasyon Bütçesi)', fontsize=12)
plt.ylabel('Doğruluk', fontsize=12)
plt.title('FGSM Saldırısı Altında Model Robustness Eğrisi', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.savefig('robustness_egrisi.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Atölye 2: Adversarial Görüntü Görselleştirme

```python
# ─── Tek bir görüntüde FGSM etkisi ───
images, labels = next(iter(test_loader))
idx = 0  # İlk görüntü
x = images[idx:idx+1].to(device)
y = labels[idx:idx+1].to(device)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
eps_values = [0, 0.05, 0.1, 0.2, 0.3]

for i, eps in enumerate(eps_values):
    if eps == 0:
        x_show = x
    else:
        x_show = fgsm_attack(model, x, y, eps)

    # Üst: görüntü
    axes[0, i].imshow(x_show.squeeze().cpu(), cmap='gray')
    pred = model(x_show).argmax(1).item()
    conf = torch.softmax(model(x_show), 1).max().item()
    color = 'green' if pred == y.item() else 'red'
    axes[0, i].set_title(f'ε={eps}\nTahmin:{pred} ({conf:.2f})', color=color)
    axes[0, i].axis('off')

    # Alt: pertürbasyon
    if eps == 0:
        axes[1, i].imshow(torch.zeros_like(x).squeeze().cpu(), cmap='RdBu', vmin=-0.3, vmax=0.3)
    else:
        delta = (x_show - x).squeeze().cpu()
        axes[1, i].imshow(delta, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[1, i].set_title(f'δ (×10 büyütülmüş)')
    axes[1, i].axis('off')

plt.suptitle('FGSM Saldırısı: Farklı ε Değerlerinde Etki', fontsize=14)
plt.tight_layout()
plt.savefig('fgsm_gorsellestirme.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Atölye 3: Normal vs. Adversarial Eğitim Karşılaştırması

```python
# Pseudo-code karşılaştırma (tam eğitim uzun sürer)

# Normal eğitim
for epoch in range(10):
    for x, y in train_loader:
        loss = criterion(model_normal(x), y)
        loss.backward()
        optimizer_normal.step()

# Adversarial eğitim (PGD-AT)
for epoch in range(10):
    for x, y in train_loader:
        # PGD ile x_adv üret
        x_adv = pgd_attack(model_robust, x, y, eps=8/255, alpha=2/255, steps=7)
        # Adversarial örnek üzerinde eğit
        loss = criterion(model_robust(x_adv), y)
        loss.backward()
        optimizer_robust.step()

# Sonuçlar (beklenen):
# ┌─────────────────┬────────────┬───────────────┐
# │ Model           │ Clean Acc  │ Robust Acc    │
# │                 │            │ (PGD, ε=8/255)│
# ├─────────────────┼────────────┼───────────────┤
# │ Normal eğitim   │   98.0%    │    0.0%       │
# │ Adversarial eğt.│   84.5%    │   45.2%       │
# └─────────────────┴────────────┴───────────────┘
```

---

# Parça 4 — Genel Kavram Haritası

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    PARÇA 4 — TAM KAVRAM HARİTASI                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SALDIRILAR (Parça 4A)              SAVUNMALAR (Parça 4B)           ║
║  ═══════════════════                ══════════════════               ║
║                                                                      ║
║  x_adv = x + δ, ||δ||∞ ≤ ε         Adversarial Training            ║
║       │                              min_θ max_δ L(θ,x+δ,y)         ║
║       ├─ FGSM (tek adım)                │                           ║
║       ├─ PGD (çok adım) ★               ├─ Input Transformations    ║
║       ├─ C&W (min pertürbasyon)          ├─ Certified (RandomSmooth) ║
║       └─ AutoAttack (ensemble)           └─ Detection-based         ║
║                                                                      ║
║  Gradyan Analizi (Makale A)         Metrikler                        ║
║  CNN: yerel, yapılandırılmış        ├─ Clean Accuracy               ║
║  ViT: global, dağınık               ├─ Robust Accuracy              ║
║                                      ├─ Attack Success Rate          ║
║  Transferability (Makale A)          └─ Robustness Eğrisi           ║
║  CNN→CNN: yüksek                                                     ║
║  CNN↔ViT: düşük                    ⚠️ Gradient Masking tuzağı!     ║
║                                                                      ║
║  ──────────────────────────────────────────────────────────────      ║
║  → PARÇA 5: Frekans domain'de pertürbasyon analizi                   ║
║  → PARÇA 5: Açıklanabilirlik (Grad-CAM, Attention Maps)             ║
║  → PARÇA 6: Makalelerin tam analizi                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

# Sonraki Parçaya Köprü — Parça 5'e Hazırlık

Adversarial pertürbasyonları piksel uzayında inceledik. Ama daha derin bir anlayış için iki yeni perspektife ihtiyacımız var:

**1. Frekans Domain Perspektifi (Parça 5A):**
- Pertürbasyonlar görüntünün hangi frekans bileşenlerini bozuyor?
- CNN ve ViT farklı frekans bantlarına mı duyarlı?
- Frekans bilgisi savunmayı güçlendirebilir mi?

**2. Açıklanabilirlik Perspektifi (Parça 5B):**
- Model kararını neye bakarak veriyor?
- Adversarial saldırı modelin "dikkatini" nasıl değiştiriyor?
- Açıklama tutarlılığı (Explanation Consistency) robustness'ı artırabilir mi?

Bu iki perspektif, Makale B'nin (FE-ViT-ECL) temel yapı taşlarıdır.

---

*Devam: Parça 5A — Fourier Dönüşümü, DCT & Frekans Domain Analizi →*
