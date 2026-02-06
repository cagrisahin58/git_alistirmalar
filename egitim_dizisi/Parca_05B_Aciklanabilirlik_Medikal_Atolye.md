---
title: "Parça 5B: Açıklanabilirlik, Medikal Görüntü Analizi & Kod Atölyesi"
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
  - \fancyhead[L]{Parça 5B — Açıklanabilirlik \& Medikal}
  - \fancyhead[R]{Adversarial DL Eğitim Serisi}
---

\newpage

# Açıklanabilirlik (Explainability) — Neden Önemli?

## Kara Kutu Sorunu

**Analoji:** İki doktor düşün:
- **Doktor A:** "Hastasınız." (başka açıklama yok)
- **Doktor B:** "Bu kan değerleriniz normalin üstünde, bu röntgende şu bölgede bir gölge var, bu nedenle hastalık şüphesi var."

Hangisine güvenirsin? Tabii ki Doktor B'ye. Derin öğrenme modelleri genellikle **Doktor A** gibi davranır — karar verir ama nedenini açıklamaz.

```
  Kara Kutu Model:                 Açıklanabilir Model:

  [Görüntü] ──▶ [????] ──▶ Kedi   [Görüntü] ──▶ [Model] ──▶ Kedi
                                                     │
                                               [Açıklama Haritası]
                                                ┌──────────────┐
                                                │     ████     │
                                                │    ██████    │
                                                │     ████     │ ← "Buraya
                                                │              │    bakarak
                                                └──────────────┘    karar verdim"
```

## Açıklanabilirlik Taksonomisi

| Kriter | Tür 1 | Tür 2 |
|:---|:---|:---|
| **Zamanlama** | Post-hoc (sonradan) | Ante-hoc (tasarım gereği) |
| **Kapsam** | Yerel (tek tahmin) | Küresel (genel davranış) |
| **Bağımlılık** | Model-agnostik | Model-spesifik |

> 📌 **Makale Bağlantısı:** Her iki makale de **post-hoc, yerel, model-spesifik** açıklanabilirlik yöntemleri kullanır: CNN için Grad-CAM, ViT için dikkat haritaları.

---

\newpage

# Grad-CAM (Gradient-weighted Class Activation Mapping)

## Selvaraju et al. (2017)

Grad-CAM, CNN'lerin son evrişim katmanındaki gradyanları kullanarak **hangi bölgeye bakarak karar verdiğini** gösteren bir ısı haritası üretir.

## Matematiksel Formülasyon

**Adım 1:** Sınıf $c$ için son evrişim katmanının $k$. özellik haritası $A^k$'nın önem ağırlığı:

$$\alpha_k^c = \underbrace{\frac{1}{Z}\sum_i \sum_j}_{\text{Global Average Pooling}} \underbrace{\frac{\partial y^c}{\partial A_{ij}^k}}_{\text{Gradyan}}$$

**Adım 2:** Ağırlıklı toplam + ReLU:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

ReLU neden? Sadece **pozitif etkili** özellikleri görmek istiyoruz.

## Adım Adım Süreç

```
  ① Girdi x'i modele ver (ileri yayılım)
  ② Hedef sınıf c'nin skoru y^c'yi al
  ③ y^c'yi son conv katmanının çıktısına göre türevle
  ④ Gradyanları global average pool et → αₖ ağırlıkları
  ⑤ Özellik haritalarını αₖ ile ağırlıklı topla
  ⑥ ReLU uygula → Grad-CAM ısı haritası
  ⑦ Girdi görüntüsü boyutuna yeniden boyutlandır (bilinear interpolation)

  [Girdi] → [Conv...Conv] → [Son Conv] → [FC] → yᶜ
                               │                   │
                               A^k          ∂yᶜ/∂A^k
                               │                   │
                               └───── × ────────────┘
                                       │
                                  Σ_k (α_k · A^k)
                                       │
                                    ReLU
                                       │
                                  [Isı Haritası]
```

## PyTorch Uygulaması

```python
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook'ları kaydet
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()

        # İleri yayılım
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Geri yayılım
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Ağırlıklar (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # α_k

        # Ağırlıklı toplam
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Girdi boyutuna yeniden boyutlandır
        cam = nn.functional.interpolate(cam, size=input_tensor.shape[2:],
                                         mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

# Kullanım
model = models.resnet50(pretrained=True)
model.eval()
grad_cam = GradCAM(model, model.layer4[-1])

# x = ... (girdi görüntüsü tensörü, [1, 3, 224, 224])
# heatmap = grad_cam.generate(x)
# plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Görüntü üzerine bindirme
```

> 📌 **Makale Bağlantısı:** Her iki makale de Grad-CAM'i görselleştirme aracı olarak kullanır. Makale A, adversarial pertürbasyon altında Grad-CAM haritalarının nasıl değiştiğini inceler.

> **Öğrendiklerimiz — Grad-CAM**
>
> ✅ Son conv katmanının gradyanlarından ısı haritası üretir
> ✅ "Model nereye bakarak karar verdi?" sorusuna görsel cevap
> ✅ CNN'lere özgü (conv katmanı gerektirir)

---

\newpage

# Dikkat Haritaları (Attention Maps) — ViT İçin

## ViT'in Doğal Açıklanabilirliği

ViT'lerde dikkat ağırlıkları (Parça 3B) doğrudan **açıklanabilirlik** aracı olarak kullanılabilir: "hangi yama hangi yamaya dikkat ediyor?"

```
  [CLS] tokeninin dikkat dağılımı:

  ┌──┬──┬──┬──┬──┬──┬──┐
  │.2│.1│.0│.0│.0│.1│.0│    [CLS] en çok sol üst yamalara
  ├──┼──┼──┼──┼──┼──┼──┤    dikkat ediyor → orada nesne var
  │.1│.3│.1│.0│.0│.0│.0│
  ├──┼──┼──┼──┼──┼──┼──┤
  │.0│.1│.0│.0│.0│.0│.0│
  └──┴──┴──┴──┴──┴──┴──┘
```

## Attention Rollout

Tek bir katmanın dikkat ağırlıkları yanıltıcı olabilir. **Attention Rollout**, tüm katmanlar boyunca dikkati biriktirir:

$$\tilde{A}_l = 0.5 \cdot A_l + 0.5 \cdot I \quad \text{(artık bağlantı etkisi)}$$

$$R = \prod_{l=1}^{L} \tilde{A}_l$$

```python
def attention_rollout(attention_weights):
    """Tüm katmanlar boyunca dikkat rollout hesapla.

    Args:
        attention_weights: list of (B, H, N, N) tensörleri, her katman için
    """
    result = torch.eye(attention_weights[0].size(-1))

    for attention in attention_weights:
        # Başlıklar üzerinde ortalama
        attention_avg = attention.mean(dim=1)  # (B, N, N)
        # Artık bağlantı etkisi
        attention_residual = 0.5 * attention_avg + 0.5 * torch.eye(attention_avg.size(-1))
        # Normalize
        attention_residual = attention_residual / attention_residual.sum(dim=-1, keepdim=True)
        # Biriktir
        result = result @ attention_residual

    # [CLS] tokeninin diğer tüm yamalara dikkati
    cls_attention = result[0, 0, 1:]  # İlk token (CLS) → diğer yamalar
    # Yama grid'ine yeniden şekillendir
    num_patches = int(cls_attention.size(0) ** 0.5)
    attention_map = cls_attention.reshape(num_patches, num_patches)

    return attention_map.detach().numpy()
```

## CNN Grad-CAM vs. ViT Attention

| Özellik | Grad-CAM (CNN) | Attention Map (ViT) |
|:---|:---|:---|
| Kaynak | Son conv katmanı gradyanları | Dikkat ağırlıkları |
| Çözünürlük | Özellik haritası boyutu (ör. 7×7) | Yama grid boyutu (ör. 14×14) |
| Post-hoc? | Evet (backprop gerekli) | Kısmen (ağırlıklar doğrudan mevcut) |
| Güvenilirlik | Orta (yaklaşık) | Tartışmalı (dikkat ≠ açıklama her zaman) |

> 📌 **Makale Bağlantısı:** Makale A, adversarial pertürbasyon altında CNN Grad-CAM ve ViT dikkat haritalarının nasıl değiştiğini karşılaştırır.

---

\newpage

# Explanation Consistency (Açıklama Tutarlılığı)

## Tanım

Eğer bir model gerçekten sağlam (robust) ise, clean ve adversarial girdi için **aynı bölgelere bakarak** karar vermeli.

$$\text{EC}(\mathbf{x}, \mathbf{x}_{adv}) = \text{sim}\left(E(f_\theta, \mathbf{x}),\; E(f_\theta, \mathbf{x}_{adv})\right)$$

- $E(f_\theta, \mathbf{x})$: modelin $\mathbf{x}$ için açıklama haritası (Grad-CAM veya Attention)
- $\text{sim}$: benzerlik ölçüsü (kosinüs benzerliği, $L_2$ mesafesi vb.)

```
  Robust model:                    Kırılgan model:

  Clean:     Adversarial:          Clean:     Adversarial:
  ┌────────┐ ┌────────┐           ┌────────┐ ┌────────┐
  │  ████  │ │  ████  │           │  ████  │ │ ████   │
  │ ██████ │ │ ██████ │           │ ██████ │ │    ████│
  │  ████  │ │  ████  │           │  ████  │ │ ██     │
  └────────┘ └────────┘           └────────┘ └────────┘
  Benzer!    EC ≈ 0.95            Farklı!    EC ≈ 0.30
  (aynı yere bakıyor)            (dikkat kaydı!)
```

## Explanation Consistency Loss (ECL)

> 📌 **Makale Bağlantısı:** Bu, Makale B'deki **ECL** bileşeninin tam karşılığıdır.

$$L_{ECL} = 1 - \text{sim}\left(E(f_\theta, \mathbf{x}),\; E(f_\theta, \mathbf{x}_{adv})\right)$$

Toplam eğitim kaybı:

$$\boxed{L_{total} = \underbrace{L_{cls}}_{\text{sınıflandırma}} + \alpha \underbrace{L_{adv}}_{\text{adversarial robust.}} + \beta \underbrace{L_{ECL}}_{\text{açıklama tutarlılığı}}}$$

| Terim | Rolü | Bağlantı |
|:---|:---|:---|
| $L_{cls}$ | Doğru sınıflandırma | Parça 2A |
| $L_{adv}$ | Adversarial dayanıklılık | Parça 4B |
| $L_{ECL}$ | Açıklama tutarlılığı | **Bu bölüm** |

**ECL'nin faydaları:**

1. Model adversarial girdide de doğru bölgelere bakmaya zorlanır
2. Robustness ve açıklanabilirlik birlikte iyileşir
3. Özellikle medikal görüntülerde: "sahte bölgeyi her koşulda göster"

```python
def explanation_consistency_loss(cam_clean, cam_adv):
    """Açıklama tutarlılığı kaybı.

    Args:
        cam_clean: Clean girdi için Grad-CAM haritası (B, H, W)
        cam_adv: Adversarial girdi için Grad-CAM haritası (B, H, W)
    """
    # Düzleştir
    cam_clean_flat = cam_clean.flatten(1)  # (B, H*W)
    cam_adv_flat = cam_adv.flatten(1)

    # Kosinüs benzerliği
    cos_sim = nn.functional.cosine_similarity(cam_clean_flat, cam_adv_flat, dim=1)

    # ECL = 1 - benzerlik
    ecl = 1.0 - cos_sim.mean()
    return ecl

# Eğitim döngüsünde:
# loss = loss_cls + alpha * loss_adv + beta * ecl_loss
```

> **Öğrendiklerimiz — Explanation Consistency**
>
> ✅ EC: clean ve adversarial açıklamalar benzer olmalı
> ✅ ECL: bu tutarlılığı eğitim kaybına ekleyerek zorla
> ✅ $L_{total} = L_{cls} + \alpha L_{adv} + \beta L_{ECL}$
> ✅ Makale B'nin üç ana bileşeninden biri

---

\newpage

# Medikal Görüntü Analizi Bağlamı

## Medikal Görüntü Türleri

| Tür | Açıklama | Örnek Kullanım |
|:---|:---|:---|
| **X-ray** | İki boyutlu radyografi | Akciğer, kemik |
| **CT** | Kesitsel tomografi | Tümör tespiti |
| **MRI** | Manyetik rezonans | Beyin, yumuşak doku |
| **Ultrason** | Ses dalgası | Gebelik, kalp |
| **Patoloji** | Mikroskop görüntüleri | Kanser doku analizi |

## Medikal Görüntü Sahteciliği (Forgery)

**Motivasyonlar:**

- Sigorta dolandırıcılığı (hastalık uydurma)
- Araştırma sahteciliği (sonuçları manipüle etme)
- Kimlik sahteciliği (başkasının görüntüsünü kullanma)

**Sahtecilik türleri:**

```
  Copy-Move:           Splicing:            Deepfake/Generation:

  ┌──────────┐        ┌──────────┐         ┌──────────┐
  │ A ──▶ A' │        │  A + B   │         │  AI ile  │
  │ (kopyala │        │ (iki     │         │  üretilmiş│
  │  yapıştır)│       │  görüntüyü│        │  sahte   │
  └──────────┘        │  birleştir)│        │  görüntü │
                      └──────────┘         └──────────┘
```

## Klinik Gereklilikler

| Gereklilik | Açıklama | İlgili Kavram |
|:---|:---|:---|
| **Yüksek duyarlılık** | Sahteyi kaçırma! | Recall, sensitivity |
| **Açıklanabilirlik** | Neden sahte? Nereye bakıyorsun? | Grad-CAM, ECL |
| **Adversarial robustness** | Saldırgan sahteciği gizleyebilir | Adversarial training |
| **Gizlilik** | Hasta verisi korunmalı | HIPAA, KVKK |

> 📌 **Makale Bağlantısı:** Makale B'nin tüm tasarımı bu gerekliliklere dayanır:
> - **FE:** Sahtecilik izlerini frekans domain'de yakala
> - **ViT:** Küresel bağlamla sahte bölgeleri tespit et
> - **ECL:** Adversarial saldırı altında bile tutarlı açıklama ver

---

# Parça 5 — Genel Kavram Haritası

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    PARÇA 5 — TAM KAVRAM HARİTASI                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  FREKANS ANALİZİ (5A)              AÇIKLANABILIRLIK (5B)             ║
║  ═══════════════════                ═══════════════════               ║
║                                                                      ║
║  FFT / DCT                          Grad-CAM (CNN)                   ║
║  ├── Düşük frekans: şekiller        ├── Son conv gradyanları        ║
║  ├── Yüksek frekans: kenarlar       └── Isı haritası                ║
║  └── Pertürbasyon → yüksek f                                        ║
║       │                              Attention Maps (ViT)            ║
║       ▼                              ├── Dikkat ağırlıkları         ║
║  CNN: texture bias (yüksek f)        └── Attention rollout          ║
║  ViT: shape bias (düşük-orta f)                                     ║
║       │                                      │                       ║
║       ▼                                      ▼                       ║
║  ┌─────────────────────────────────────────────────┐                 ║
║  │        EXPLANATION CONSISTENCY (ECL)             │                 ║
║  │  Clean açıklama ≈ Adversarial açıklama           │                ║
║  │  L_total = L_cls + α·L_adv + β·L_ECL            │                 ║
║  └─────────────────────────────────────────────────┘                 ║
║                          │                                           ║
║            Medikal Görüntü Analizi Bağlamı                           ║
║            ├── Sahtecilik tespiti                                    ║
║            ├── Adversarial robustness                                ║
║            └── Klinik açıklanabilirlik                               ║
║                          │                                           ║
║                          ▼                                           ║
║  ┌───────────────────────────────────────────┐                       ║
║  │  MAKALE B: FE-ViT-ECL                      │                      ║
║  │  FE (Parça 5A) + ViT (Parça 3B) + ECL (5B) │                     ║
║  └───────────────────────────────────────────┘                       ║
║                                                                      ║
║  → PARÇA 6: Her iki makalenin tam anatomisi                         ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

# Sonraki Parçaya Köprü — Parça 6'ya Hazırlık

Artık **tüm yapı taşlarına** sahibiz:

| Parça | Öğrendiklerimiz | Makale Bağlantısı |
|:---|:---|:---|
| **1** | Matematik, Python | Tüm hesaplamaların temeli |
| **2** | ML, sinir ağları, PyTorch | Model eğitimi |
| **3** | CNN, ViT | Makale A ve B'nin model mimarileri |
| **4** | Adversarial saldırı/savunma | Her iki makalenin doğrudan konusu |
| **5** | Frekans, açıklanabilirlik | Makale B'nin FE ve ECL bileşenleri |

**Son parçada (Parça 6):** Tüm bu bilgileri birleştirerek hedef makaleleri **bileşen bileşen** çözeceğiz, eleştirel okuma yapacağız ve kendi araştırma sorularımızı formüle edeceğiz.

---

*Devam: Parça 6A — Akademik Makale Okuma & Makale A Analizi →*
