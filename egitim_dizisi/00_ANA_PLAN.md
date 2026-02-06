# Sıfırdan Adversarial Derin Öğrenme Araştırmacısı Yetiştirme
## Ana Plan (Master Outline)

---

## 🎯 Hedef

Aşağıdaki iki araştırma makalesini bileşen bileşen anlayabilecek, uygulayabilecek ve benzer çalışmalar üretebilecek düzeye ulaşmak:

1. **Makale A:** *"A Comparative Study of Convolutional and Transformer Architectures Under Adversarial Perturbations: Gradient Characteristics and Transferability Analysis"*
2. **Makale B:** *"FE-ViT-ECL: Adversarially Robust Medical Image Forgery Detection via Frequency-Enhanced Vision Transformers with Explanation Consistency Loss"*

---

## 👤 Öğrenci Profili

- Lise mezunu; matematik, fizik ve temel fen bilgisi sağlam
- Bilgisayar bilimleri hakkında okuma düzeyinde genel kültür, programlama deneyimi yok
- Motivasyonu yüksek, akademik kariyer hedefliyor

---

## 📐 Pedagojik Yaklaşım

| İlke | Açıklama |
|------|----------|
| **Spiral müfredat** | Her parça bir öncekinin üzerine inşa edilir; aynı kavramlar giderek artan derinlikte tekrar edilir |
| **Analoji → Sezgi → Formalizm → Kod** | Her kavram bu dört aşamalı akışla sunulur |
| **İç referanslama** | Parçalar arası çapraz bağlantılar açıkça belirtilir |
| **Somut örnekler** | Soyut kavramlar günlük hayattan analojilerle başlar |
| **Teori-pratik dengesi** | Her bölüm "ne/neden" (teori) ve "nasıl" (kod) olarak iki katmanlıdır |

---

## 📚 Parça Yapısı Özeti (6 Parça)

```
Parça 1: Matematiksel Temeller & Programlamaya Giriş
   │
   ▼
Parça 2: Makine Öğrenmesi & Sinir Ağlarının Temelleri
   │
   ▼
Parça 3: Bilgisayarlı Görü & Derin Ağ Mimarileri
   │
   ▼
Parça 4: Adversarial Makine Öğrenmesi
   │
   ▼
Parça 5: Frekans Domain Analizi & Açıklanabilirlik
   │
   ▼
Parça 6: Entegrasyon — Hedef Makalelerin Anatomisi
```

---

## 📖 Parça Detayları

---

### PARÇA 1: Matematiksel Temeller & Programlamaya Giriş
**Tahmini Sayfa Sayısı:** 55–65 sayfa

#### 1.1 Giriş & Motivasyon
- Eğitim dizisinin genel haritası
- "Neden matematik? Neden programlama?" — hedef makalelerle bağlantı
- Bir adversarial örneğin ilk gösterimi (kara kutu olarak): "Bu pandayı gibbon olarak gören yapay zeka"

#### 1.2 Lineer Cebir Temelleri
- **Analojilerle giriş:** Vektörler → ok ve yön benzetmesi; matrisler → dönüşüm makinesi
- **Skaler, vektör, matris, tensör** hiyerarşisi
- Vektör işlemleri: toplama, skaler çarpım, iç çarpım (dot product), norm
- Matris işlemleri: çarpma, transpose, determinant, ters matris
- **Tensör kavramı** — derin öğrenmedeki veri yapılarına ilk köprü
- Özdeğer ve özvektör (eigenvalue/eigenvector) — sezgisel giriş
- 📐 *Makale bağlantısı:* "Bir görüntü aslında 3 boyutlu bir tensördür (yükseklik × genişlik × kanal)"

#### 1.3 Kalkülüs Temelleri
- **Analoji:** Türev → anlık hız; dağdan iniş benzetmesi
- Tek değişkenli türev: tanım, temel kurallar, zincir kuralı (chain rule)
- Kısmi türev (partial derivative) ve gradyan vektörü
- **Gradyan kavramı** — "en dik yokuş yönü"
- Gradyan inişi (gradient descent) — sezgisel açıklama
- 📐 *Makale bağlantısı:* "Adversarial saldırılar gradyanı kullanarak 'en kötü yönde' gider (Parça 4'te detay)"

#### 1.4 Olasılık & İstatistik Temelleri
- Olasılık uzayı, koşullu olasılık, Bayes teoremi
- Olasılık dağılımları: uniform, normal (Gauss), Bernoulli
- Beklenen değer, varyans, standart sapma
- Softmax fonksiyonu — olasılık dağılımına dönüşüm (sezgisel)
- 📐 *Makale bağlantısı:* "Sınıflandırma modellerinin çıktısı bir olasılık dağılımıdır"

#### 1.5 Python Programlamaya Giriş
- Kurulum: Python, Anaconda/pip, Jupyter Notebook
- Temel veri tipleri, kontrol yapıları, fonksiyonlar
- Listeler, sözlükler, döngüler
- Dosya okuma/yazma temelleri

#### 1.6 NumPy & Matplotlib
- NumPy dizileri (array) — vektör/matris/tensör karşılığı
- Temel NumPy işlemleri: oluşturma, dilimleme, broadcasting, lineer cebir
- Matplotlib ile veri görselleştirme: çizgi grafik, scatter plot, imshow (görüntü gösterimi)
- **Mini proje:** Rastgele bir görüntü oluştur, üzerine gürültü ekle, piksel dağılımını çiz

#### 1.7 Kod Atölyesi
- Lineer cebir işlemlerini NumPy ile uygulama
- Basit bir gradyan inişi simülasyonu (2D fonksiyon minimizasyonu)
- Görüntüyü matris olarak yükleyip manipüle etme

#### 1.8 Kavram Haritası & Sonraki Parçaya Köprü
- Kavram ilişki diyagramı
- "Bir sonraki parçada bu matematik araçlarını bir 'öğrenen sistem' inşa etmek için kullanacağız"

---

### PARÇA 2: Makine Öğrenmesi & Sinir Ağlarının Temelleri
**Tahmini Sayfa Sayısı:** 60–70 sayfa

#### 2.1 Giriş & Motivasyon
- "Makinelerin öğrenmesi ne demek?" — ezber değil genelleme
- Hedef makaleler bağlamında: "Bu modeller nasıl eğitildi ki saldırılabiliyorlar?"
- Parça 1'den köprü: "Gradyan kavramını artık bir öğrenme aracı olarak kullanacağız"

#### 2.2 Öğrenme Paradigmaları
- Gözetimli öğrenme (supervised learning): giriş-çıkış eşleşmesi
- Gözetimsiz öğrenme (unsupervised learning): kalıp bulma
- Pekiştirmeli öğrenme (reinforcement learning): deneme-yanılma
- 📐 *Makale bağlantısı:* "Her iki makaledeki modeller gözetimli öğrenme ile eğitilmiş sınıflandırıcılardır"

#### 2.3 Perceptron'dan Çok Katmanlı Ağlara
- **Analoji:** Tek nöron → bir karar kapısı; ağ → karar kapılarının hiyerarşisi
- Perceptron: lineer sınıflandırma, ağırlıklar ve eşik
- XOR problemi ve lineer ayrılabilirlik sınırlaması
- Çok katmanlı algılayıcı (MLP / Multi-Layer Perceptron)
- Gizli katmanlar ve temsil öğrenme

#### 2.4 Aktivasyon Fonksiyonları
- Neden doğrusal olmama (non-linearity) gerekli?
- Sigmoid, Tanh, ReLU, GELU — grafik ve özellik karşılaştırması
- 📐 *Makale bağlantısı:* "ViT'lerde GELU, CNN'lerde ReLU yaygın kullanılır (Parça 3'te detay)"

#### 2.5 Kayıp Fonksiyonları (Loss Functions)
- **Analoji:** Kayıp fonksiyonu → sınav notu; düşük kayıp → iyi öğrenme
- Mean Squared Error (MSE) — regresyon
- Cross-Entropy Loss — sınıflandırma
- Kayıp yüzeyinin şekli ve minimizasyon
- 📐 *Makale bağlantısı:* "Adversarial saldırılar kayıp fonksiyonunu MAKSİMİZE eder (Parça 4'te detay)"

#### 2.6 Geri Yayılım (Backpropagation)
- **Analoji:** Hata sinyalinin fabrikada geriye doğru iletilmesi
- Zincir kuralı (Parça 1) ile türev hesaplama
- Hesaplama grafı (computational graph) kavramı
- Adım adım örnek: 2 katmanlı ağda backpropagation
- 📐 *Makale bağlantısı:* "Adversarial gradyanlar tam olarak backpropagation ile hesaplanır"

#### 2.7 Optimizasyon Algoritmaları
- Stokastik Gradyan İnişi (SGD): mini-batch kavramı
- Momentum, learning rate kavramları
- Adam optimizer — adaptif öğrenme hızı
- Learning rate scheduling

#### 2.8 Aşırı Öğrenme & Düzenlileştirme (Overfitting & Regularization)
- Eğitim/doğrulama/test ayrımı
- Aşırı öğrenme belirtileri ve nedenleri
- Düzenlileştirme yöntemleri: L1/L2, Dropout, early stopping, data augmentation
- 📐 *Makale bağlantısı:* "Adversarial training bir düzenlileştirme tekniği olarak da görülebilir (Parça 4)"

#### 2.9 PyTorch'a Giriş
- Tensörler, autograd, temel yapı blokları
- `nn.Module`, `forward()`, eğitim döngüsü
- Dataset ve DataLoader kavramları

#### 2.10 Kod Atölyesi
- PyTorch ile MNIST üzerinde basit MLP sınıflandırıcı eğitme
- Eğitim/doğrulama kayıp eğrilerini çizme
- Modelin tahminlerini görselleştirme

#### 2.11 Kavram Haritası & Sonraki Parçaya Köprü
- Kavram ilişki diyagramı (Parça 1 kavramlarıyla bağlantılı)
- "Bir sonraki parçada bu sinir ağlarını görüntü verisine özelleştirilmiş mimarilerle güçlendireceğiz: CNN ve Transformer"

---

### PARÇA 3: Bilgisayarlı Görü & Derin Ağ Mimarileri
**Tahmini Sayfa Sayısı:** 65–75 sayfa

#### 3.1 Giriş & Motivasyon
- "Bir bilgisayar görüntüyü nasıl görür?" — piksel, kanal, çözünürlük
- Hedef makaleler bağlamında: "Her iki makale de CNN ve ViT mimarilerini karşılaştırıyor"
- Parça 2'den köprü: "MLP her pikseli bağımsız girdi olarak alır — bu çok verimsiz. Daha akıllı bir yol var mı?"

#### 3.2 Dijital Görüntü Temsili
- Gri tonlama ve renkli görüntüler (RGB)
- Piksel değerleri, normalizasyon, veri ön işleme
- Yaygın veri setleri: MNIST, CIFAR-10, ImageNet

#### 3.3 Evrişim (Convolution) İşlemi
- **Analoji:** Büyüteçle resmi tarama → yerel örüntü yakalama
- Evrişim çekirdeği (kernel/filter), stride, padding
- Özellik haritaları (feature maps)
- Havuzlama (pooling): max pooling, average pooling

#### 3.4 CNN Mimarileri — Tarihsel Gelişim
- **LeNet-5** (1998): Temel CNN yapısı
- **AlexNet** (2012): Derin öğrenme devrimi
- **VGGNet** (2014): Derinliğin gücü
- **ResNet** (2015): Artık bağlantılar (skip connections) ve vanishing gradient çözümü
- **EfficientNet** (2019): Model ölçekleme stratejisi
- Her mimari için: temel fikir, yapısal yenilik, parametre sayısı
- 📐 *Makale bağlantısı:* "Makale A, ResNet ve EfficientNet'i adversarial saldırılara karşı test eder"

#### 3.5 Dikkat Mekanizması (Attention Mechanism)
- **Analoji:** Bir cümlede veya resimde "neye odaklanmalıyım?" sorusu
- Query, Key, Value üçlüsü — kütüphanede kitap arama analojisi
- Ölçekli iç çarpım dikkat (Scaled Dot-Product Attention)
- Çok başlı dikkat (Multi-Head Attention)
- Matematiksel formülasyon: Attention(Q, K, V) = softmax(QK^T / √d_k) V

#### 3.6 Transformer Mimarisi
- Orijinal Transformer: encoder-decoder yapısı
- Konumsal kodlama (positional encoding) — "sıra bilgisi nasıl korunur?"
- Katman normalizasyonu (Layer Normalization) ve artık bağlantılar
- Feed-Forward ağ blokları

#### 3.7 Vision Transformer (ViT)
- **Temel fikir:** Görüntüyü yamalara (patches) böl, her yamayı bir "kelime" gibi işle
- Yama gömme (patch embedding), sınıf tokeni ([CLS])
- ViT mimarisi detayı: katmanlar, başlıklar, gömme boyutları
- ViT'nin eğitim veri ihtiyacı ve transfer öğrenme
- 📐 *Makale bağlantısı:* "Makale B, FE-ViT-ECL mimarisinin çekirdeğinde ViT kullanır"

#### 3.8 CNN vs. Transformer Karşılaştırması
- Tümevarımsal önyargı (inductive bias): yerellik vs. küresel bağlam
- Receptive field farkları
- Hesaplama maliyeti karşılaştırması
- Hibrit mimariler (CNN + Transformer)
- 📐 *Makale bağlantısı:* "Makale A'nın temel araştırma sorusu: Bu iki mimari adversarial saldırılara nasıl farklı tepki verir?"

#### 3.9 Transfer Öğrenme & Önceden Eğitilmiş Modeller
- Önceden eğitilmiş ağırlıklar (pre-trained weights) kavramı
- İnce ayar (fine-tuning) stratejileri
- torchvision ve timm kütüphaneleri

#### 3.10 Kod Atölyesi
- PyTorch ile basit CNN inşası ve CIFAR-10 üzerinde eğitim
- Önceden eğitilmiş ResNet ile transfer öğrenme
- Basit bir ViT modelinin yapısını inceleme (timm kütüphanesi)
- CNN ve ViT özellik haritalarını / dikkat haritalarını görselleştirme

#### 3.11 Kavram Haritası & Sonraki Parçaya Köprü
- CNN → ViT → Hibrit mimariler ilişki haritası
- "Bu güçlü modellerin bir zayıflığı var: adversarial örnekler. Bir sonraki parçada bu zayıflığı sistematik olarak inceleyeceğiz."

---

### PARÇA 4: Adversarial Makine Öğrenmesi
**Tahmini Sayfa Sayısı:** 70–80 sayfa

#### 4.1 Giriş & Motivasyon
- "İnsan gözüyle aynı görünen ama yapay zekayı kandıran görüntüler" — etkileyici örnekler
- Adversarial ML'nin tarihçesi: Szegedy et al. (2013) keşfi
- Hedef makaleler bağlamında: "Bu parça her iki makalenin doğrudan konusudur"
- Parça 2-3'ten köprü: "Backpropagation'ı öğrenme için kullandık; şimdi aynı aracı saldırı için kullanacağız"

#### 4.2 Adversarial Örnekler: Tanım ve Formalizasyon
- Pertürbasyon (δ) kavramı ve norm kısıtları (L₀, L₂, L∞)
- **Analoji:** "Bir resme gözle görülmeyecek kadar ince toz serpme"
- Matematiksel tanım: x_adv = x + δ, ||δ||_p ≤ ε
- Hedefli (targeted) vs. hedefsiz (untargeted) saldırılar

#### 4.3 Tehdit Modelleri
- Beyaz kutu (white-box): model mimarisine ve ağırlıklarına tam erişim
- Siyah kutu (black-box): sadece girdi-çıktı erişimi
- Gri kutu (gray-box): kısmi bilgi
- 📐 *Makale bağlantısı:* "Makale A hem beyaz kutu hem transferability (siyah kutu proxy) analizi yapar"

#### 4.4 Saldırı Yöntemleri — Derinlemesine

##### 4.4.1 FGSM (Fast Gradient Sign Method)
- Goodfellow et al. (2014)
- Tek adımda gradyan işareti: x_adv = x + ε · sign(∇_x L(θ, x, y))
- **Analoji:** "Kayıp yüzeyinde en dik yokuşun yönüne bir adım at"
- Hızlı ama kaba; adım adım türetme

##### 4.4.2 PGD (Projected Gradient Descent)
- Madry et al. (2018)
- FGSM'nin iteratif versiyonu: küçük adımlarla ilerleme ve ε-topu içine projeksiyon
- İç maksimizasyon problemi: max_{δ∈S} L(θ, x+δ, y)
- 📐 *Makale bağlantısı:* "Her iki makale de PGD'yi temel saldırı yöntemi olarak kullanır"

##### 4.4.3 C&W (Carlini & Wagner)
- Carlini & Wagner (2017)
- Optimizasyon tabanlı saldırı: pertürbasyonu minimize ederken yanlış sınıflandırma sağla
- f(x') ≤ 0 kısıtının Lagrangian formülasyonu
- Farklı norm versiyonları (L₀, L₂, L∞)

##### 4.4.4 AutoAttack
- Croce & Hein (2020)
- Ensemble saldırı: APGD-CE, APGD-DLR, FAB, Square Attack bileşenleri
- Parametresiz değerlendirme standardı
- 📐 *Makale bağlantısı:* "Makale A'da robustness değerlendirme için AutoAttack kullanılır"

#### 4.5 Gradyan Karakteristikleri Analizi
- CNN ve Transformer modellerinde gradyan dağılımı farkları
- Gradyan büyüklüğü, yönü ve düzgünlüğü (smoothness)
- Gradyan tabanlı saldırıların etkinliğini belirleyen faktörler
- 📐 *Makale bağlantısı:* "Makale A'nın temel katkısı: mimari farklılıkların gradyan karakteristiklerini nasıl etkilediği"

#### 4.6 Transferability (Aktarılabilirlik)
- Tanım: Bir model için üretilen adversarial örneğin başka bir modeli de kandırabilmesi
- Transferability'yi etkileyen faktörler: mimari benzerlik, eğitim verisi, saldırı gücü
- CNN→CNN, CNN→ViT, ViT→CNN, ViT→ViT transferability kalıpları
- 📐 *Makale bağlantısı:* "Makale A'nın ikinci temel katkısı: CNN-Transformer arası transferability analizi"

#### 4.7 Adversarial Savunma Mekanizmaları
- **Adversarial eğitim (Adversarial Training):** Eğitim setine adversarial örnekler ekleme
- **Giriş dönüşümleri (Input Transformation):** JPEG sıkıştırma, bit derinliği azaltma, randomize düzleştirme
- **Sertifikalı savunmalar (Certified Defenses):** Randomized smoothing, provable robustness
- **Model ensembling ve çeşitlendirme**
- 📐 *Makale bağlantısı:* "Makale B, adversarial training ile robustness sağlar"

#### 4.8 Robustness Değerlendirme Metrikleri
- Clean accuracy vs. robust accuracy
- Pertürbasyon bütçesi (ε) altında doğruluk
- Attack success rate (ASR)
- Değerlendirme tuzakları: gradient masking, obfuscated gradients

#### 4.9 Kod Atölyesi
- FGSM saldırısı: PyTorch ile adım adım uygulama
- PGD saldırısı: iteratif versiyon
- Farklı ε değerlerinde saldırı etkisini görselleştirme
- Basit adversarial training döngüsü
- CNN ve ViT üzerinde karşılaştırmalı saldırı deneyi

#### 4.10 Kavram Haritası & Sonraki Parçaya Köprü
- Saldırı-savunma ekosistemi haritası
- "Adversarial pertürbasyonları daha iyi anlamak için başka bir perspektife ihtiyacımız var: frekans domain. Ayrıca modellerin 'neden' böyle karar verdiğini anlamamız gerekiyor: açıklanabilirlik."

---

### PARÇA 5: Frekans Domain Analizi & Açıklanabilirlik (Explainability)
**Tahmini Sayfa Sayısı:** 60–70 sayfa

#### 5.1 Giriş & Motivasyon
- "Adversarial pertürbasyonlar görüntünün hangi bileşenlerini bozuyor?" — frekans perspektifi
- "Model neden yanlış karar veriyor?" — açıklanabilirlik perspektifi
- Parça 4'ten köprü: "Saldırıları piksel uzayında gördük; şimdi frekans uzayında da analiz edeceğiz"

#### 5.2 Sinyal İşleme Temelleri
- **Analoji:** Ses → frekanslar (bas/tiz); görüntü → frekanslar (düz alanlar/kenarlar)
- Frekans kavramı görüntülerde: düşük frekans (düz bölgeler, genel şekiller) vs. yüksek frekans (kenarlar, dokular, gürültü)

#### 5.3 Fourier Dönüşümü (Fourier Transform)
- Joseph Fourier'in fikri: her sinyal sinüslerin toplamıdır
- 1D Fourier Transform: matematiksel tanım
- 2D Fourier Transform (görüntüler için): F(u,v)
- Genlik spektrumu (magnitude spectrum) ve faz spektrumu (phase spectrum)
- **Hızlı Fourier Dönüşümü (FFT):** Hesaplama verimliliği
- Görüntü üzerinde FFT uygulaması ve yorumlama

#### 5.4 Ayrık Kosinüs Dönüşümü (DCT — Discrete Cosine Transform)
- FFT ile karşılaştırma: neden DCT?
- JPEG sıkıştırmadaki rolü
- DCT katsayıları ve frekans bant analizi
- 📐 *Makale bağlantısı:* "Makale B, DCT tabanlı frekans zenginleştirme (frequency enhancement) kullanır"

#### 5.5 Frekans Domain'de Adversarial Pertürbasyon Analizi
- Adversarial pertürbasyonların frekans dağılımı
- CNN vs. ViT: hangi frekans bantlarına duyarlı?
- Yüksek frekans bileşenlerinin adversarial saldırılardaki rolü
- Frekans tabanlı savunma mekanizmaları
- 📐 *Makale bağlantısı:* "Makale B'deki FE (Frequency Enhancement) modülü tam olarak bu analize dayanır"

#### 5.6 Açıklanabilirlik (Explainability) — Neden Önemli?
- Kara kutu modeller ve güven problemi
- Özellikle medikal yapay zeka bağlamında: "Doktor modelin neye bakarak karar verdiğini bilmeli"
- Açıklanabilirlik türleri: post-hoc vs. ante-hoc, yerel vs. küresel

#### 5.7 Grad-CAM (Gradient-weighted Class Activation Mapping)
- Temel fikir: son evrişim katmanının gradyanlarını kullanarak "dikkat haritası" üretme
- Matematiksel formülasyon
- CNN'lerde uygulama
- Sınırlılıklar ve varyantlar (Grad-CAM++, Score-CAM)
- 📐 *Makale bağlantısı:* "Her iki makale de Grad-CAM'i görselleştirme aracı olarak kullanır"

#### 5.8 Dikkat Haritaları (Attention Maps)
- ViT'lerde dikkat ağırlıklarının görselleştirilmesi
- Katman ve başlık bazında dikkat analizi
- Attention rollout ve attention flow yöntemleri
- CNN Grad-CAM vs. ViT Attention Map karşılaştırması
- 📐 *Makale bağlantısı:* "Makale A, dikkat haritalarını adversarial pertürbasyon altında karşılaştırır"

#### 5.9 SHAP (SHapley Additive exPlanations) — Kısa Giriş
- Oyun teorisi temelli açıklama: her özelliğin katkısı
- Derin öğrenme için DeepSHAP
- Grad-CAM ve Attention Map ile karşılaştırma

#### 5.10 Explanation Consistency (Açıklama Tutarlılığı)
- Tanım: clean ve adversarial girdi için açıklamaların benzer olması gerekliliği
- Neden önemli: sahte (forgery) bölgelerin tespitinde tutarlılık
- Explanation Consistency Loss formülasyonu
- 📐 *Makale bağlantısı:* "Makale B'deki ECL (Explanation Consistency Loss) tam olarak bu kavram üzerine kuruludur"

#### 5.11 Medikal Görüntü Analizi Bağlamı
- Medikal görüntü türleri: X-ray, CT, MRI, patoloji
- Medikal görüntü sahteciliği (forgery): motivasyon ve tehditler
- Veri seti özellikleri, etiketleme zorlukları, etik hususlar
- Klinik gereklilikler: duyarlılık, özgüllük, açıklanabilirlik
- 📐 *Makale bağlantısı:* "Makale B'nin uygulama alanı medikal görüntü sahtecilik tespiti"

#### 5.12 Kod Atölyesi
- FFT/DCT ile görüntü frekans analizi
- Adversarial pertürbasyonun frekans spektrumunu görselleştirme
- Grad-CAM uygulaması (PyTorch)
- ViT dikkat haritası çıkarma ve görselleştirme
- Clean vs. adversarial girdi için açıklama tutarlılığını karşılaştırma

#### 5.13 Kavram Haritası & Sonraki Parçaya Köprü
- Frekans analizi + Açıklanabilirlik + Adversarial ML bağlantı haritası
- "Artık tüm yapı taşlarına sahibiz. Son parçada bunları birleştirerek hedef makaleleri bileşen bileşen çözeceğiz."

---

### PARÇA 6: Entegrasyon — Hedef Makalelerin Anatomisi
**Tahmini Sayfa Sayısı:** 55–65 sayfa

#### 6.1 Giriş & Motivasyon
- "5 parça boyunca öğrendiğimiz her şey bu iki makaleyi anlamamız içindi"
- Bu parçanın amacı: makaleleri okumak, anlamak, eleştirmek ve benzer çalışma tasarlamak

#### 6.2 Akademik Makale Okuma Metodolojisi
- Bir araştırma makalesinin bölümleri: abstract, introduction, related work, method, experiments, discussion, conclusion
- İlk okuma, ikinci okuma, eleştirel okuma stratejileri
- Not alma ve anahtar bilgi çıkarma teknikleri

#### 6.3 Makale A: CNN vs. Transformer Adversarial Karşılaştırma
**Bileşen bileşen analiz:**

##### 6.3.1 Araştırma Sorusu ve Hipotezler
- Temel soru: CNN ve Transformer mimarileri adversarial pertürbasyonlara nasıl farklı tepki verir?
- Alt sorular: gradyan karakteristikleri, transferability kalıpları

##### 6.3.2 Deney Tasarımı
- Kullanılan modeller (Parça 3 bağlantısı)
- Kullanılan saldırı yöntemleri (Parça 4 bağlantısı)
- Veri setleri ve değerlendirme metrikleri
- Kontrol değişkenleri ve deneysel protokol

##### 6.3.3 Sonuçların Yorumlanması
- Gradyan analizi bulguları
- Transferability matrisi yorumlama
- Mimari farklılıkların adversarial robustness'a etkisi
- Bulguların daha geniş bağlamda anlamı

#### 6.4 Makale B: FE-ViT-ECL Mimarisi
**Bileşen bileşen analiz:**

##### 6.4.1 Problem Tanımı
- Medikal görüntü sahteciliği tehdidi
- Mevcut yöntemlerin sınırlılıkları
- Adversarial robustness + açıklanabilirlik + frekans analizi birleşimi

##### 6.4.2 FE-ViT-ECL Mimarisinin Yapı Taşları
- **FE (Frequency Enhancement):** DCT tabanlı frekans zenginleştirme modülü (Parça 5.4)
- **ViT:** Vision Transformer omurgası (Parça 3.7)
- **ECL (Explanation Consistency Loss):** Açıklama tutarlılığı kaybı (Parça 5.10)
- Bu bileşenlerin nasıl entegre edildiği — mimari diyagram analizi

##### 6.4.3 Eğitim Stratejisi
- Adversarial training süreci
- Multi-task loss: sınıflandırma kaybı + ECL
- Hiperparametre seçimleri ve gerekçeleri

##### 6.4.4 Deneysel Sonuçlar
- Clean accuracy vs. robust accuracy sonuçları
- Ablation study: her bileşenin katkısı
- Karşılaştırmalı analiz: baseline modeller vs. FE-ViT-ECL

#### 6.5 İki Makale Arasındaki Bağlantılar
- Ortak kavramlar ve perspektifler
- Makale A'nın bulguları Makale B'nin tasarımını nasıl bilgilendirir?
- Birlikte ele alındığında ortaya çıkan büyük resim

#### 6.6 Eleştirel Okuma ve Gelecek Çalışma Fikirleri
- Her iki makalenin güçlü ve zayıf yönleri
- Potansiyel genişletme yönleri
- Araştırma sorusu formüle etme alıştırması

#### 6.7 Araştırma Metodolojisi Temelleri
- Literatür taraması nasıl yapılır?
- Araştırma sorusu nasıl formüle edilir?
- Deney tasarımı ilkeleri
- Akademik yazım temelleri: LaTeX, BibTeX, şekil/tablo hazırlama
- Yayın süreci: hakemli dergi, konferans, arXiv

#### 6.8 Kod Atölyesi — Kapanış Projesi
- Makale A'dan bir deneyin replikasyonu: CNN vs. ViT üzerinde FGSM/PGD saldırısı ve karşılaştırma
- Makale B'den bir bileşenin uygulaması: DCT tabanlı frekans zenginleştirme veya basit ECL
- Sonuçları görselleştirme ve mini rapor yazma

#### 6.9 Genel Kavram Haritası (Tüm Parçalar)
- 6 parçanın tüm kavramlarını birleştiren büyük kavram haritası
- Öğrenme yolculuğunun özeti

#### 6.10 Sonraki Adımlar
- Önerilen ileri okuma listesi
- Takip edilmesi gereken konferanslar ve dergiler (NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, IEEE TIFS)
- Açık kaynak araçlar ve kaynaklar (RobustBench, CleverHans, ART, Foolbox)
- Araştırma topluluklarına katılım

---

## 📊 Sayfa Dağılımı Özeti

| Parça | Konu | Tahmini Sayfa |
|-------|------|---------------|
| 1 | Matematiksel Temeller & Programlama | 55–65 |
| 2 | Makine Öğrenmesi & Sinir Ağları | 60–70 |
| 3 | Bilgisayarlı Görü & Derin Mimariler | 65–75 |
| 4 | Adversarial Makine Öğrenmesi | 70–80 |
| 5 | Frekans Analizi & Açıklanabilirlik | 60–70 |
| 6 | Entegrasyon & Makale Anatomisi | 55–65 |
| **Toplam** | | **365–425** |

---

## 📋 Parçalar Arası Bağımlılık Matrisi

```
         P1    P2    P3    P4    P5    P6
P1  [  -  ] [    ] [    ] [    ] [    ] [    ]
P2  [ ██ ] [  -  ] [    ] [    ] [    ] [    ]
P3  [ █  ] [ ██ ] [  -  ] [    ] [    ] [    ]
P4  [ █  ] [ ██ ] [ ██ ] [  -  ] [    ] [    ]
P5  [ █  ] [ █  ] [ █  ] [ ██ ] [  -  ] [    ]
P6  [ █  ] [ █  ] [ ██ ] [ ██ ] [ ██ ] [  -  ]

██ = Güçlü bağımlılık (doğrudan ön koşul)
█  = Zayıf bağımlılık (kavramsal referans)
```

---

## 🔤 Notasyon ve Terminoloji Sözleşmesi

Tüm parçalarda tutarlı olarak kullanılacak notasyon:

| Sembol | Anlam |
|--------|-------|
| x | Girdi verisi (görüntü) |
| y | Gerçek etiket |
| ŷ | Model tahmini |
| θ | Model parametreleri |
| f_θ(x) | Model fonksiyonu |
| L(θ, x, y) | Kayıp fonksiyonu |
| ∇_x | x'e göre gradyan |
| δ | Adversarial pertürbasyon |
| ε | Pertürbasyon bütçesi |
| x_adv | Adversarial örnek (x + δ) |
| ‖·‖_p | L_p normu |

---

## 📝 Her Parçanın İç Yapı Şablonu

Her parça şu bölümleri içerecek:

1. **Giriş & Motivasyon** — Bu parçada ne öğreneceğiz? Hedef makaleler bağlamında neden önemli?
2. **Konu Anlatımları** — Analoji → Sezgi → Formalizm → Kod akışıyla
3. **Ara Özetler** — Her ana bloğun sonunda "öğrendiklerimiz" kutusu
4. **Uygulama / Kod Atölyesi** — Parçanın konularını birleştiren pratik çalışma
5. **Kavram Haritası** — O parçadaki ve önceki parçalarla bağlantılı kavram ilişki diyagramı
6. **Sonraki Parçaya Köprü** — Motivasyon oluşturan geçiş paragrafı

---

## ✅ Üretim Sırası

1. ~~Ana Plan (bu doküman) → Onay~~
2. Parça 1: Matematiksel Temeller & Programlamaya Giriş
3. Parça 2: Makine Öğrenmesi & Sinir Ağlarının Temelleri
4. Parça 3: Bilgisayarlı Görü & Derin Ağ Mimarileri
5. Parça 4: Adversarial Makine Öğrenmesi
6. Parça 5: Frekans Domain Analizi & Açıklanabilirlik
7. Parça 6: Entegrasyon — Hedef Makalelerin Anatomisi

---

*Bu plan onayınıza sunulmuştur. Gerekli gördüğünüz değişiklikleri belirtiniz; onay sonrası Parça 1'in üretimine başlanacaktır.*
