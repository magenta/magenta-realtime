# Magenta RT Teknik Raporu

**Özet**

Bu rapor, Magenta RT sisteminin teknik ayrıntılarını açıklamaktadır. Magenta RT, metin veya ses istemleriyle yönlendirilebilen, gerçek zamanlı, sürekli müzik sesi üretimi için tasarlanmış bir Python kitaplığıdır. Sistem üç ana bileşenden oluşur: SpectroStream ses kodeği, MusicCoCa stil gömme modeli ve bu ikisini birleştiren bir Transformer tabanlı Büyük Dil Modeli (LLM). Bu rapor, her bir bileşenin mimarisini, yapılandırmasını, birbirleriyle nasıl etkileşimde bulunduğunu ve genel sistemin nasıl çalıştığını detaylandıracaktır. Magenta RT, Google DeepMind tarafından geliştirilmiştir ve MusicFX DJ Modu ile Lyria RealTime API gibi daha büyük sistemlerin temelini oluşturan araştırma ve teknolojiyi kullanır.

**1. Giriş**

Müzik, insan kültürünün temel bir parçasıdır ve teknolojinin ilerlemesiyle birlikte müzik oluşturma ve deneyimleme biçimlerimiz de sürekli olarak gelişmektedir. Son yıllarda, yapay zeka (YZ) ve özellikle derin öğrenme, müzik üretimi alanında devrim niteliğinde yenilikler sunmuştur. Magenta RT, bu yeniliklerin bir ürünü olarak, kullanıcıların yerel cihazlarında akıcı ve sürekli müzik sesini gerçek zamanlı olarak üretebilmelerini sağlayan açık kaynaklı bir Python kitaplığıdır.

Gerçek zamanlı müzik üretimi, özellikle canlı performanslar, etkileşimli enstalasyonlar veya dinamik oyun müzikleri gibi uygulamalar için önemli zorluklar sunar. Bu zorluklar arasında düşük gecikme süresiyle yüksek kaliteli ses üretimi, kullanıcı girdilerine anında yanıt verebilme ve üretilen müziğin tutarlılığını ve müzikalitesini koruma ihtiyacı bulunmaktadır. Magenta RT, bu zorluklara SpectroStream adlı verimli bir ses kodeği, MusicCoCa adlı esnek bir stil gömme modeli ve bu iki bileşeni akıllıca yöneten güçlü bir Transformer tabanlı Büyük Dil Modeli (LLM) aracılığıyla çözümler sunmayı amaçlamaktadır.

Magenta RT, Google'ın [MusicFX DJ Mode](https://labs.google/fx/tools/music-fx-dj) ve [Lyria RealTime API](http://goo.gle/lyria-realtime) gibi daha kapsamlı müzik üretim araçlarının arkasındaki araştırma ve teknolojiyi temel alır. Bu araçların açık kaynaklı bir tamamlayıcısı olarak, araştırmacılara, geliştiricilere ve sanatçılara gerçek zamanlı müzik üretimi teknolojilerini keşfetme ve kendi projelerinde kullanma imkanı sunar.

Bu rapor, Magenta RT sisteminin derinlemesine bir teknik analizini sunmaktadır. İlk olarak, ilgili çalışmalara ve arka plan bilgilerine değinilecek, ardından sistemin genel mimarisi ve üç ana bileşeni (SpectroStream, MusicCoCa ve LLM) ayrıntılı olarak incelenecektir. Son olarak, sistemin genel işleyişi, eğitim detayları, potansiyel kullanım alanları, bilinen sınırlamalar ve gelecekteki çalışma olasılıkları tartışılacaktır.

**2. Arka Plan ve İlgili Çalışmalar**

Magenta RT'nin geliştirilmesi, müzik üretimi, ses temsili ve koşullu dizi modellemesi alanlarındaki önemli ilerlemelere dayanmaktadır.

*   **Müzik Üretimi Modelleri:** Derin öğrenmenin müzik üretimine uygulanması, örnek tabanlı sentezden (örneğin, [WaveNet](https://arxiv.org/abs/1609.03499) – van den Oord et al., 2016) sembolik müzik üretimine (örneğin, [Music Transformer](https://arxiv.org/abs/1809.04281) – Huang et al., 2018) ve daha yakın zamanda doğrudan ses üreten büyük ölçekli modellere (örneğin, [Jukebox](https://openai.com/blog/jukebox/) – Dhariwal et al., 2020; [MusicLM](https://arxiv.org/abs/2301.11325) – Agostinelli et al., 2023) kadar geniş bir yelpazede çalışmalar üretmiştir. Magenta RT, özellikle MusicLM'in metin koşullu ses üretimi yaklaşımından ilham alır, ancak gerçek zamanlı ve akışlı üretime odaklanır.

*   **Ses Temsilleri ve Kodekler:** Ham ses dalga formları yüksek boyutlu olduğundan, etkili bir temsil öğrenmek veya ayrık belirteçlere sıkıştırmak, özellikle Transformer gibi modellere giriş olarak kullanıldığında önemlidir. [SoundStream](https://arxiv.org/abs/2107.03312) (Zeghidour et al., 2021) ve onun üzerine inşa edilen RVQ (Residual Vector Quantization) tabanlı yaklaşımlar, yüksek kaliteli sesi düşük bit hızlarında temsil etme konusunda önemli başarılar elde etmiştir. Magenta RT'nin SpectroStream kodeği de bu tür bir RVQ tabanlı ses belirtekleme yaklaşımını benimser. Benzer şekilde, [EnCodec](https://arxiv.org/abs/2210.13438) (Défossez et al., 2022) de sinirsel ses sıkıştırma alanında dikkate değer bir çalışmadır.

*   **Metin ve Sesten Ortak Gömme Modelleri:** Müziği metinle veya diğer seslerle koşullandırmak, kullanıcıların üretilen müziğin stilini veya içeriğini yönlendirmesine olanak tanır. [CoCa (Contrastive Captioner)](https://arxiv.org/abs/2205.01917) (Yu et al., 2022) ve [CLAP (Contrastive Language-Audio Pretraining)](https://arxiv.org/abs/2211.06687) (Elizalde et al., 2022) gibi modeller, farklı modalitelerdeki (örneğin, metin ve ses) verileri ortak bir gömme alanına eşlemek için kontrastif öğrenmeyi kullanır. MusicCoCa, bu tür bir yaklaşımı benimseyerek hem metin açıklamalarından hem de referans ses kliplerinden stil gömmeleri oluşturur, [Huang et al. (2022)](https://arxiv.org/abs/2208.12415) çalışmasındaki fikirlere dayanır.

*   **Transformer Mimarileri ve LLM'ler:** [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) mimarisi, dikkat mekanizması sayesinde uzun menzilli bağımlılıkları modellemedeki başarısıyla dizi modellemede baskın hale gelmiştir. Büyük Dil Modelleri (LLM'ler), metin üretimi başta olmak üzere birçok alanda çığır açmıştır ve müzik gibi diğer sıralı veri türlerine de başarıyla uygulanmıştır. Magenta RT, SpectroStream tarafından üretilen ses belirteçlerini ve MusicCoCa tarafından üretilen stil belirteçlerini işlemek ve yeni ses belirteçleri üretmek için bir encoder-decoder Transformer LLM kullanır.

Bu çalışmalar, Magenta RT'nin üzerine inşa edildiği temeli oluşturur ve sistemin her bir bileşeninin tasarım kararlarını etkiler.

**3. Sistem Mimarisi**

Magenta RT, modüler bir tasarıma sahip olup üç ana bileşenden oluşur: (1) SpectroStream ses kodeği, (2) MusicCoCa stil gömme modeli ve (3) bu iki modelden gelen bilgileri birleştirerek sürekli müzik üreten bir Büyük Dil Modeli (LLM).

Genel akış `notebooks/diagram.gif` dosyasında görselleştirilmiştir. Temel etkileşim şu şekildedir: Kullanıcıdan bir stil istemi (metin veya örnek bir ses dosyası) alınır. Bu istem, MusicCoCa tarafından işlenerek bir stil gömmesine veya stil belirteçlerine dönüştürülür. Aynı zamanda, sistem bir ses bağlamı tutar; bu, daha önce üretilmiş veya sağlanmış olan sesin SpectroStream tarafından ayrık belirteçlere dönüştürülmüş halidir. LLM, bu stil belirteçlerini ve bağlam belirteçlerini alarak bir sonraki ses parçasının belirteçlerini üretir. Son olarak, bu yeni üretilen belirteçler SpectroStream kod çözücüsü tarafından tekrar duyulabilir sese dönüştürülür. Bu süreç, sürekli bir müzik akışı oluşturmak için tekrarlanır.

**3.1. SpectroStream: Ses Kodeği**

SpectroStream, Magenta RT'nin temel ses işleme bileşenidir. Ana görevi, yüksek sadakatli stereo sesi (varsayılan olarak 48kHz) ayrık bir belirteç dizisine (veya gömmelere) dönüştürmek (kodlama) ve bu belirteç dizisinden orijinal sesi mümkün olduğunca doğru bir şekilde yeniden yapılandırmaktır (kod çözme).

*   **Amaç:**
    *   Sesi, LLM tarafından işlenebilecek sıkıştırılmış, ayrık bir temsile dönüştürmek.
    *   Bu temsilden yüksek kaliteli sesi yeniden oluşturmak.

*   **Mimari:** SpectroStream, [SoundStream](https://arxiv.org/abs/2107.03312) (Zeghidour et al., 2021) çalışmasından ilham alan bir artık vektör niceleme (Residual Vector Quantization - RVQ) tabanlı sinirsel ses kodeğidir. Üç bölümden oluşur:
    1.  **Kodlayıcı (Encoder):** Ses dalga formunu bir dizi sürekli gömme vektörüne dönüştürür.
    2.  **Nicelendirici (Quantizer):** Sürekli gömme vektörlerini RVQ kullanarak ayrık belirteçlere niceler. `MODEL.md`'ye göre, SpectroStream 64 RVQ derinliğine ve her kod kitabında 1024 (10 bit) koda sahiptir. Bu, saniyede 25 kare (frame) hızında yaklaşık 16 kbps'lik bir bit hızıyla sonuçlanır.
    3.  **Kod Çözücü (Decoder):** Ayrık belirteçlerden ses dalga formunu yeniden oluşturur.

*   **Yapılandırma (`magenta_rt.spectrostream.SpectroStreamConfiguration`):**
    *   `sample_rate`: 48000 Hz
    *   `num_channels`: 2 (stereo)
    *   `frame_rate`: 25.0 Hz (her çerçeve 40ms veya 1920 ses örneği @ 48kHz)
    *   `embedding_dim`: 256
    *   `rvq_depth`: 64 (maksimum)
    *   `rvq_codebook_size`: 1024

*   **Giriş/Çıkış:**
    *   `encode(waveform)`: `audio.Waveform` -> `(çerçeve_sayısı, rvq_derinliği)` şeklinde akustik belirteçler.
    *   `decode(tokens)`: Akustik belirteçler -> `audio.Waveform`.

*   **Kullanılan Modeller:** `SpectroStreamSavedModel` (TensorFlow) ve `SpectroStreamJAX` (JAX için). Modeller `savedmodels/ssv2_48k_stereo/` altında bulunur.

**3.2. MusicCoCa: Stil Gömme Modeli**

MusicCoCa, üretilecek müziğin stilini tanımlamak için kullanılır, metin veya ses girdilerini ortak bir gömme alanına eşler.

*   **Amaç:**
    *   Stil tanımlayıcılarını sayısal bir temsile (gömme vektörü) dönüştürmek.
    *   Farklı stillerin karıştırılmasına olanak tanımak.

*   **Mimari:** [CoCa (Yu+ 22)](https://arxiv.org/abs/2205.01917) ve [Huang+ 22](https://arxiv.org/abs/2208.12415) çalışmalarına dayanan kontrastif eğitimli bir modeldir.
    1.  **Metin Kodlayıcı**
    2.  **Ses Kodlayıcı**
    3.  **Kontrastif Öğrenme**
    4.  **RVQ ile Niceleme:** Stil gömmeleri, `tokenize` yöntemiyle ayrık belirteçlere nicelenebilir.

*   **Yapılandırma (`magenta_rt.musiccoca.MusicCoCaConfiguration`):**
    *   `sample_rate`: 16000 Hz (ses girişi için)
    *   `clip_length`: 10.0 saniye
    *   `embedding_dim`: 768
    *   `rvq_depth`: 12
    *   `rvq_codebook_size`: 1024

*   **Giriş/Çıkış:**
    *   `embed(text_or_audio)`: `str` veya `audio.Waveform` -> `(embedding_dim,)` stil gömmesi.
    *   `tokenize(embeddings)`: Stil gömmesi -> `(rvq_depth,)` stil belirteçleri.

*   **Kullanılan Model:** `MusicCoCaV212F`. Modeller `savedmodels/musiccoca_mv212f_cpu_compat` ve `savedmodels/musiccoca_mv212_quant` altında bulunur.

**3.3. Büyük Dil Modeli (LLM): Koşullu Ses Belirteci Üretimi**

LLM, SpectroStream'den gelen ses bağlamını ve MusicCoCa'dan gelen stil bilgisini alarak yeni ses belirteçleri üreten bir encoder-decoder Transformer mimarisine (T5X tabanlı) sahiptir.

*   **Amaç:** Verilen ses bağlamını devam ettirerek ve sağlanan stile uyarak yeni ses belirteçleri üretmek.

*   **Giriş Hazırlığı (`MagentaRTT5X.generate_chunk` içinde):**

    Aşağıda LLM'e giren belirteçlerin basitleştirilmiş bir akışı verilmiştir:

    ```
    Kullanıcı Stili (Metin/Ses) -> MusicCoCa -> Stil Gömme -> MusicCoCa.tokenize() -> Stil RVQ Belirteçleri (12 derinlik)
                                                                                       |
                                                                                       v
    Geçmiş Ses -> SpectroStream.encode() -> Bağlam RVQ Belirteçleri (16 derinlik)      LLM Encoder için Stil Belirteçleri (ilk 6 katman)
        |                                                                              (utils.rvq_to_llm ile vocab'a eşlenir)
        v
    LLM Encoder için Bağlam Belirteçleri (ilk 4 katman)
    (utils.rvq_to_llm ile vocab'a eşlenir)

        -------------------------------------> LLM Encoder <---------------------------------------
                                                     |
                                                     v
                                                LLM Decoder -> Üretilen LLM Ses Belirteçleri
                                                                     |
                                                                     v (utils.llm_to_rvq)
                                                 Üretilen SpectroStream RVQ Belirteçleri (16 derinlik)
                                                                     |
                                                                     v (SpectroStream.decode())
                                                                 Yeni Ses Parçası
    ```

    1.  **Bağlam Belirteçleri (Encoder Girişi için):**
        *   `MagentaRTState.context_tokens` (son 10s, 16 RVQ derinliği) kullanılır.
        *   LLM kodlayıcısı için ilk `config.encoder_codec_rvq_depth` (4) katmanı alınır.
        *   `utils.rvq_to_llm` ile LLM kelime dağarcığına eşlenir. Eksikse MASK kullanılır.
        *   Boyut: `(context_length_frames * encoder_codec_rvq_depth)` (yaklaşık 1000 belirteç).

    2.  **Stil Belirteçleri (Encoder Girişi için):**
        *   MusicCoCa'dan gelen stil RVQ belirteçlerinin (12 derinlik) ilk `config.encoder_style_rvq_depth` (6) katmanı kullanılır.
        *   `utils.rvq_to_llm` ile LLM kelime dağarcığına eşlenir. Stil yoksa MASK kullanılır.
        *   Boyut: `(encoder_style_rvq_depth,)` (6 belirteç).

    3.  **Kodlayıcı Girişinin Birleştirilmesi ve CFG:**
        *   Bağlam ve stil belirteçleri birleştirilir.
        *   Sınıflandırıcı İçermeyen Yönlendirme (CFG) için iki kopya oluşturulur: biri stille, diğeri maskelenmiş stille.

    4.  **Kod Çözücü Girişi (Başlangıç):** Sıfırlardan oluşan bir dizi.

*   **Üretim Süreci:**
    *   Kod çözücü, otoregresif olarak `config.decoder_codec_rvq_depth` (16) derinliğinde ses belirteçleri üretir.
    *   `temperature`, `top_k`, `guidance_weight` gibi parametreler kullanılır.

*   **Çıkış İşleme:**
    *   Üretilen LLM belirteçleri `utils.llm_to_rvq` ile SpectroStream RVQ formatına dönüştürülür.

*   **Kullanılan Modeller:** "base" (`llm_base_x4286_c1860k`) veya "large" (`llm_large_x3047_c1860k`).

**4. MagentaRT Sistemi (`magenta_rt.system.MagentaRT`)**

`MagentaRT` sınıfı, tüm bileşenleri bir araya getirir.

*   **Yapılandırma (`MagentaRTConfiguration`):**
    *   `chunk_length`: 2.0s
    *   `context_length`: 10.0s
    *   `crossfade_length`: 0.04s (SpectroStream bir çerçeve uzunluğu)
    *   Diğer RVQ derinlikleri ve kelime dağarcığı tanımları.

*   **Durum Yönetimi (`MagentaRTState`):**
    *   `context_tokens`: Son `context_length` saniyelik SpectroStream RVQ belirteçlerini (`decoder_codec_rvq_depth` ile) depolar.
    *   `chunk_index`: Üretilen parça sayısını takip eder.
    *   `update()`: Yeni üretilen belirteçlerle durumu günceller (kayan bağlam penceresi).

*   **Ana İşlev (`MagentaRTT5X.generate_chunk`):**
    1.  Durumu ve stili alır.
    2.  Girişleri LLM için hazırlar (RVQ derinliklerini ayarlar, kelime dağarcığına eşler, CFG için kopyalar).
    3.  LLM ile yeni ses belirteçleri üretir.
    4.  Üretilen belirteçleri SpectroStream RVQ formatına dönüştürür.
    5.  Çapraz geçiş için önceki bağlamdan bir çerçeve ekleyerek SpectroStream ile sesi çözer.
    6.  Durumu günceller.
    7.  Ses dalga formunu ve güncellenmiş durumu döndürür.

*   **Çapraz Geçiş Uygulaması:** `generate_chunk` çapraz geçiş için materyali sağlar; asıl birleştirme `audio.concatenate` ile yapılır.

**5. Eğitim Detayları**

*   **Veri Kümesi:** ~190.000 saatlik stok müzik, çoğunlukla enstrümantal.
*   **Donanım:** TPUv6e / Trillium.
*   **Yazılım:** JAX, T5X, SeqIO.

**6. Kullanım Senaryoları ve Uygulamalar**

*   **Etkileşimli Müzik Oluşturma:** Canlı Performans, Erişilebilir Müzik Yapımı, Video Oyunları.
*   **Araştırma:** Transfer Öğrenimi.
*   **Kişiselleştirme:** İnce Ayar (yakında).
*   **Eğitim:** Müzik Kavramlarını Keşfetme.

**7. Sınırlamalar ve Riskler**

*   **Bilinen Sınırlamalar:**
    *   **Müzik Tarzı Kapsamı:** Batı enstrümantal ağırlıklı.
    *   **Vokal Üretimi:** Genellikle sözsüz.
    *   **Gecikme:** ~2 saniye (parça uzunluğu nedeniyle).
    *   **Sınırlı Bağlam:** 10 saniye, uzun vadeli yapı eksikliği.
*   **Riskler:** Ekonomik ve kültürel etki, telif hakkı ihlali, kötüye kullanım.

**8. Sonuç ve Gelecek Çalışmalar**

Magenta RT, açık kaynaklı, gerçek zamanlı ve stil koşullu müzik üretimi için önemli bir araçtır.

*   **Gelecek Çalışmalar:** Daha uzun bağlam, daha düşük gecikme, genişletilmiş stil/vokal kapsamı, ince ayar arayüzleri, gerçek zamanlı ses girişi, daha sezgisel kontroller.

**Ekler**

*   **Kullanılan Kısaltmalar:**
    *   AI: Artificial Intelligence (Yapay Zeka)
    *   API: Application Programming Interface (Uygulama Programlama Arayüzü)
    *   CFG: Classifier-Free Guidance (Sınıflandırıcı İçermeyen Yönlendirme)
    *   CNN: Convolutional Neural Network (Evrişimli Sinir Ağı)
    *   LLM: Large Language Model (Büyük Dil Modeli)
    *   MIDI: Musical Instrument Digital Interface (Müzik Enstrümanı Dijital Arayüzü)
    *   RVQ: Residual Vector Quantization (Artık Vektör Niceleme)
    *   TPU: Tensor Processing Unit (Tensör İşlem Birimi)
