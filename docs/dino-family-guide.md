# DINO ファミリー まとめ

> Self-Supervised Vision Transformers の進化と Open-Vocabulary Segmentation への応用

---

## 概要

DINO（**DI**stillation with **NO** labels）は Meta AI が開発した自己教師あり学習（SSL）による Vision Transformer（ViT）のシリーズ。ラベルなし画像のみから学習し、ピクセルレベルの密な空間理解を獲得する点が最大の特徴。CLIP と異なり、**画像テキストペアを必要としない**ため、任意の画像ドメインに適用可能。

|                      | DINO (ICCV 2021)      | DINOv2 (TMLR 2023) | DINOv3 (2025)                |
| -------------------- | --------------------- | ------------------ | ---------------------------- |
| 発表時期             | 2021年4月             | 2023年4月          | 2025年8月                    |
| 学習データ           | ImageNet-1k (120万枚) | LVD-142M (1.4億枚) | LVD-1689M (17億枚)           |
| 最大モデル           | ViT-B/16 (86M)        | ViT-g/14 (1.1B)    | ViT-7B/16 (67億)             |
| 損失関数             | DINO (自己蒸留)       | DINO + iBOT (MIM)  | DINO + iBOT + Gram Anchoring |
| 密な特徴品質         | 良好（創発的）        | 非常に良好         | 最高水準                     |
| ADE20K 線形評価      | 20.2 mIoU             | 35.5 mIoU          | 47.6 mIoU                    |
| ADE20K M2F評価       | —                     | 49.0 mIoU          | 63.0 mIoU                    |
| テキストアライメント | なし                  | なし               | dino.txt (後付け)            |

![DINO Paper](images/dino-paper.png)

---

## 1. DINO (2021)

**Paper**: ["Emerging Properties in Self-Supervised Vision Transformers"](https://arxiv.org/abs/2104.14294)  
**Authors**: Mathilde Caron, Hugo Touvron, et al. (Inria / Meta AI)  
**Venue**: ICCV 2021

### 核心アイデア: ラベルなし自己蒸留（Self-Distillation with No Labels）

DINO は Student-Teacher フレームワークを用いる:

1. **Teacher**: Student の指数移動平均（EMA）。勾配更新なしでパラメータの安定した平均を保持
2. **Student**: 通常の勾配降下で学習
3. 同一画像の異なる augmentation（グローバル crop + ローカル crop）をそれぞれに入力
4. Student の出力分布が Teacher の出力分布に一致するよう Cross-Entropy 損失で学習

```
Teacher = EMA(Student)  ← 勾配なしで徐々に更新
Loss = - Teacher_output * log(Student_output)  ← ソフトラベルによる蒸留
```

- Teacher が全ての画像で同じ出力（一様分布）に崩壊しないよう、**Sinkhorn-Knopp centering** を導入
- 複数解像度の crop（global 224px + local 96px）を用いる「multi-crop training」が空間認識の鍵

### 創発的特性（Emerging Properties）

DINO 最大の発見は、**明示的なピクセル教師なしで、自己注意マップに意味的セグメンテーションが自然に現れる** こと:

1. **セグメンテーションマップの創発**: 最終層の patch token を PCA で可視化すると、前景物体が自然に分離される
2. **優れた k-NN 分類器**: ViT-S/16 で ImageNet top-1 78.3%（教師なし特徴のまま）
3. **線形評価**: ViT-Base で ImageNet top-1 80.1%

この発見が、DINO 特徴を密な予測タスク（セグメンテーション、深度推定、対応点マッチングなど）に使う潮流の原点となった。

### なぜ空間認識が生まれるのか？

Multi-crop training により、ネットワークは「同じ物体が異なる位置・スケールで写っている」複数の view 間で表現を一致させる必要がある。この制約が暗黙的に空間的対応関係の学習を促す。

---

## 2. DINOv2 (2023)

**Paper**: ["DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193)  
**Authors**: Maxime Oquab, Timothée Darcet, et al. (25名, Meta AI)  
**Venue**: TMLR 2023

![DINOv2 Paper](images/dinov2-paper.png)

### DINO からの主な改良点

#### a) データ規模の大幅拡大: LVD-142M

- 非キュレーションの Web 画像約 12 億枚から、自己教師あり類似度に基づき **1.42 億枚** を自動選別
- 人手ラベルやメタデータを一切使わない完全自動パイプライン
- ImageNet-22k などの既存データセットと「近い」画像を検索・取得

#### b) DINO + iBOT の統合損失

- **DINO loss**: 画像レベルの自己蒸留（class token の分布一致）
- **iBOT loss**: パッチレベルの Masked Image Modeling（ランダムにマスクしたパッチを予測）
- iBOT の追加でセグメンテーション性能が約 +3 mIoU 向上

#### c) 大規模化と蒸留

- **ViT-g/14** (11 億パラメータ) を Teacher として学習
- 学習済み Teacher から Student（ViT-S/B/L）に蒸留
- 蒸留された Student も OpenCLIP を上回る性能

#### d) 技術的改良

- **KoLeo regularizer**: 特徴が特徴空間内で均一に分散するよう制約 → 検索性能が +8% 向上
- **FlashAttention, FSDP, mixed precision** → 学習 2 倍高速化、メモリ 1/3 削減
- **高解像度適応フェーズ**: 224px で事前学習後、518px で追加学習

### 主な能力

- 凍結特徴のまま: 分類、セグメンテーション、深度推定、インスタンス検索で SOTA
- 分布シフトに頑健（ImageNet-A, ObjectNet 等）
- PCA でセグメンテーションマップが自然に現れる（DINO の特性を継承）
- **ADE20K 線形評価: 35.5 mIoU**（DINO: 20.2）

---

## 3. DINOv3 (2025)

**Paper**: ["DINOv3"](https://arxiv.org/abs/2508.10104)  
**Authors**: Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, et al. (25名, Meta FAIR)  
**GitHub**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (10.2k stars)

![DINOv3 GitHub](images/dinov3-github.png)
![DINOv3 Paper](images/dinov3-paper.png)

### 3 つの核心的貢献

#### a) 7B パラメータへのスケール + LVD-1689M

| 項目               | DINOv2            | DINOv3                                        |
| ------------------ | ----------------- | --------------------------------------------- |
| 最大モデル         | ViT-g (1.1B)      | **ViT-7B (6,716M)**                           |
| データセット       | LVD-142M (1.42億) | **LVD-1689M (16.89億)**                       |
| アーキテクチャ改良 | —                 | Axial RoPE, register tokens, SwiGLU FFN       |
| 学習スケジュール   | Cosine            | **Constant schedule**（理論上無限に学習可能） |

- 学習は H100 GPU クラスタで FSDP sharding + mixed precision
- 新たに **ConvNeXt バックボーン**（Tiny/Small/Base/Large）も蒸留

#### b) Gram Anchoring — スケールを可能にした中核技術

DINOv3 の最大の技術的ブレークスルー:

> **問題**: モデルを大きくし長時間学習すると、dense patch feature の品質が劣化する（ノイズだらけの類似度マップ、「チェッカーボードアーティファクト」）

> **解決: Gram Anchoring** — 学習中に patch feature の Gram 行列を、より良い品質を持つ過去のチェックポイント（anchor）の Gram 行列に近づけるよう正則化

これにより、画像レベルの特徴を改善し続けながら、密な特徴の品質を維持できる。この発明なしには 7B へのスケールは不可能だった。

#### c) マルチアーキテクチャ蒸留 + 特殊ドメイン対応

- 1 つの Teacher から複数 Student への同時蒸留
- **衛星画像特化バージョン**: SAT-493M データセットで学習した ViT-L + ViT-7B
- 衛星版で樹冠高推定（CHMv2）: **R² = 0.88**

### 凍結特徴での性能（fine-tuning なし）

| タスク                                  | DINOv3 性能                       |
| --------------------------------------- | --------------------------------- |
| ADE20K セマンティックセグメンテーション | **63.0 mIoU** (ViT-7B + M2F head) |
| COCO 物体検出                           | **66.1 mAP** (ViT-7B)             |
| 単眼深度推定 (NYUv2)                    | SOTA                              |
| 3D 対応点マッチング (NAVI)              | 全 SSL/WSL 手法を大幅に凌駕       |
| 教師なし物体発見                        | DINOv2 から大幅改善               |
| 動画セグメンテーション追跡              | ノンパラメトリックで最高性能      |
| ImageNet 線形評価                       | CLIP 派生モデルと同等             |

### dino.txt — テキストアライメントの後付け

DINOv3 の重要な拡張として、**学習済み特徴に後からテキストアライメントを追加**する手法:

- CLIP と異なり、事前学習時に画像テキストペアを必要としない
- 事後的に軽量なテキストアライメント層を学習
- ゼロショット open-vocabulary セグメンテーションが可能に
- DINOv3 の密な特徴品質 × CLIP 的なテキスト対応の両取り

---

## 4. OVS 研究における DINO の位置づけ

### CLIP vs DINO: 本質的違い

| 特性                       | CLIP                                    | DINO (v2/v3)                                   |
| -------------------------- | --------------------------------------- | ---------------------------------------------- |
| 学習パラダイム             | 弱教師あり (画像+テキスト)              | 自己教師あり (画像のみ)                        |
| 学習目的                   | 画像全体 ⇔ テキストの大域的位置合わせ   | 複数 view 間の自己蒸留 + パッチ予測            |
| 空間情報                   | ほぼ保持されない (class token のみ有効) | patch token に自然に保持される                 |
| 密な特徴品質               | 低い (セグメンテーションに不向き)       | 高い (PCA で意味マップが現れる)                |
| テキスト対応               | ネイティブ                              | 後付け (dino.txt)                              |
| ドメイン柔軟性             | テキストが存在するドメインのみ          | **任意の画像ドメイン** (衛星、医療、顕微鏡...) |
| セグメンテーション線形評価 | ~15-20 mIoU                             | **47.6 mIoU** (DINOv3)                         |
| 対応点マッチング           | 弱い                                    | 非常に強い                                     |
| 深度推定                   | 弱い                                    | 強い (幾何情報を特徴に内包)                    |

### OVS 論文での使われ方

2025〜2026 年の OVS 論文では、DINO 特徴を CLIP の「空間認識の弱さ」を補完するために使うパターンが主流:

1. **OVS-DINO** (Apr 2026): DINO の境界認識能力を SAM で再活性化 → Cityscapes +6.3%
2. **DR-Seg** (Apr 2026): CLIP 特徴を semantics/substructure に分解し、DINO で構造補強
3. **SPAR** (CVPR 2026): DINOv3 の特徴を sliding-window teacher に使用
4. **GeoGuide** (CVPR 2026): 3D セグメンテーションで DINO の幾何特徴を活用

### なぜ DINO が空間認識に優れるのか

```
CLIP: 「この画像には犬が写っている」← 全体の意味だけ
DINO: 「画像のこの位置に犬がいる」    ← 位置も含めた理解
```

CLIP の contrastive loss は画像を 1 つの意味単位として扱うため、空間情報は捨てられる。DINO の multi-crop training は異なる view 間での表現一致を強制するため、ネットワークは「どこに何があるか」を暗黙的に学習する。DINOv2/v3 の iBOT loss（マスクされたパッチの予測）がこれをさらに強化する。

---

## 5. モデルバリエーション一覧

### DINOv3 (最新)

| Model     | Params | Embedding Dim | Heads |
| --------- | ------ | ------------- | ----- |
| ViT-S/16  | 21M    | 384           | 6     |
| ViT-S+/16 | 29M    | 448           | 7     |
| ViT-B/16  | 86M    | 768           | 12    |
| ViT-L/16  | 300M   | 1024          | 16    |
| ViT-H+/16 | 840M   | 1408          | 22    |
| ViT-7B/16 | 6,716M | 3072          | 48    |

| ConvNeXt Model | Params |
| -------------- | ------ |
| ConvNeXt-Tiny  | 29M    |
| ConvNeXt-Small | 50M    |
| ConvNeXt-Base  | 89M    |
| ConvNeXt-Large | 198M   |

### DINOv2

| Model    | Params | 特徴             |
| -------- | ------ | ---------------- |
| ViT-S/14 | 21M    | 蒸留版、軽量     |
| ViT-B/14 | 86M    | 蒸留版、バランス |
| ViT-L/14 | 300M   | 蒸留版           |
| ViT-g/14 | 1,100M | Teacher、最大    |

---

## 6. 研究上の示唆

### OVS 研究に DINO を使う意義

1. **CLIP が苦手な空間精度を補完** → 境界精度、小物体、混雑シーン
2. **任意ドメインに適用可能** → 医用、衛星、工業など、テキストが存在しないドメインでの OVS
3. **dino.txt によりテキスト対応も可能** → 密な特徴品質 + ゼロショット認識の両立
4. **凍結特徴で十分な性能** → fine-tuning 不要で効率的な研究開発

### 未解決の課題

- DINOv3 でも **細粒度カテゴリの識別** は依然困難（CLIP と同様）
- **階層的部分理解**（物体のパーツ認識）は未対応
- dino.txt のテキストアライメントは CLIP のネイティブ対応に及ばない面がある
- Gram Anchoring でスケール可能になったが、**どこまでスケールし続けられるか**の理論的限界は不明

### 参考リンク

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [DINO Paper (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294)
- [DINOv2 Paper (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193)
- [DINOv3 Paper (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104)
- [dino.txt: DINOv2 Meets Text](https://arxiv.org/abs/2412.16334)
