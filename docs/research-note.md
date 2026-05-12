# Cycle 33

とにかく全コードと論文を理解し、1対1で対応させる。特に図表やコード、数式。これが追試

追試が終わったら自分のデータでやってみる。失敗するパターンを複数集めて分析

各モジュール（prompt-mask, CLIP encoder, mask decoder）についてこれがなくなったら何が起きるか、などモジュールの役割を仮説を立てて検証し、理解する

十分と言えるぐらいコードベースを理解したら次にいく

---

# OV-Seg Paper

## Masked Image

オブジェクトだけが切り出された画像。背景は全て単一の「背景」として0で埋められる

### (a) CLIP の学習

CLIPはそもそも自然画像とそのcaptionのpairで学習される

### (b) Mask Proposal Generator

mask proposal generatorはその領域がどの物体かは把握せず、ただ一つの物体と認識されるものを切り出し、画像サイズを調整する。こうして切り出されたmasked imagesと、推論時に与えられるテキストから抽出された名詞とのマッチングが行われ、最終的に各画像の非ゼロトークンの領域に対してセグメンテーションが行われる。（そして、最後にこれらが再度統合される？）

### (c) CLIP の性能低下

bで作成された一物体のみが映る画像に対してのCLIPの性能の低さを示している。CLIPはaのような自然画像で学習されているので自然でないmasked imageに対してはそのdomain gapから性能が大幅に下がる。classとなっているほうが分類自体が最高にうまくいった場合

## Training Data: COCO-Stuff vs COCO Captions

> Such manually annotated masks are accurate but classes are limited to a closed set (e.g., 171 classes for COCO-stuff). We hypothesize that the lack of text diversity causes the finetuned CLIP to lose the generalization ability to open vocabulary concepts. Instead, we collect training data by mining an existing image-caption dataset (e.g., COCO Captions).

擬似データ作成でCOCO-stuffだと性能劣化したのにCOCO Captionsだと性能が上がった理由は？

## Masked Image の問題点

- 背景に役立つ情報が含まれないので学べることがない
- 自然画像じゃないのでCLIPをfinetuningする際にdomain shiftが起こり得る。自然画像に対して性能が悪化する方向にシフトする

## Mask Prompt Tuning

> When tokenizing a masked image, we replace the "zero tokens" with learnable prompt tokens. During finetuning, we either train prompts only and freeze CLIP's weights, or train both of them.

4/20の発表でmask-prompt tuningの時はCLIPの重みを固定すると話したが、実際、学習するのは

1. **mask promptだけ（CLIPは重み固定）**
2. **mask-prompt & CLIP**（両方）

がある。そして後者の方が改善は大きくなるという独自の改善（他の同様手法ではみられない）

前者は、そのCLIPが他の場所でも使われるようなタスクで他のモジュールに影響を与えないという特徴がある

---

# Method of OV-Seg

## Loss

ここに載っているLossはどのモデルのloss？これは何を学習しているのか？classificationはどこ？

**予想：**

- **Loss_mask**: mask formerのloss
- **Loss_cls**: CLIP？

でこれらは同時には行われないで左から段階を踏む感じ？

## Embeddings

- **N proposal embeddings**の **C** はCLIPの埋め込み次元 (CLIPでViT-B/16ならC=512)
- **N mask proposals**では H\*W のマスクがN個生成される。そしてそのN個のマスクそれぞれに予想したクラス情報が付与されている。予測はC次元の分布で、このCはtraining setに含まれるクラスの数（つまりN proposal embeddingsの次元数より遥かに少ない）。図中ではKになっている
- **v**: オブジェクトの数Nに対しそれぞれC=512の埋め込みを与える。
- **t**: またcaptionから分類するK個のtokenに該当するものがあれば抜き出し、そのtokenをCLIP text encで同じくC=512の埋め込みベクトルに変換する

これらはそれぞれ視覚表現とテキスト表現と異なるが、CLIPは対応する視覚・言語表現が幾何的に近くになるように事前学習されているのでコサイン類似度を取ることでそのオブジェクトがある領域についてなんのオブジェクトがあるのかがわかる

**vとvハットの違いは？**

## Mask Former

初めMask FormerはCheng et al. (2021/2022) による既存モデル

## Mask Prompt Tuning（詳細）

- **T**: トークン化されたオブジェクトのあるピクセルのトークン。パッチ数xトークンの埋め込み次元
- **M_p**: {0, 1}で各パッチが完全にマスクになっているかそうでないかを示す
- **P**: prompt token。サイズはTと同じ

---

# チェックポイントファイル

| ファイル                                       | 対応モデル | mask prompt 形状 |
| ---------------------------------------------- | ---------- | ---------------- |
| `checkpoints/ovseg_R101c_vitB16_ft_mpt.pth.pt` | ViT-B/16   | (3, 196, 768)    |
| `checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth` | ViT-L/14   | (3, 256, 1024)   |

実際にViT-B/16のマスクを逆行列をかけてdecodeした。画像として表示してみると解釈性はないものだった

---

# Questions

何を目的として学習するものなの？

lossが下がるように。
CLIPが獲得している自然画像での分布と、ゼロトークンを含んだものとでは大きな差がある。このドメインギャップを埋めるために自然なシフトをさせつつ性能を復活させるのが目的？

---

# 処理パイプライン

## Modified Mask Former

画像をオブジェクトごとに区切ってmask imageを生成する。COCO-Stuffを使って学習する

## CLIP Finetuning

COCO-Captionsを使ってファインチューニングをする
