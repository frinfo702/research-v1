#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: none,
  footer: [GLA-CLIP presented by Kenichiro Goto],
)

#set text(font: "Hiragino Kaku Gothic ProN", size: 19pt, lang: "en")
#set par(leading: 0.78em)
#set list(indent: 1.1em, spacing: 0.34em)
#show raw: set text(font: "Geist Mono", size: 13pt)
#show heading.where(level: 1): set text(size: 27pt, weight: "bold")

#let ink = rgb("#111111")
#let accent = rgb("#1d4ed8")
#let muted = rgb("#5f6368")

#let slide-head(title) = [
  #text(size: 22pt, weight: "bold", fill: ink)[#title]
  #v(0.35em)
]

#let claim(body) = block(
  inset: 0pt,
  below: 0.55em,
)[
  #text(size: 24pt, weight: "bold", fill: accent)[#body]
]

#let sub(body) = text(size: 15pt, fill: muted)[#body]


#title-slide[
  = Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation

  ByeongCheol Lee, Hyun Seok Seong, Sangeek Hyun, Gilhan Park, WonJun Moon, Jae-Pil Heo
]

#slide[
  #slide-head([できた: 課題を言い切った (ref: 3.1.2)])

  #grid(
    columns: (0.9fr, 1.1fr),
    gutter: 1.1em,
    [
      #claim[原因は学習不足ではなく、window ごとの文脈分断]

      高解像度 OVSS では sliding-window 推論が必要だが、window を独立に処理すると同じ物体でも予測がずれる。

      - 境界で grid artifact が出る
      - large object と stuff category で特に悪化
      - BER でその不整合を定量化

      #v(0.4em)
      #sub[重なり領域を平均しても、意味のズレ自体は直らない。]
    ],
    [
      #figure(
        image("images/comparison-of-segmentation-consistency-near-widnow-boundaries.png", width: 100%),
        caption: [Window 境界の不整合が主要な失敗要因。],
      )
    ],
  )
]

#slide[
  #slide-head([できた: 推論時だけで補正した (ref: 3)])

  #grid(
    columns: (1.05fr, 0.95fr),
    gutter: 1.1em,
    [
      #figure(
        image("images/overview-of-gla-clip-framework.png", width: 100%),
        caption: [GLA-CLIP の全体像。],
      )
    ],
    [
      #claim[再学習なしで、window をまたぐ受容野を回復]

      - *Key-Value Extension*:
        参照先だけ全 window に広げる
      - *Proxy Anchor*:
        local query の window bias を弱める
      - *Dynamic Normalization*:
        object scale ごとに global attention を調整

      #v(0.4em)
      #sub[単に global context を足すだけでなく、bias と scale のずれも同時に補正する。]
    ],
  )
]

#slide[
  #slide-head([できた: 結果が出た (ref: 4.2)])

  #grid(
    columns: (0.95fr, 1.05fr),
    gutter: 1.1em,
    [
      #claim[既存法に後付けで効き、window artifact も減る]

      - 8 benchmark で一貫して改善
      - CLIP-DINOiser / SCLIP では平均 *+1.6 mIoU*
      - tuning なしで *44.0 mIoU*

      #v(0.5em)
      #sub[性能改善が「表の数値」だけでなく「見た目の整合性」にも出ている点が強い。]
    ],
    [
      #figure(
        image("images/open-vocabulary-semantic-segmentation-results.png", width: 100%),
        caption: [複数手法への追加で安定した利得。],
      )
    ],
  )
]

#slide[
  #slide-head([できた: 見た目でも一貫した (ref: 4.2)])

  #grid(
    columns: (1.08fr, 0.92fr),
    gutter: 1.1em,
    [
      #figure(
        image("images/qualitative-results-among-proxyclip.png", width: 100%),
        caption: [Window をまたぐ予測の段差が小さい。],
      )
    ],
    [
      #claim[同じ領域に、同じ意味を返しやすくなった]

      - Pascal VOC / COCO-Stuff / Cityscapes で改善
      - 背景や大きい物体のラベルが安定
      - BER 改善が視覚的一貫性に対応

      #v(0.4em)
      #sub[この論文の説得力は、artifact を定量・定性の両方で押さえている点にある。]
    ],
  )
]

#slide[
  #slide-head([できない (ref: 6)])

  #claim[全 window の token を見るので、推論は重い]

  - 計算量とメモリコストが増える
  - 高品質 token を選別する仕組みまでは入っていない
  - 学習なしで効く代わりに、効率面はまだ未解決
]

#slide[
  #slide-head([今後 (ref: 6)])

  #claim[次は「全部見る」から「必要な token だけ見る」へ]

  - 疎な global retrieval で計算量を落とす
  - object scale に応じた token selection を明示化する
  - training-free のまま、より大きい画像と複雑な scene に広げる
]
