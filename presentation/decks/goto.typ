#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: none,
  footer: [個別ゼミ / Open-Vocabulary Segmentation 研究構想],
)

#set text(font: "Hiragino Kaku Gothic ProN", size: 18pt, lang: "ja")
#set par(leading: 0.8em)
#set list(indent: 1.1em, spacing: 0.36em)
#show raw: set text(font: "Geist Mono", size: 13pt)
#show heading.where(level: 1): set text(size: 28pt, weight: "bold")

#let ink = rgb("#111111")
#let accent = rgb("#0f766e")
#let muted = rgb("#5f6368")

#let slide-head(title) = [
  #text(size: 22pt, weight: "bold", fill: ink)[#title]
  #v(0.35em)
]

#let tag(label, color: accent) = [
  #text(fill: color, weight: "bold")[#label]
]

#title-slide[
  = 個別ゼミ

  training-free Open-Vocabulary Semantic Segmentation を軸に、拡散モデルの導入可能性を検討中
]

#slide[
  #slide-head([現在の立ち位置])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    [
      #tag([決まったこと])
      - 扱う分野は Open-Vocabulary Segmentation
      - 当面の主タスクは training-free Open-Vocabulary Semantic Segmentation (OVSS)
      - 研究の方向として拡散モデルを取り入れたい
        - ただ、これも前セメスターを引きずっているだけで、アプローチ自体は探索を続ける
    ],
    [
      #tag([まだ決まっていないこと], color: rgb("#b45309"))

      - 具体的な問題設定
      - 中心仮説
      - 新規性の軸
      - 比較対象と評価指標
      // - 最終的な投稿先
    ],
  )
]

#slide[
  #slide-head([現時点で決めたこと])

  - *タスク*:
    training-free OVSS を第一候補
  - *データセット方針*:
    主要 4 データセットは全て扱うつもり
  - *比較の考え方*:
    自分に有利な設定だけに寄せず、強い baseline を 2〜4 本は置く
  - *研究姿勢*:
    SOTA で全て改善するよりも、公平な設定で意味のある改善を示す
  - *条件*:
    学期中に回せる計算量と実装難易度を強く意識する
]

#slide[
  #slide-head([未定な論点])

  #grid(
    columns: (1.02fr, 0.98fr),
    gutter: 1.1em,
    [

      - *問題設定*:
        何が既存法の弱点なのか
      - *仮説*:
        「既存手法はなぜ弱いのか」を一言で言えるか
      - *新規性*:
        手法だけでなく、評価・分析・設定の軸でも出せるか
      - *評価指標*:
        mIoU だけで十分か。seen / unseen や境界品質も要るか
    ],
    [
      #tag([避けようと思っている], color: rgb("#7c3aed"))

      - 面白そうだが重すぎるテーマ
      - 自分提案法だけ有利な評価設定
      - 改善しても理由を説明できないテーマ
      - 失敗時の逃げ道がない実験計画
    ],
  )
]

#slide[
  #slide-head([関連論文])

  #grid(
    columns: (0.95fr, 1.05fr),
    gutter: 1em,
    [
      #tag([CVPR 2026 / CVPR 2025 / ICCV 2025 を中心に調査])

      - GLA-CLIP
      - PEARL
      - FLOSS
      - CorrCLIP
      - Training-Free Class Purification
      - Feature Purification Matters
      - DIH-CLIP
    ],
    [
      - *Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation*
      - 拡散モデルを OVSS に使う流れは既にある
      - まずは training-free OVSS の失敗原因を整理し、その上で diffusion がどこに効果的かを切り分ける必要がある
    ],
  )
]


#slide[
  #slide-head([実験計画の基準])

  - *実験可能性*:
    3 か月で最低限の結果が出るか
  - *再現性*:
    baseline の再現が現実的か
  - *分析可能性*:
    アブレーション、可視化、失敗例分析、計算量比較ができるか
  - *拡張性*:
    失敗した場合でも別案へ切り替えられるか
  - *出口*:
    CV 系、ML 系、応用系のどの方向性か
]

#slide[
  #slide-head([今後])

  1. 評価指標を mIoU 以外も含めて整理する
  2. 最初に回すデータセットと最小実験系を決める
]
