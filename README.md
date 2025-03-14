# 資産運用シミュレーター

このプロジェクトは、**Julia + Pluto.jl** を使って構築した「資産運用シミュレーター」です。S&P500を想定したリターンのモンテカルロシミュレーションを行い、積立投資の結果を可視化することができます。

投資期間や毎月の積立額、初期投資額などをパラメータとして指定すると、各月の資産額の分布（四分位点や中央値）や最終資産額の分布、元本割れ確率などをインタラクティブに確認できます。

---

## 機能概要

- **モンテカルロシミュレーション**
  - EGARCHモデルによって生成した月次リターン系列を使って資産額を更新
  - 投資期間・投資上限・積立額などの条件を設定可能
- **可視化**
  - 各月の資産額の四分位数(Q1, Q3)と中央値を折れ線グラフで表示
  - 投資累計額との比較を同じグラフ上に描画
  - 最終資産額のヒストグラム・元本割れ確率の計算
- **Pluto.jlによるインタラクティブUI**
  - ブラウザ上でスライダーやテキスト入力を変えるたびに自動で再計算＆グラフ更新

---

## 使い方

1. **Juliaのインストール**

   - [Julia公式サイト](https://julialang.org/)から最新の安定版をダウンロード＆インストールしてください。

2. **このリポジトリをクローン or ダウンロード**

   - ターミナルで `git clone https://github.com/taiki-haga/portfolio-simulation.git`
   - あるいは GitHub 上の「Code」→「Download ZIP」からダウンロードします。

3. **パッケージの準備**

   - Juliaを起動してパッケージモード(`]`)に入り、必要なパッケージを追加します:

     ```julia
     ] add Pluto PlutoUI PyPlot
     
     ```

   - または本リポジトリにある `Project.toml` を使用して環境を構築しても構いません。

4. **Plutoの起動 & ノートブックを開く**

   - Julia REPL上で:

     ```julia
     using Pluto
     Pluto.run()
     
     ```

   - ブラウザが開いたら、「Open a Notebook」で `portfolio_montecarlo_simulation.jl` (Plutoノートブックファイル)を選択します。

   - ノートブックの右上にある「Run notebook code」をクリックするとコードが実行されます。

   - パラメータをインタラクティブに変更しながら、資産額のシミュレーション結果を確認できます。

---

## 注意・免責事項

- **本シミュレーターは、あくまで金融時系列の確率的生成モデルに基づくものであり、将来の実際の運用成績を保証するものではありません。**
- 現実の市場では突発的なイベントや非定常的な構造変化が起こり得るため、シミュレーション結果はあくまで参考値として扱ってください。
- 投資判断は自己責任で行い、必要に応じて専門家の助言を求めてください。
