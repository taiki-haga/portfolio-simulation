### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 90fd6c99-f3db-4dec-937a-2aae26b1d638
using PlutoUI, Statistics, PyPlot

# ╔═╡ 37434e69-59ae-469e-ab10-0786b6b7f2af
md"""
## インデックス投資のモンテカルロシミュレーション

みなさん、NISAで積立投資はしてますか？

これは、Julia + Pluto.jl を使って構築した「資産運用シミュレーター」です。S&P500を想定したリターンのモンテカルロシミュレーションを行い、積立投資の結果を可視化することができます。

このシミュレーターの特徴は、資産額の期待値だけでなく、**株価の確率的な変動を考慮した資産額の「ばらつき」**も予測してくれる点にあります。例えば、月3万円を10年間投資したとき、運悪く最終資産額が元本（360万円）を下回ってしまう確率を計算することもできます（この場合はおよそ7%となります）。

このシミュレーターの使い方を説明します：
1. 投資期間（カ月）を入力します（最大48カ月）。
2. 初期投資額（万円）を入力します。これはゼロでも構いません。
3. 毎月の積立額（万円）を入力します（最大10万円）。最初の月に初期投資額を投資した後、次の月から一定額を積み立てていくことになります。
4. 総投資額（万円）を入力します（最大1800万円）。累計投資額がこの総投資額を越えると、それ以降の月の積立額はゼロになります。

以上のパラメータを設定すると、各月の資産額の分布（四分位点や中央値）や最終資産額の分布、元本割れ確率などをインタラクティブに確認できます。
"""

# ╔═╡ 341e3655-0387-436f-960b-efb6a460fc2a
begin
	# 各種関数の定義
	
	#= 
    	simulate_egarch(
	        T::Int,
	        ω::Float64,
	        α::Float64,
	        β::Float64,
	        γ::Float64,
	        μ::Float64,
	        T_warmup::Int
	    ) -> r::Vector{Float64}, σ::Vector{Float64}
	
	EGARCH(1,1,1)モデルのシミュレーションを実行して、リターンとボラティリティを返す。
	- T: 出力データの長さ
	- ω, α, β, γ: モデルパラメータ
	- μ: 平均リターン
	- T_warmup: ウォームアップの長さ
	=#
	function simulate_egarch(
    	T::Int,
    	ω::Float64,
    	α::Float64,
    	β::Float64,
    	γ::Float64,
    	μ::Float64,
    	T_warmup::Int
	)
    	# ウォームアップを含むシミュレーションの長さ
    	T_total = T + T_warmup
	
    	# 各配列の初期化
    	r_total = zeros(T_total)     # ウォームアップを含むリターン
    	σ_total = zeros(T_total)     # ウォームアップを含むボラティリティ
    	ln_σ2_total = zeros(T_total) # ウォームアップを含む対数ボラティリティ
	
    	# 標準正規分布の絶対値|z_t|の平均値
    	E_abs_z = sqrt(2 / π)
	
    	# 対数ボラティリティの初期化（任意）
    	ln_σ2_total[1] = ω
	
    	# 正規乱数
    	z_total = randn(T_total)
	
    	for t in 1:T_total
	        if t > 1
	            # ln(σ_t^2)をアップデート
	            ln_σ2_total[t] = ω + β * ln_σ2_total[t-1] + γ * z_total[t-1] + α * (abs(z_total[t-1]) - E_abs_z)
 	       end
	
			# σ_tをアップデート
	        σ_total[t] = sqrt(exp(ln_σ2_total[t]))

	        # r_tをアップデート
	        r_total[t] = μ + σ_total[t] * z_total[t]
	    end

	    # ウォームアップ期間を除く
	    r = r_total[T_warmup+1:end]
	    σ = σ_total[T_warmup+1:end]

	    return r, σ
	end

	#= 
	    return_simulator(T::Int) -> Vector{Float64}

	EGARCH(1,1,1)モデルをシミュレートして、S&P500を模倣するリターンを返す。
	=#
	function return_simulator(T::Int)
	    # EGARCH(1,1,1)モデルのパラメータ
	    ω = -2.293              # Omega (ω)
	    α = 0.337               # Alpha (α) - magnitude effect
	    β = 0.638               # Beta (β) - persistence
	    γ = -0.345              # Gamma (γ) - asymmetry
	    μ = 0.00692             # Mu (μ) - average return

	    T_warmup = 100          # ウォームアップ期間

	    returns_simulated, volatility_simulated = simulate_egarch(T, ω, α, β, γ, μ, T_warmup)

	    return returns_simulated
	end

	#=
	    calc_invested_cumulative(
	        total_month::Int,
	        initial_investment::Float64,
	        monthly_investment::Float64,
	        total_investment::Float64
	    ) -> Vector{Float64}

	投資期間 total_month ヶ月について、
	- 初期投資額 initial_investment
	- 毎月の積立額 monthly_investment
	- 累計投資上限 total_investment
	
	これらの条件下で各月末までに累積投資額を1次元ベクトルとして返す。
	=#
	function calc_invested_cumulative(
	    total_month::Int,
	    initial_investment::Float64,
	    monthly_investment::Float64,
	    total_investment::Float64
	)
	    invested_cum = zeros(Float64, total_month)
	    invested_so_far = 0.0
	
	    for t in 1:total_month
	        if t == 1
	            # 初月に初期投資
	            invested_so_far += initial_investment
	        else
	            # 2ヶ月目以降は monthly_investment を追加
	            if invested_so_far < total_investment
	                remain = total_investment - invested_so_far
	                invest_now = min(monthly_investment, remain)
	                invested_so_far += invest_now
	            end
	        end
	        invested_cum[t] = invested_so_far
	    end

	    return invested_cum
	end

	#=
    	simulate_portfolio(
	        invested_cum::Vector{Float64},
	        n_sims::Int
	    ) -> asset_paths::Matrix{Float64}
	
	- invested_cum: 長さ total_month のベクトルで、
	  月 t 時点までの累計投資額を表す (事前に決定論的に計算済み)
	- n_sims: シミュレーション回数

	戻り値:
	- asset_paths: size (n_sims, total_month)
	  各シミュレーション、各月の資産額
	=#
	function simulate_portfolio(invested_cum::Vector{Float64}, n_sims::Int)
	    total_month = length(invested_cum)
	    asset_paths = zeros(n_sims, total_month)
	
	    for sim in 1:n_sims
	        # 月次リターン系列
	        r_series = return_simulator(total_month)

	        # 各月の資産額を計算
	        value = 0.0  # 前月末の資産額（最初は0とし、1ヶ月目で初期投資ぶんを加える）
	        for t in 1:total_month
	            # 追加投資額 = 今月までの累計 - 前月までの累計
	            additional = (t == 1) ? invested_cum[1] : (invested_cum[t] - invested_cum[t-1])

	            value += additional         # 今月分の投資額を加える
	            value *= exp(r_series[t])   # 今月のリターンを反映

	            asset_paths[sim, t] = value
	        end
	    end

	    return asset_paths
	end
	
end

# ╔═╡ fefd532c-8a1c-42a9-81b3-7fb3fc031a6d
md"""#### パラメータを入力："""

# ╔═╡ 3d9053f5-9e28-4d96-9bac-285816158e0a
begin
	total_month_slider = @bind total_month Slider(0:10:480, default=120)
	md"""投資期間を入力： $(total_month_slider)"""
end

# ╔═╡ 8943aee0-d93e-49bc-8d8e-21440fb3948a
md"""投資期間： $(total_month)ヶ月"""

# ╔═╡ 23191b9e-864b-4e1e-9bff-902b6c22ffa2
begin
	initial_investment_slider = @bind initial_investment Slider(0.0:10.0:500.0, default=100.0)
	md"""初期投資額を入力： $(initial_investment_slider)"""
end

# ╔═╡ f9ebba56-9282-44f5-beae-e5892797b800
md"""初期投資額： $(initial_investment)万円"""

# ╔═╡ 530a5157-5cf8-4798-aefc-ca7cc158f7d2
begin
	monthly_investment_slider = @bind monthly_investment Slider(0.0:0.1:10.0, default=3.0)
	md"""毎月の積立額を入力： $(monthly_investment_slider)"""
end

# ╔═╡ 071f424b-ebe9-4b1f-b7d3-189097f064d2
md"""毎月の積立額： $(monthly_investment)万円"""

# ╔═╡ 4f348175-b373-4e4f-8746-66782ee65802
begin
	total_investment_slider = @bind total_investment Slider(0.0:10.0:1800.0, default=500.0)
	md"""総投資額を入力： $(total_investment_slider)"""
end

# ╔═╡ ec51d1d1-dd6d-4345-890c-bc6a5d472387
md"""総投資額： $(total_investment)万円"""

# ╔═╡ 700a9fa1-659b-4e0f-8251-359bb4b81383
md"""#### 累計投資額を計算："""

# ╔═╡ 038baf26-c4cb-45b2-90a6-b6804aeef34e
# 各月末までの「累計投資額」を計算
invested_cum = calc_invested_cumulative(
    total_month, 
	initial_investment, 
	monthly_investment, 
	total_investment
)

# ╔═╡ 45962491-09c9-4ac8-aaff-44c197c714dd
md"""#### 資産額の時系列を1000個計算："""

# ╔═╡ c4d80d5b-25d2-4dbd-81df-4acc98c279ca
# リターンを1000回モンテカルロし、資産推移を得る
asset_paths = simulate_portfolio(invested_cum, 1000)

# ╔═╡ a0eea725-eb11-4e09-be0e-0acb83d60d5d
# 各月の資産額の中央値と四分位数(Q1, Q3)を計算
begin
	month_median = zeros(total_month)
	month_q1 = zeros(total_month)
	month_q3 = zeros(total_month)
	for t in 1:total_month
    	col = asset_paths[:, t]
    	month_median[t] = median(col)
    	month_q1[t] = quantile(col, 0.25)
    	month_q3[t] = quantile(col, 0.75)
	end
end

# ╔═╡ 83e0433a-5206-4a93-8f2d-5b592eb3e910
md"""#### 各月の資産額の中央値と第1四分位数・第3四分位数をプロット："""

# ╔═╡ dc937b82-d7b6-4cc1-88fb-e3c10d43ef19
begin
	figure()
	# Q1~Q3の帯を塗りつぶし
	fill_between(
    	1:total_month, 
	    month_q1, 
    	month_q3, 
    	color="lightblue", 
    	alpha=0.3, 
    	label="Q1–Q3 range"
	)

	# 中央値をプロット
	plot(1:total_month, month_median, color="black", linewidth=2, label="Median")

	# 投資累計額をプロット
	plot(1:total_month, invested_cum, color="red", linestyle="--", linewidth=2, label="Invested so far")

	xlabel("Month")
	ylabel("Asset Value (¥10⁴)")
	title("Asset Growth Simulation")
	legend(loc="upper left")
	
	gcf()
end

# ╔═╡ 967889bd-5c7f-4aac-a9ba-b0c9698686b6
md"""最終資産額の中央値： $(round(month_median[end], digits=1))万円"""

# ╔═╡ 171a81bd-6113-4d49-b9db-2fe6c72443d3
md"""最終資産額の第1四分位数（下から25%）： $(round(month_q1[end], digits=1))万円"""

# ╔═╡ 03374f35-12c5-47c2-a186-78ab90cf2917
md"""最終資産額の第3四分位数（下から75%）： $(round(month_q3[end], digits=1))万円"""

# ╔═╡ 757e17fe-140c-4041-a0d5-c2e7aefe7598
md"""#### 最終資産額："""

# ╔═╡ dd7faac2-0cc7-46ad-bc6e-274308bc4c00
# 最終資産額
final_values = asset_paths[:, end]

# ╔═╡ a6c2f05e-ad84-4d8e-bc88-ff540f86d2aa
md"""#### 最終資産額の分布："""

# ╔═╡ 04e5166a-407d-4604-a9ad-fe118e0a3bc0
begin
	figure()
	
	# ヒストグラムをプロット (bins=50 でビン数を50に指定)
	hist(final_values, bins=50, color="skyblue", edgecolor="black")

	# 最終資産額の中央値と四分位数(Q1, Q3)を計算
	med_val = median(final_values)
	q1_val  = quantile(final_values, 0.25)
	q3_val  = quantile(final_values, 0.75)
	
	# 中央値と四分位数(Q1, Q3)を縦線で描画
	axvline(med_val, color="red", linestyle="--", label="Median")
	axvline(q1_val, color="blue", linestyle="--", label="Q1")
	axvline(q3_val, color="green", linestyle="--", label="Q3")
	
	xlabel("Final Asset Value (¥10⁴)")
	ylabel("Frequency")
	title("Final Asset Value Distribution")
	legend(loc="best")
	
	gcf()
end

# ╔═╡ 6257034f-f2c1-48aa-ad82-1cccb65ca3aa
md"""##### 元本割れが起こる確率： $(round(mean(final_values .< invested_cum[end]) * 100, digits=2))%"""

# ╔═╡ d31b5fa1-cd4a-4a54-ab1c-816c27757b5a
md"""##### すべてのサンプルの中で最低の最終資産額：$(round(minimum(final_values), digits=1))万円"""

# ╔═╡ b05231b8-6d85-4138-90c4-0b69208d8bac
md"""## 付録"""

# ╔═╡ a9933642-2cef-4a81-9ca3-11f84fe94b00
md"""
#### 概要
1. 1985年~2025年のS&P500のデータ（ドルベース）を利用してモデルを構築しています。
2. 過去40年間の月次リターンが定常な確率過程に従うとみなし、この傾向が今後も続くことを仮定しています。
3. S&P500に完璧に連動するインデックス投資を想定しており、為替の影響や手数料は考慮していません。

#### モデル

``t``カ月目の株価を``P_t``として、リターンを``r_t = \ln(P_t / P_{t-1})``によって定義します。
月次リターン``r_t``をシミュレートする確率モデルとして、ここではEGARCHモデルを採用しました：

``r_t = \mu + \sigma_t z_t ``

``\mu``: 平均リターン

``\sigma_t``: 分散

``z_t``: 平均0・分散1の正規乱数

``\ln(\sigma_t^2) = \omega + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2) + \alpha (|z_{t-1}| - \sqrt{2/\pi})``

つまり、``r_t``は平均リターン``\mu``のまわりでランダムにゆらぐわけですが、その分散（ボラティリティ）は過去のゆらぎの影響を受けて時々刻々と変動するという形になっています。
一般に、何かのきっかけでリターンが大きく変動すると、しばらくの間ボラティリティが大きい期間が続く傾向があります。
最後の式は、過去のゆらぎが未来のボラティリティにどの程度影響を与えるのかをモデル化しています。

パラメータ``\mu``, ``\omega``, ``\alpha``, ``\beta``, ``\gamma``はS&P500のデータを最もよくフィットするように決定しました：

``\mu = 0.00692``, 
``\omega = -2.293``, 
``\alpha = 0.337``, 
``\beta = 0.638``, 
``\gamma = -0.345``.

EGARCHモデルをシミュレートすることによってS&P500を模倣した月次リターン``r_t``の時系列が得られたとします。
インデックスファンドがS&P500に完全に連動しているならば、``t``カ月時点の資産額``V_t``は

``V_{t} = e^{r_t} (V_{t-1} + A_{t-1})``

という漸化式に従います。
ここで``A_t``は``t``カ月目の積立額です。
初期値``V_1``は指定された初期投資額に設定されます。
また、積立投資は2カ月目から開始するとして、``A_1=0``としています。
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoUI = "~0.7.61"
PyPlot = "~2.11.5"
Statistics = "~1.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "d8e4e4aa9b1c5f6bfac98542c88c9b439c03df78"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "0371ca706e3f295481cbf94c8c36692b072285c2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.5"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═90fd6c99-f3db-4dec-937a-2aae26b1d638
# ╟─37434e69-59ae-469e-ab10-0786b6b7f2af
# ╟─341e3655-0387-436f-960b-efb6a460fc2a
# ╟─fefd532c-8a1c-42a9-81b3-7fb3fc031a6d
# ╟─3d9053f5-9e28-4d96-9bac-285816158e0a
# ╟─8943aee0-d93e-49bc-8d8e-21440fb3948a
# ╟─23191b9e-864b-4e1e-9bff-902b6c22ffa2
# ╟─f9ebba56-9282-44f5-beae-e5892797b800
# ╟─530a5157-5cf8-4798-aefc-ca7cc158f7d2
# ╟─071f424b-ebe9-4b1f-b7d3-189097f064d2
# ╟─4f348175-b373-4e4f-8746-66782ee65802
# ╟─ec51d1d1-dd6d-4345-890c-bc6a5d472387
# ╟─700a9fa1-659b-4e0f-8251-359bb4b81383
# ╟─038baf26-c4cb-45b2-90a6-b6804aeef34e
# ╟─45962491-09c9-4ac8-aaff-44c197c714dd
# ╟─c4d80d5b-25d2-4dbd-81df-4acc98c279ca
# ╟─a0eea725-eb11-4e09-be0e-0acb83d60d5d
# ╟─83e0433a-5206-4a93-8f2d-5b592eb3e910
# ╟─dc937b82-d7b6-4cc1-88fb-e3c10d43ef19
# ╟─967889bd-5c7f-4aac-a9ba-b0c9698686b6
# ╟─171a81bd-6113-4d49-b9db-2fe6c72443d3
# ╟─03374f35-12c5-47c2-a186-78ab90cf2917
# ╟─757e17fe-140c-4041-a0d5-c2e7aefe7598
# ╟─dd7faac2-0cc7-46ad-bc6e-274308bc4c00
# ╟─a6c2f05e-ad84-4d8e-bc88-ff540f86d2aa
# ╟─04e5166a-407d-4604-a9ad-fe118e0a3bc0
# ╟─6257034f-f2c1-48aa-ad82-1cccb65ca3aa
# ╟─d31b5fa1-cd4a-4a54-ab1c-816c27757b5a
# ╟─b05231b8-6d85-4138-90c4-0b69208d8bac
# ╟─a9933642-2cef-4a81-9ca3-11f84fe94b00
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
