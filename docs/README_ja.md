<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./ultrarag_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./ultrarag.svg">
    <img alt="UltraRAG" src="./ultrarag.svg" width="55%">
  </picture>
</p>

<h3 align="center">
より少ないコード、より低い障壁、より速いデプロイ
</h3>

<p align="center">
|
<a href="https://ultrarag.openbmb.cn/pages/en/getting_started/introduction"><b>ドキュメント</b></a>
|
<a href="https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark"><b>データセット</b></a>
|
<a href="https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily"><b>論文デイリー</b></a>
|
<a href="./README_zh.md"><b>简体中文</b></a>
|
<a href="../README.md"><b>English</b></a>
|
<b>日本語</b>
|
</p>

---

*最新ニュース* 🔥

- [2026.01.23] 🎉 UltraRAG 3.0 リリース：「ブラックボックス」開発にさよなら—すべての推論ロジックを明確に可視化 👉|[📖 ブログ](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/ultrarag3_0.md)|
- [2026.01.20] 🎉 AgentCPM-Report モデル公開！DeepResearchがついにローカライズ：8B オンデバイス ライティングエージェント AgentCPM-Report がオープンソース化 👉 |[🤗 モデル](https://huggingface.co/openbmb/AgentCPM-Report)|

<details>
<summary>過去のニュース</summary>

- [2025.11.11] 🎉 UltraRAG 2.1 リリース：ナレッジ取り込みとマルチモーダルサポートの強化、より完全な統一評価システム！
- [2025.09.23] 新しいRAG論文デイリーダイジェスト、毎日更新 👉 |[📖 論文](https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily)|
- [2025.09.09] 軽量DeepResearch Pipelineローカルセットアップチュートリアルをリリース 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1p8JfziEwM)|[📖 ブログ](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/01_build_light_deepresearch.md)|
- [2025.09.01] UltraRAGのインストールと完全なRAGウォークスルービデオをステップバイステップでリリース 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1B9apz4E7K/?share_source=copy_web&vd_source=7035ae721e76c8149fb74ea7a2432710)|[📖 ブログ](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/00_Installing_and_Running_RAG.md)|
- [2025.08.28] 🎉 UltraRAG 2.0 リリース！UltraRAG 2.0が完全アップグレード：わずか数十行のコードで高性能RAGを構築、研究者がアイデアとイノベーションに集中できるよう支援！UltraRAG v2のコードは[v2](https://github.com/OpenBMB/UltraRAG/tree/v2)で参照できます。
- [2025.01.23] UltraRAG リリース！大規模モデルがナレッジベースをより良く理解・活用できるように。UltraRAG 1.0のコードは[v1](https://github.com/OpenBMB/UltraRAG/tree/v1)で引き続き利用可能です。

</details>

---

## UltraRAGについて

<a href="https://trendshift.io/repositories/18747" target="_blank"><img src="https://trendshift.io/api/badge/repositories/18747" alt="OpenBMB%2FUltraRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

UltraRAGは、[Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro)アーキテクチャ設計に基づく初の軽量RAG開発フレームワークで、清華大学の[THUNLP](https://nlp.csai.tsinghua.edu.cn/)、東北大学の[NEUIR](https://neuir.github.io)、[OpenBMB](https://www.openbmb.cn/home)、[AI9stars](https://github.com/AI9Stars)が共同で開発しました。

研究探索と産業プロトタイピング向けに設計されたUltraRAGは、コアRAGコンポーネント（Retriever、Generationなど）を独立した**MCPサーバー**として標準化し、**MCPクライアント**の強力なワークフローオーケストレーション機能と組み合わせています。開発者はYAML設定だけで、条件分岐やループなどの複雑な制御構造を正確にオーケストレーションできます。

<p align="center">
  <picture>
    <img alt="UltraRAG" src="./architecture.png" width=90%>
  </picture>
</p>

### UltraRAG UI

UltraRAG UIは従来のチャットインターフェースの境界を超え、オーケストレーション、デバッグ、デモンストレーションを組み合わせた視覚的RAG統合開発環境（IDE）へと進化しました。

システムには強力な組み込みPipeline Builderが搭載されており、「キャンバス構築」と「コード編集」間の双方向リアルタイム同期をサポートし、パイプラインパラメータとプロンプトのきめ細かいオンライン調整が可能です。さらに、パイプライン構造設計からパラメータチューニング、プロンプト生成まで、開発ライフサイクル全体を支援するインテリジェントAIアシスタントを導入しています。構築が完了すると、ロジックフローはワンクリックでインタラクティブな対話システムに変換できます。システムはナレッジベース管理コンポーネントをシームレスに統合し、ドキュメントQ&A用のカスタムナレッジベースを構築できます。これにより、基盤となるロジック構築からデータガバナンス、最終的なアプリケーションデプロイまでのワンストップクローズドループを実現しています。

<!-- <p align="center">
  <picture>
    <img alt="UltraRAG_UI" src="./chat_menu.png" width=80%>
  </picture>
</p> -->


https://github.com/user-attachments/assets/fcf437b7-8b79-42f2-bf4e-e3b7c2a896b9


### 主な特長

- 🚀 **複雑なワークフローのローコードオーケストレーション**
  - **推論オーケストレーション**：順次実行、ループ、条件分岐などの制御構造をネイティブサポート。開発者はYAML設定ファイルを書くだけで、数十行のコードで複雑な反復RAGロジックを実装できます。

- ⚡ **モジュラー拡張と再現**
	- **アトミックサーバー**：MCPアーキテクチャに基づき、機能を独立したサーバーに分離。新機能は関数レベルのツールとして登録するだけでワークフローにシームレスに統合でき、極めて高い再利用性を実現します。

- 📊 **統一評価とベンチマーク比較**
  - **研究効率**：標準化された評価ワークフローを内蔵し、すぐに使える主流の研究ベンチマーク。統一されたメトリクス管理とベースライン統合により、実験の再現性と比較効率を大幅に向上させます。

- ✨ **高速インタラクティブプロトタイプ生成**
  - **ワンクリックデリバリー**：面倒なUI開発にさよなら。たった1つのコマンドで、Pipelineロジックを即座にインタラクティブな対話型Web UIに変換し、アルゴリズムからデモンストレーションまでの距離を短縮します。


## インストール

2つのインストール方法を提供しています：ローカルソースコードインストール（パッケージ管理には`uv`の使用を推奨）とDockerコンテナデプロイ

### 方法1：ソースコードインストール

Python環境と依存関係の管理には[uv](https://github.com/astral-sh/uv)の使用を強く推奨します。インストール速度を大幅に向上させることができます。

**環境の準備**

uvをまだインストールしていない場合は、以下を実行してください：

```shell
## 直接インストール
pip install uv
## ダウンロード
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**ソースコードのダウンロード**

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

**依存関係のインストール**

使用目的に応じて、以下のモードのいずれかを選択して依存関係をインストールしてください：

**A: 新しい環境を作成** `uv sync`を使用して仮想環境を自動的に作成し、依存関係を同期します：

- コア依存関係：UltraRAG UIのみを使用するなど、基本的なコア機能のみを実行する場合：
  ```shell
  uv sync
  ```

- フルインストール：UltraRAGの検索、生成、コーパス処理、評価機能を完全に体験したい場合：
  ```shell
  uv sync --all-extras
  ```
- オンデマンドインストール：特定のモジュールのみを実行する場合は、必要に応じて対応する`--extra`を指定：

  ```shell
  uv sync --extra retriever   # 検索モジュールのみ
  uv sync --extra generation  # 生成モジュールのみ
  ```

インストール後、仮想環境を有効化します：

```shell
# Windows CMD
.venv\Scripts\activate.bat

# Windows Powershell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**B: 既存の環境にインストール** 現在アクティブなPython環境にUltraRAGをインストールするには、`uv pip`を使用：

```shell
# コア依存関係
uv pip install -e .

# フルインストール
uv pip install -e ".[all]"

# オンデマンドインストール
uv pip install -e ".[retriever]"
```

### 方法2：Dockerコンテナデプロイ

ローカルPython環境の設定を避けたい場合は、Dockerを使用してデプロイできます。

**コードとイメージの取得**

```shell
# 1. リポジトリをクローン
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG

# 2. イメージを準備（いずれかを選択）
# オプションA：Docker Hubからプル
docker pull hdxin2002/ultrarag:v0.3.0-base-cpu # ベースバージョン（CPU）
docker pull hdxin2002/ultrarag:v0.3.0-base-gpu # ベースバージョン（GPU）
docker pull hdxin2002/ultrarag:v0.3.0          # フルバージョン（GPU）

# オプションB：ローカルでビルド
docker build -t ultrarag:v0.3.0 .

# 3. コンテナを起動（ポート5050は自動的にマッピング）
docker run -it --gpus all -p 5050:5050 <docker_image_name>
```

**コンテナの起動**

```shell
# コンテナを起動（デフォルトでポート5050がマッピング）
docker run -it --gpus all -p 5050:5050 <docker_image_name>
```

注意：コンテナ起動後、UltraRAG UIは自動的に実行されます。ブラウザで`http://localhost:5050`に直接アクセスして使用できます。

### インストールの確認

インストール後、以下のサンプルコマンドを実行して環境が正常かどうかを確認してください：

```shell
ultrarag run examples/sayhello.yaml
```

以下の出力が表示されれば、インストールは成功です：

```
Hello, UltraRAG v3!
```


## クイックスタート

初心者から上級者まで、完全なチュートリアル例を提供しています。学術研究でも産業アプリケーション構築でも、ここでガイダンスを見つけることができます。詳細は[ドキュメント](https://ultrarag.openbmb.cn/pages/en/getting_started/introduction)をご覧ください。

### 研究実験
研究者向けに設計されており、データ、実験ワークフロー、可視化分析ツールを提供しています。
- [入門](https://ultrarag.openbmb.cn/pages/en/getting_started/quick_start)：UltraRAGに基づいて標準RAG実験ワークフローを素早く実行する方法を学びます。
- [評価データ](https://ultrarag.openbmb.cn/pages/en/develop_guide/dataset)：RAG分野で最も一般的に使用される公開評価データセットと大規模検索コーパスをダウンロードし、研究ベンチマークテストに直接使用できます。
- [ケース分析](https://ultrarag.openbmb.cn/pages/en/develop_guide/case_study)：ワークフローの各中間出力を深く追跡する視覚的ケーススタディインターフェースを提供し、分析とエラー帰属を支援します。
- [コード統合](https://ultrarag.openbmb.cn/pages/en/develop_guide/code_integration)：PythonコードでUltraRAGコンポーネントを直接呼び出し、より柔軟なカスタマイズ開発を実現する方法を学びます。

### デモシステム
開発者とエンドユーザー向けに設計されており、完全なUIインタラクションと複雑なアプリケーションケースを提供しています。
- [クイックスタート](https://ultrarag.openbmb.cn/pages/en/ui/start)：UltraRAG UIを起動し、管理者モードでの様々な高度な設定に慣れる方法を学びます。
- [デプロイガイド](https://ultrarag.openbmb.cn/pages/en/ui/prepare)：Retriever、生成モデル（LLM）、Milvusベクトルデータベースのセットアップを含む、詳細な本番環境デプロイチュートリアル。
- [Deep Research](https://ultrarag.openbmb.cn/pages/en/demo/deepresearch)：フラッグシップケース、Deep Research Pipelineをデプロイ。AgentCPM-Reportモデルと組み合わせることで、自動的に複数ステップの検索と統合を行い、数万語の調査レポートを生成できます。

## コントリビューション

コードの提出とテストを行ってくださった以下の貢献者に感謝します。また、包括的なRAGエコシステムを共同で構築する新しいメンバーを歓迎します！

標準プロセスに従って貢献できます：**このリポジトリをフォーク → イシューを提出 → プルリクエスト（PR）を作成**。

<a href="https://github.com/OpenBMB/UltraRAG/contributors">
  <img src="https://contrib.rocks/image?repo=OpenBMB/UltraRAG&nocache=true" />
</a>

## サポートのお願い

このリポジトリがあなたの研究に役立つと思われましたら、サポートを示すために⭐をつけることを検討してください。

<a href="https://star-history.com/#OpenBMB/UltraRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
 </picture>
</a>

## お問い合わせ

- 技術的な問題や機能リクエストについては、[GitHub Issues](https://github.com/OpenBMB/UltraRAG/issues)をご利用ください。
- 使用方法についての質問、フィードバック、RAG技術に関するディスカッションについては、[WeChatグループ](https://github.com/OpenBMB/UltraRAG/blob/main/docs/wechat_qr.png)、[Feishuグループ](https://github.com/OpenBMB/UltraRAG/blob/main/docs/feishu_qr.png)、[Discord](https://discord.gg/yRFFjjJnnS)に参加して、私たちとアイデアを交換してください。
- ご質問、フィードバック、またはご連絡をご希望の場合は、メールでお気軽にお問い合わせください：yanyk.thu@gmail.com

<table>
  <tr>
    <td align="center">
      <img src="./wechat_qr.png" alt="WeChat Group QR Code" width="220"/><br/>
      <b>WeChatグループ</b>
    </td>
    <td align="center">
      <img src="./feishu_qr.png" alt="Feishu Group QR Code" width="220"/><br/>
      <b>Feishuグループ</b>
    </td>
    <td align="center">
      <a href="https://discord.gg/yRFFjjJnnS">
        <img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Join Discord"/>
      </a><br/>
      <b>Discord</b>
  </td>
  </tr>
</table>
