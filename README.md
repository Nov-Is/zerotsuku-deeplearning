## 環境構築
```
docker compose build
```

コンテナ内に入るとき
```
docker compose exec python-dev bash
```

## データセットの取得

MNISTデータセットは「ゼロから作るDeep Learning」の公式リポジトリから取得してください。

```
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git
cp -r deep-learning-from-scratch/dataset ./dataset
```

