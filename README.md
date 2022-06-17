# depthai_jojo
ジョジョ立ちしたらスタンドを出すシステム。
[以前作ったものの](https://www.youtube.com/watch?v=SuFs3hz-SXA) AI カメラ使用バージョン

## Usage
```
$ git clone https://github.com/i-icc/depthai_jojo.git
$ cd depthai_jojo
$ python -m pip install -r requirements.txt
$ python main.py -cam
$ python main.py -vid ./movies/input/001.mov
```

骨格を表示しない場合は nd オプションをつける
`$ python main.py -cam -nd`