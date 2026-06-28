# Brush IR

基于Brush的融合红外与 RGB 双模态图像的三维重建系统

<img src="./README.assets/image-20260628132801619.png" alt="image-20260628132801619" width="80%" />

## Features

- 支持红外+RGB双模态输入<br><img src="./README.assets/image-20260628133527278.png" alt="image-20260628133527278" width="25%" /><img src="./README.assets/image-20260628133542641.png" alt="image-20260628133542641" width="25%" />
- 可自定义红外相机与RGB相机的偏移距离和角度<br><img src="./README.assets/image-20260628133640019.png" alt="image-20260628133640019" width="25%" />
- RGB训练完成后，使用红外数据进行位置强化<br><img src="./README.assets/image-20260628133736457.png" alt="image-20260628133736457" width="30%" />

## Usage

- 将NIR数据集放在RGB图片的子目录下<br><img src="./README.assets/image-20260628134016719.png" alt="image-20260628134016719" width="25%" />
- 选择Directory，加载一个包含colmap、RGB图集、IR图集的文件夹<br><img src="./README.assets/image-20260628133854502.png" alt="image-20260628133854502" width="25%" /><br><img src="./README.assets/image-20260628134104814.png" alt="image-20260628134104814" width="25%" />
- 勾选Enable IR training，设置参数<br><img src="./README.assets/image-20260628134302207.png" alt="image-20260628134302207" width="25%" />
- Start！<br><img src="./README.assets/image-20260628134339998.png" alt="image-20260628134339998" width="30%" />

## Build

与[brush](https://github.com/ArthurBrussee/brush)完全相同，以Windows/macOS/Linux为例：

Use `cargo run --release` from the workspace root to make an optimized build. Use `cargo run` to run a debug build.

## Acknowledgements

Based on [brush](https://github.com/ArthurBrussee/brush)