# Brush IR

基于Brush的融合红外与 RGB 双模态图像的三维重建系统

![image-20260628132801619](./README.assets/image-20260628132801619.png)

## Features

- 支持红外+RGB双模态输入<br>![image-20260628133527278](./README.assets/image-20260628133527278.png)![image-20260628133542641](./README.assets/image-20260628133542641.png)
- 可自定义红外相机与RGB相机的偏移距离和角度<br><img src="./README.assets/image-20260628133640019.png" alt="image-20260628133640019" style="zoom:13%;" />
- RGB训练完成后，使用红外数据进行位置强化<br><img src="./README.assets/image-20260628133736457.png" alt="image-20260628133736457" style="zoom: 13%;" />

## Usage

- 将NIR数据集放在RGB图片的子目录下<br><img src="./README.assets/image-20260628134016719.png" alt="image-20260628134016719" style="zoom:13%;" />
- 选择Directory，加载一个包含colmap、RGB图集、IR图集的文件夹<br><img src="./README.assets/image-20260628133854502.png" alt="image-20260628133854502" style="zoom: 15%;" /><br><img src="./README.assets/image-20260628134104814.png" alt="image-20260628134104814" style="zoom: 20%;" />
- 勾选Enable IR training，设置参数<br><img src="./README.assets/image-20260628134302207.png" alt="image-20260628134302207" style="zoom: 5%;" />
- Start！<br><img src="./README.assets/image-20260628134339998.png" alt="image-20260628134339998" style="zoom:15%;" />

## Build

与[brush](https://github.com/ArthurBrussee/brush)完全相同，以Windows/macOS/Linux为例：

Use `cargo run --release` from the workspace root to make an optimized build. Use `cargo run` to run a debug build.

## Acknowledgements

Based on [brush](https://github.com/ArthurBrussee/brush)

