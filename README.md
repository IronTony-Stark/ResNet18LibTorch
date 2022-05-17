# ResNet18LibTorch

## ResNet18 Inference from C++ using LibTorch

### Important Notes
1. Works with MSVC compiler. Does not work with MinGW.
2. LibTorch Debug and Release binaries are not ABI compatible. 
That means you will run into weird and incorrect behaviour while 
building in Debug and using Release binaries. In my case I was getting
the "File Not Found" error when `torch::jit::load`ing the model although 
the file definitely was there.
3. You might need to fix some paths. Unfortunately I didn't have the time 
to double-check everything, so there might be a mistake somewhere.

### Links
[LibTorch](https://pytorch.org/get-started/locally/)

[OpenCV](https://opencv.org/releases/)

LibTorch and OpenCV binaries should be located in the parent directory 
of the project (see `CMakeLists.txt`)

Trained models and example images are located in the `assets` folder.

### TODO
1. Double-check and fix paths
2. Add a Python code example to get a TorchScript model
3. Double-check [PyTorch Tutorial](https://pytorch.org/cppdocs/installing.html)
4. Remove unnecessary asset files
5. Move assets to releases