#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

// cv::imshow("Image", img);
// cv::waitKey();
int main() {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(
                "../assets/intel-resnet18.pt"
                //"../assets/imagenet-resnet18.pt"
        );
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model: " << e.what();
        return 1;
    }

    module.eval();

    cv::Mat img = cv::imread("../assets/3.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    cv::resize(img, img, cv::Size(256, 256), cv::INTER_LINEAR);
    centerCrop(img, img, cv::Size(224, 224));

    at::Tensor input = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    input = input.permute({2, 0, 1}).to(torch::kFloat) / 255.0;

    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    input = normalize_transform(input);

    at::Tensor output = module.forward({input.unsqueeze(0)}).toTensor().squeeze_(0);

    at::Tensor probabilities = output.softmax(0);

    std::vector<std::string> categories = readClasses(
            "../assets/intel-classes.txt"
            //"../assets/imagenet-classes.txt"
    );
//    assert(categories.size() == 1000);

    auto[top5_prob, top5_id] = probabilities.topk(5);

    for (size_t i = 0; i < top5_prob.numel(); ++i)
        std::cout << categories[top5_id[static_cast<int>(i)].item<int64_t>()] << " - "
                  << top5_prob[static_cast<int>(i)].item<float>() << '\n';
}