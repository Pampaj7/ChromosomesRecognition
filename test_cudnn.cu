#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    if (cudnn) {
        std::cout << "cuDNN initialized successfully!" << std::endl;
        cudnnDestroy(cudnn);
    } else {
        std::cout << "Failed to initialize cuDNN." << std::endl;
    }
    return 0;
}
