#include <iostream>
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include<ctime>
using namespace cv;
using namespace std;

//#define USE_CUDA

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#endif // 

/************** 图像输入预处理 ************/
std::vector<float> preprocess_image(Mat img_dst, std::vector<float> input_tensor_values)
{
    std::size_t counter = 0;
    for (unsigned k = 0; k < 3; k++)
    {
        for (unsigned i = 0; i < img_dst.rows; i++)
        {
            for (unsigned j = 0; j < img_dst.cols; j++)
            {
                input_tensor_values[counter++] = (static_cast<float>(img_dst.at<cv::Vec3b>(i, j)[k]) - 128.) / 256.0;
            }
        }
    }
    return input_tensor_values;
}
/************** 绘制关键点 ************/
void draw_keypoints(Mat &img_o, std::vector<int> pts)
{
    for (unsigned k = 0; k < 5; k++)
    {
        int i = k*8;
        unsigned char R = 255, G = 0, B = 0;
        
        switch(k)
        {
        case 0:R = 255; G = 0; B = 0;break;
        case 1:R = 255; G = 0; B = 255;break;
        case 2:R = 255; G = 255; B = 0;break;
        case 3:R = 0; G = 255; B = 0;break;
        case 4:R = 0; G = 0; B = 255;break;
        default:printf("error\n");
        }
        
        line(img_o, Point(pts[0], pts[1]), Point(pts[i + 2], pts[i + 3]), Scalar(B,G,R), 2, CV_AA);
        line(img_o, Point(pts[i + 2], pts[i + 3]), Point(pts[i + 4], pts[i + 5]), Scalar(B, G, R), 2, CV_AA);
        line(img_o, Point(pts[i + 4], pts[i + 5]), Point(pts[i + 6], pts[i + 7]), Scalar(B, G, R), 2, CV_AA);
        line(img_o, Point(pts[i + 6], pts[i + 7]), Point(pts[i + 8], pts[i + 9]), Scalar(B, G, R), 2, CV_AA);
    }
}

int main(int argc, char* argv[]) {

    /***************** ONNX 测试 ****************/

    const wchar_t* model_path = L"./resnet_50_size-256.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    printf(" --->>> ONNX Runtime USE_CUDA \n");
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session session(env, model_path, session_options);
    //Ort::Session session(env, model_path, session_options);
    printf(" load onnx model");
    // print model input layer (node names, types, shape etc.)

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::cout << session.GetInputName(0, allocator) << std::endl;
    std::cout << session.GetOutputName(0, allocator) << std::endl;
    //--------------------------------------------------------------
    printf("Number of inputs = %zu\n", num_input_nodes);

    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;
    /********************************* 打印模型 节点 *****************************/
    //迭代所有的输入节点
    for (int i = 0; i < num_input_nodes; i++) {
        //输出输入节点的名称
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        //batch_size=1
        input_node_dims[0] = 1;
    }

    //打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        char* output_name = session.GetOutputName(i, allocator);
        printf("Output: %d name=%s\n", i, output_name);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    // 使用样本数据对模型进行评分，并检验出入值的合法性
    size_t input_tensor_size = 3 * 256 * 256;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!
    std::vector<float> input_tensor_values(input_tensor_size);

    // 加载样本文件夹
    vector<cv::String> fn;
    cv::String pattern = "./sample/*.jpg";
    glob(pattern, fn, false);
    size_t count = fn.size(); //number of png files in images folder

    Size dsize = Size(256, 256);
    int64_t output_tensor_size = 42;// 关键点输出 （x,y）*21= 42

    for (size_t k = 0; k < 10; k++)
    {
        for (size_t i = 0; i < count; i++)
        {
            Mat img_o = imread(fn[i]);

            Mat img_dst = Mat(dsize, CV_8UC3);
            resize(img_o, img_dst, dsize);

            //printf("cols: %d ,rows: %d\n", img_dst.cols, img_dst.rows);

            input_tensor_values = preprocess_image(img_dst, input_tensor_values);

            auto start_time = clock();
            // 为输入数据创建一个Tensor对象
            try
            {
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
                assert(input_tensor.IsTensor());
                // 推理得到结果
                auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, input_node_names.size(), output_node_names.data(), output_node_names.size());
                assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

                // Get pointer to output tensor float values
                //float* output = output_tensors[0].GetTensorMutableData<float>();
                float* output = output_tensors.front().GetTensorMutableData<float>();
                //printf(" --->>> Number of outputs = %d\n", output_tensors.size());
                auto end_time = clock();
                printf("  -> ONNX Inference Cost Time: %.5f Seconds\n", static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC);
                
                // 获取关键点像素坐标
                std::vector<int>results(output_tensor_size);
                for (unsigned i = 0; i < 21; i++)
                {
                    int x = output[i * 2] * img_o.cols;
                    int y = output[i * 2 + 1] * img_o.rows;
                    results[i * 2] = x;
                    results[i * 2 + 1] = y;
                    circle(img_o, Point(x, y), 4, Scalar(255, 155, 0), 2);
                }

                // 绘制关键点
                draw_keypoints(img_o, results);

            }
            catch (Ort::Exception& e)
            {
                printf(e.what());
            }
            // 创建一个名为 "图片"窗口    
            namedWindow("keypoint", 0);
            // 在窗口中显示图片   
            imshow("keypoint", img_o);
            if (waitKey(1) == 27) break;
            img_o.release();
            img_dst.release();
        }
    }
    cout << "well done !" << endl;
    return 0;
}