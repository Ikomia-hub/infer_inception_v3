#include "InceptionV3.h"
#include "Graphics/CGraphicsLayer.h"

CInceptionV3::CInceptionV3(): COcvDnnProcess(), CClassificationTask()
{
    m_pParam = std::make_shared<CInceptionV3Param>();
}

CInceptionV3::CInceptionV3(const std::string &name, const std::shared_ptr<CInceptionV3Param> &pParam):
    COcvDnnProcess(), CClassificationTask(name)
{
    m_pParam = std::make_shared<CInceptionV3Param>(*pParam);
}

size_t CInceptionV3::getProgressSteps()
{
    return 3;
}

int CInceptionV3::getNetworkInputSize() const
{
    int size = 224;

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    auto pParam = std::dynamic_pointer_cast<CInceptionV3Param>(m_pParam);
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
        size = size + (m_sign * 32);

    return size;
}

std::vector<std::string> CInceptionV3::getOutputsNames() const
{
    // Return empty list as we only want the last layer
    return std::vector<std::string>();
}

void CInceptionV3::globalInputChanged(bool bNewSequence)
{
    setNewInputState(bNewSequence);
}

void CInceptionV3::run()
{
    beginTaskRun();

    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    if (pInput == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

    auto pParam = std::dynamic_pointer_cast<CInceptionV3Param>(m_pParam);
    if (pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Source image is empty", __func__, __FILE__, __LINE__);

    //Force model files path
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString();
    pParam->m_modelFile = pluginDir + "/Model/tensorflow_inception_graph.pb";
    pParam->m_labelsFile = pluginDir + "/Model/imagenet_names.txt";

    if (!Utils::File::isFileExist(pParam->m_modelFile))
    {
        std::cout << "Downloading model..." << std::endl;
        std::string downloadUrl = Utils::Plugin::getModelHubUrl() + "/" + m_name + "/tensorflow_inception_graph.pb";
        download(downloadUrl, pParam->m_modelFile);
    }

    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> dnnOutputs;
    CMat imgSrc;

    //Need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn(pParam);
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
            readClassNames(pParam->m_labelsFile);
        }

        double inferTime = 0.0;
        if (isWholeImageClassification())
        {
            inferTime = forward(imgSrc, dnnOutputs, pParam);
            manageWholeImageOutput(dnnOutputs[0]);
        }
        else
        {
            auto objects = getInputObjects();
            for (size_t i=0; i<objects.size(); ++i)
            {
                auto subImage = getObjectSubImage(objects[i]);
                inferTime += forward(subImage, dnnOutputs, pParam);
                manageObjectOutput(dnnOutputs[0], objects[i]);
            }
        }
        emit m_signalHandler->doProgress();

        m_customInfo.clear();
        m_customInfo.push_back(std::make_pair("Inference time (ms)", std::to_string(inferTime)));
        endTaskRun();
        emit m_signalHandler->doProgress();
    }
    catch(std::exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }
}

void CInceptionV3::manageWholeImageOutput(cv::Mat &dnnOutput)
{
    //Sort the 1 x n matrix of probabilities
    cv::Mat sortedIdx;
    cv::sortIdx(dnnOutput, sortedIdx, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
    std::vector<std::string> classes;
    std::vector<std::string> confidences;

    for(int i=0; i<sortedIdx.cols; ++i)
    {
        int classId = sortedIdx.at<int>(i);
        std::string className = classId < (int)m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);
        classes.push_back(className);
        confidences.push_back(std::to_string(dnnOutput.at<float>(classId)));
    }
    setWholeImageResults(classes, confidences);
}

void CInceptionV3::manageObjectOutput(cv::Mat &dnnOutput, const ProxyGraphicsItemPtr &objectPtr)
{
    //Sort the 1 x n matrix of probabilities
    cv::Mat sortedIdx;
    cv::sortIdx(dnnOutput, sortedIdx, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
    int classId = sortedIdx.at<int>(0, 0);
    double confidence = dnnOutput.at<float>(classId);
    addObject(objectPtr, classId, confidence);
}
