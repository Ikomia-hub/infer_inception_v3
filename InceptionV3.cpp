#include "InceptionV3.h"
#include "Graphics/CGraphicsLayer.h"

CInceptionV3::CInceptionV3(): COcvDnnProcess()
{
    m_pParam = std::make_shared<CInceptionV3Param>();
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

CInceptionV3::CInceptionV3(const std::string &name, const std::shared_ptr<CInceptionV3Param> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CInceptionV3Param>(*pParam);
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
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

void CInceptionV3::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CInceptionV3Param>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

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
    cv::Mat dnnOutput;
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
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
        }

        int size = getNetworkInputSize();
        double scaleFactor = getNetworkInputScaleFactor();
        cv::Scalar mean = getNetworkInputMean();
        auto inputBlob = cv::dnn::blobFromImage(imgSrc, scaleFactor, cv::Size(size,size), mean, false, false);
        m_net.setInput(inputBlob);
        dnnOutput = m_net.forward(m_outputLayerName);
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    readClassNames();
    endTaskRun();
    emit m_signalHandler->doProgress();
    manageOutput(dnnOutput);
    emit m_signalHandler->doProgress();

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
    {
        m_sign *= -1;
        m_bNewInput = false;
    }
}

void CInceptionV3::manageOutput(cv::Mat &dnnOutput)
{
    forwardInputImage();

    //Sort the 1 x n matrix of probabilities
    cv::Mat sortedIdx;
    cv::sortIdx(dnnOutput, sortedIdx, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);

    auto classId = sortedIdx.at<int>(0, 0);
    double confidence = dnnOutput.at<float>(classId);

    //Graphics output
    auto pGraphicsOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(1));
    assert(pGraphicsOutput);
    pGraphicsOutput->setNewLayer("InceptionV3");
    pGraphicsOutput->setImageIndex(0);

    //Measures output
    auto pMeasureOutput = std::dynamic_pointer_cast<CBlobMeasureIO>(getOutput(2));
    pMeasureOutput->clearData();

    //We don't create the final CGraphicsText instance here for thread-safety reason
    //So we saved necessary information into the output and the final object is
    //created when the output is managed by the App
    std::string className = classId < (int)m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);
    std::string label = className + " : " + std::to_string(confidence);
    pGraphicsOutput->addText(label, 30, 30);

    //Store values to be shown in results table
    for(int i=0; i<sortedIdx.cols; ++i)
    {
        classId = sortedIdx.at<int>(i);
        className = classId < (int)m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);
        pMeasureOutput->addObjectMeasure(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), dnnOutput.at<float>(classId), i, className));
    }
}
