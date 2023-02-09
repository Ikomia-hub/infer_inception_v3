#ifndef INCEPTIONV3_H
#define INCEPTIONV3_H

#include "Inceptionv3Global.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Core/CClassificationTask.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//-----------------------------//
//----- CInceptionV3Param -----//
//-----------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3Param: public COcvDnnProcessParam
{
    public:

        CInceptionV3Param() : COcvDnnProcessParam()
        {
            m_framework = Framework::TENSORFLOW;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            return paramMap;
        }
};

//------------------------//
//----- CInceptionV3 -----//
//------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3: public COcvDnnProcess, public CClassificationTask
{
    public:

        CInceptionV3();
        CInceptionV3(const std::string& name, const std::shared_ptr<CInceptionV3Param>& pParam);

        size_t                      getProgressSteps() override;
        int                         getNetworkInputSize() const override;
        std::vector<std::string>    getOutputsNames() const override;
        void                        globalInputChanged(bool bNewSequence) override;

        void    run() override;

    private:

        void    manageWholeImageOutput(cv::Mat &dnnOutput);
        void    manageObjectOutput(cv::Mat &dnnOutput, const ProxyGraphicsItemPtr &objectPtr);
};

//-------------------------------//
//----- CInceptionV3Factory -----//
//-------------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3Factory : public CTaskFactory
{
    public:

        CInceptionV3Factory()
        {
            m_info.m_name = "infer_inception_v3";
            m_info.m_shortDescription = QObject::tr("Classification deep neural network trained on ImageNet dataset. Developped by Google.").toStdString();
            m_info.m_description = QObject::tr("Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. "
                                               "Since 2014 very deep convolutional networks started to become mainstream, "
                                               "yielding substantial gains in various benchmarks. "
                                               "Although increased model size and computational cost tend to translate to immediate quality gains "
                                               "for most tasks (as long as enough labeled data is provided for training), computational efficiency and "
                                               "low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. "
                                               "Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible "
                                               "by suitably factorized convolutions and aggressive regularization. "
                                               "We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate "
                                               "substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network "
                                               "with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. "
                                               "With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error and 17.3% top-1 error on the validation set and "
                                               "3.6% top-5 error on the official test set.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Classification").toStdString();
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Christian Szegedy, Vincent Vanhoucke, Sergei Ioffe, Jon Shlens, Zbigniew Wojna";
            m_info.m_article = "Rethinking the Inception Architecture for Computer Vision";
            m_info.m_journal = "CVPR";
            m_info.m_year = 2016;
            m_info.m_license = "Apache 2 License";
            m_info.m_repo = "https://github.com/tensorflow/models/tree/master/research";
            m_info.m_keywords = "deep,learning,classification,inception," + Utils::Plugin::getArchitectureKeywords();
            m_info.m_version = "1.2.0";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pManifoldParam = std::dynamic_pointer_cast<CInceptionV3Param>(pParam);
            if(pManifoldParam != nullptr)
                return std::make_shared<CInceptionV3>(m_info.m_name, pManifoldParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pManifoldParam = std::make_shared<CInceptionV3Param>();
            assert(pManifoldParam != nullptr);
            return std::make_shared<CInceptionV3>(m_info.m_name, pManifoldParam);
        }
};

//------------------------------//
//----- CInceptionV3Widget -----//
//------------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3Widget: public COcvWidgetDnnCore
{
    public:

        CInceptionV3Widget(QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            init();
        }
        CInceptionV3Widget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(pParam, parent)
        {
            m_pParam = std::dynamic_pointer_cast<CInceptionV3Param>(pParam);
            init();
        }

    private:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CInceptionV3Param>();
        }

        void onApply() override
        {
            emit doApplyProcess(m_pParam);
        }
};

//-------------------------------------//
//----- CInceptionV3WidgetFactory -----//
//-------------------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3WidgetFactory : public CWidgetFactory
{
    public:

        CInceptionV3WidgetFactory()
        {
            m_name = "infer_inception_v3";
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CInceptionV3Widget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class INCEPTIONV3SHARED_EXPORT CInceptionV3Interface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CInceptionV3Factory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CInceptionV3WidgetFactory>();
        }
};

#endif // INCEPTIONV3_H
