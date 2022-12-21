cmake_minimum_required(VERSION 3.16)
set(PLUGIN_VERSION 1.2.0)
project(infer_inception_v3 VERSION ${PLUGIN_VERSION} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/Ikomia/Plugins/C++)
set(CMAKE_INSTALL_PLUGIN_DIR ${CMAKE_INSTALL_PREFIX}/infer_inception_v3)
set(IKOMIA_CORE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../IkomiaCore)

# Set up AUTOMOC and some sensible defaults for runtime execution
# When using Qt 6.3, you can replace the code block below with
# qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)

# Qt
find_package(Qt5 REQUIRED COMPONENTS Core Gui Sql Widgets)

# Python
find_package(Python3 ${PYTHON_VERSION_DOT} REQUIRED COMPONENTS Interpreter Development NumPy)

# OpenMP
find_package(OpenMP)

# OpenCV
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)

add_library(infer_inception_v3 SHARED
    InceptionV3.cpp
    InceptionV3.h
    Inceptionv3Global.h
)

set_target_properties(infer_inception_v3 PROPERTIES
    VERSION ${PLUGIN_VERSION}
    SOVERSION ${PLUGIN_VERSION}
)

target_compile_definitions(infer_inception_v3 PRIVATE
    BOOST_ALL_NO_LIB
    CUDA11
    INCEPTIONV3_LIBRARY
    PY310
    QT_DEPRECATED_WARNINGS
)

target_compile_features(infer_inception_v3 PRIVATE cxx_std_14)

if(MSVC)
    add_compile_options(
        -arch:AVX2
        _CRT_SECURE_NO_WARNINGS
    )
endif()

target_include_directories(infer_inception_v3 PRIVATE
    # Python
    ${Python3_INCLUDE_DIRS}
    # Numpy
    ${Python3_NumPy_INCLUDE_DIRS}
    # Ikomia
    ${IKOMIA_CORE_DIR}/Build/include/Core
    ${IKOMIA_CORE_DIR}/Build/include/DataProcess
    ${IKOMIA_CORE_DIR}/Build/include/Utils
)

target_link_directories(infer_inception_v3 PRIVATE
    ${IKOMIA_CORE_DIR}/Build/lib
)

target_link_libraries(infer_inception_v3 PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
    ikUtils
    ikCore
    ikDataProcess
)

if(WIN32)
    target_link_libraries(infer_inception_v3 PRIVATE
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_dnn${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_dnn${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(infer_inception_v3 PRIVATE
        opencv_core
        opencv_dnn
        opencv_imgproc
    )
endif()

install(TARGETS infer_inception_v3
    LIBRARY DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PLUGIN_DIR}
)