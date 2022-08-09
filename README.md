# Apple_Detection

## AIM AND OBJECTIVES

## AIM

To create a classification system which detects the quality of an Apple
and specifies whether the given Apple is rotten or not.

## OBJECTIVES

-   The main objective of the project is to do quality check of a given
    Apple within a short period of time.

-   Using appropriate datasets for recognizing and interpreting data
    using machine learning.

-   To show on the optical viewfinder of the camera module whether an
    Apple is rotten or not.

## ABSTRACT

-   An Apple can be classified based on it’s quality and then specified
    whether it is rotten or not on the live feed of the system’s camera.

-   A lot of research is being conducted in the field of Computer Vision
    and Machine Learning (ML), where machines are trained to identify
    various objects from one another. Machine Learning provides various
    techniques through which various objects can be detected.

-   One such technique is to use Resnet Net-50 model with Keras, which
    generates a small size trained model and makes ML integration
    easier.

-   Huge pre–and post–harvest losses are caused by various fruit
    diseases and unfavourable environments leading to the total failure
    of the crops. Citrus decline, apple scab, mango malformation, guava
    wilt, fire blights, banana bunchy top and wilt, brown rots of stone
    fruit, crown galls, downy and powdery mildews are the destructive
    fruit diseases causing huge losses to the fruit industry worldwide.

## INTRODUCTION 

-   This project is based on Apple detection model. We are going to
    implement this project with Machine Learning.

-   This project can also be used to gather information about the
    quality of various fruits like apple, mango, guava etc.

-   Apple quality can be further classified as raw, slightly raw, ripe,
    over ripe and rotten.

-   Based on which side of Apple appears on the camera’s viewfinder the
    Apple may appear ripe or rotten hence using multiple cameras or
    rotation of the Apple is recommended.

-   Today, each and every system is being automated. There is much less
    human intervention as it is both time consuming and non-economical.
    For the defect identification purposes, automated systems are used.
    The image processing has been widely used for identification,
    classification and quality evaluation.

-   Here a sorting process is introduced where the image of the Apple is
    captured and analysed using image processing techniques and the
    defected Apple is discarded by this process. The main aim is to do
    the quality check of the Apples within a short span of time.

-   Due to cost and inaccurate process, sorting tons of quality Apples
    to produce food products made from them is an another problem that
    is faced by most of the agricultural industries.

-   Neural networks and machine learning have been used for these tasks
    and have obtained good results.

-   Machine learning algorithms have proven to be very useful in pattern
    recognition and classification, and hence can be used for Apple
    detection as well.

## LITERATURE REVIEW 

-   Fresh fruits and vegetables are an important part of a healthy diet.
    They contain essential vitamins, minerals, fiber and other nutrients
    that are essential for good health. In fact, research has shown that
    a healthy diet rich in fruits and vegetables may reduce the risk of
    cancer and other chronic diseases.

-   Apples are rich in fiber which is very essential for the smooth
    movement of the digestive system. Apples give body energy as they
    contain carbohydrates which are the main source of energy.
    Carbohydrates in Apples are mainly sugar which actually breaks down
    easily and make a quick source of energy.

-   Antioxidants are essential for human health. These compounds mop up
    free radicals in the body that can damage the body’s cells and lead
    to diseases, such as cancers.

-   As the idiom goes one rotten apple spoils the barrel so does any
    fruit or vegetable can also spoil every fruit surrounding it. It’s a
    scientific fact. And it all has to do with ethylene, a gas produced
    internally by the fruit to stimulate ripening.The closer an apple is
    to rot, the more rot it spreads. Apple producers are commonly
    plagued by this problem because meeting the year-round demand for
    apples means that in some cases the fruit has to be stored for
    months. one spoiling apple, in a crisper drawer or a fruit bowl, or
    a storage barrel or a cross-country shipping container, or even
    still hanging on the bough, speeds the rot of every apple it
    touches, and even of ones it doesn’t touch hence finding the rotten
    ones as fast as possible becomes absolutely necessary.

-   Most fresh fruits and vegetables are picked before they are ripe.
    This allows them time to fully ripen during transportation. It also
    gives them less time to develop a full range of vitamins, minerals
    and natural antioxidants. In the US, fruits and vegetables may spend
    anywhere from 3 days to several weeks in transit before arriving at
    a distribution center.

-   During transportation, fresh produce is generally stored in a
    chilled, controlled atmosphere and treated with chemicals to prevent
    spoiling. Once they reach the supermarket, fruits and vegetables may
    spend an additional 1–3 days on display. They’re then stored in
    people’s homes for up to 7 days before being eaten.

-   Shortly after harvesting, fresh fruits and vegetables start to lose
    moisture, have a greater risk of spoiling and drop in nutrient
    value. The vitamin C in fresh vegetables begins to decline
    immediately after harvesting and continues to do so during storage.

-   Food makes the largest source of waste that goes in muncipal
    landfills and combusted for energy recovery. Many a times the Apples
    from vendors or shopkeepers go to waste because of the Apples
    turning rotten because of one Apple that was not taken out early.

## JETSON NANO COMPATIBILITY

-   The power of modern AI is now available for makers, learners, and
    embedded developers everywhere.

-   NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer
    that lets you run multiple neural networks in parallel for
    applications like image classification, object detection,
    segmentation, and speech processing. All in an easy-to-use platform
    that runs in as little as 5 watts.

-   Hence due to ease of process as well as reduced cost of
    implementation we have used Jetson nano for model detection and
    training.

-   NVIDIA JetPack SDK is the most comprehensive solution for building
    end-to-end accelerated AI applications. All Jetson modules and
    developer kits are supported by JetPack SDK.

-   In our model we have used JetPack version 4.6 which is the latest
    production release and supports all Jetson modules.

> ## PROPOSED SYSTEM
>
> 1\] Study basics of machine learning and image recognition.
>
> 2\] Start with implementation

1.  Front-end development

> B. Back-end development
>
> 3\] Testing, analyzing and improvising the model. An application using
> python and its machine learning libraries will be using machine
> learning to identify whether a given Apple is rotten or not.
>
> 4\] use datasets to interpret the object and suggest whether a given
> Apple on the camera’s viewfinder is rotten or not.

Apple Detection Module

This Module is divided into two parts:

1\. Apple Detection

-   Ability to detect the location of an apple in any input image or
    frame. The output is the bounding box coordinates on the detected
    apple.

-   For this task, initially the Data set library Kaggle was considered.
    But integrating it was a complex task so then we just downloaded the
    images from google images and made our own data set.

-   This Data set identifies an apple in a Bitmap graphic object and
    returns the bounding box image with annotation of name present.

2\. Classification Detection

-   Classification of the Apple based on when they are crossing Railway
    tracks on the viewfinder.

-   Hence YOLOv5 which is a model library from roboflow for image
    classification and vision was used.

> There are other models as well but YOLOv5 is smaller and generally
> easier to use in production. Given it is natively implemented in
> PyTorch (rather than Darknet), modifying the architecture and
> exporting and deployment to many environments is straightforward.

## INSTALLATION

> sudo apt-get remove –purge libreoffice\*
>
> sudo apt-get remove –purge thunderbird\*
>
> sudo fallocate -l 10.0G /swapfile1
>
> sudo chmod 600 /swapfile1
>
> sudo mkswap /swapfile1
>
> sudo vim /etc/fstab
>
> \#################add line###########
>
> /swapfile1 swap defaults 0 0
>
> vim \~/.bashrc
>
> \#############add line \#############
>
> exportPATH=/usr/local/cuda/bin${PATH:+:${PATH}}
>
> exportLD_LIBRARY_PATh=/usr/local/cuda/lib64${LD\\\_LIBRARY\\\_PATH:+:${LD_LIBRARY_PATH}}
>
> exportLD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
>
> sudo apt-get update
>
> sudo apt-get upgrade
>
> \################pip-21.3.1 setuptools-59.6.0
> wheel-0.37.1#############################
>
> sudo apt install curl
>
> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
>
> sudo python3 get-pip.py
>
> sudo apt-get install libopenblas-base libopenmpi-dev
>
> sudo apt-get install python3-dev build-essential autoconf libtool

> pkg-config python-opengl python-pil python-pyrex

> python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer

> libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script

> libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev

> libssl-dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev

> libsasl2-dev libffi-dev libfreetype6-dev python3-dev
>
> vim \~/.bashrc
>
> \####################### add line \####################
>
> exportOPENBLAS_CORETYPE=ARMV8
>
> source\~/.bashrc
>
> sudo pip3 install pillow
>
> curl -LO
> https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
>
> mv p57jwntv436lfrd78inwl7iml6p13fzh.whl

> torch-1.8.0-cp36-cp36m-linux_aarch64.whl
>
> sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
>
> sudo python3 -c “import torch; print(torch.cuda.is_available())”
>
> git clone –branch v0.9.1 https://github.com/pytorch/vision torchvision
>
> cdtorchvision/
>
> sudo python3 setup.py install
>
> cd
>
> git clone https://github.com/ultralytics/yolov5.git
>
> cdyolov5/
>
> sudo pip3 install numpy==1.19.4
>
> history
>
> \#####################comment torch,PyYAML and torchvision in
> requirement.txt##################################
>
> sudo pip3 install –ignore-installed PyYAML\>=5.3.1
>
> sudo pip3 install -r requirements.txt
>
> sudo python3 detect.py
>
> sudo python3 detect.py –weights yolov5s.pt –source 0
>
> \#############################################Tensorflow######################################################
>
> sudo apt-get install python3.6-dev libmysqlclient-dev
>
> sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module
> libcanberra-gtk3-module
>
> pip3 install tqdm cython pycocotools
>
> \#############
> https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.5.0%2Bnv21.8-cp36-cp36m-linux_aarch64.whl
> \######
>
> sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
> zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
>
> sudo apt-get install python3-pip
>
> sudo pip3 install -U pip testresources setuptools==49.6.0
>
> sudo pip3 install -U –no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5
> keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0
> protobuf pybind11 cython pkgconfig
>
> sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
>
> sudo pip3 install -U cython
>
> sudo apt install python3-h5py
>
> sudo pip3 install \#install downloaded tensorflow(sudo pip3 install
> –pre –extra-index-url
> https://developer.download.nvidia.com/compute/redist/jp/v46
> tensorflow)
>
> python3
>
> import tensorflow as tf
>
> tf.config.list_physical_devices(“GPU”)
>
> print(tf.reduce_sum(tf.random.normal(\[1000,1000\])))
>
> \#######################################mediapipe##########################################################
>
> git clone https://github.com/PINTO0309/mediapipe-bin
>
> ls
>
> cdmediapipe-bin/
>
> ls
>
> ./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh
>
> ls
>
> sudo pip3 install mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl

## Demo




https://user-images.githubusercontent.com/89031565/183580952-d643ffe9-bd6f-4aad-8fd1-bccd9561e126.mp4





### **ADVANTAGES**

-   Apple detection system will be of great help in minimizing the
    damage that one rotten Apple can do to the entire barrel of Apples.

-   Apple detection system shows whether the Apple in viewfinder of
    camera module is rotten or not with good accuracy.

-   Once a rotten Apple is detected on the camera module the Apple
    detection model can them convey it to the worker in the shop or
    plant or if needed this process can be completely automated to make
    machines do the work of sorting Apple.

-   When completely automated no user input is required and therefore
    works with absolute efficiency and speed.

-   It can work around the clock and therefore becomes more cost
    efficient.

##     **APPLICATION**

-   Detects a Apple and then checks whether the Apple is rotten or not
    in a given image frame or viewfinder using a camera module.

-   Can be used in shops, malls, plants, storage facilities, godowns and
    at any place where fruits and vegetables are stored and needs
    sorting.

-   Can be used as a refrence for other ai models based on Apple
    Detection.

## **FUTURE SCOPE**

-   As we know technology is marching towards automation so this project
    is one of the step towards automation.

-   Thus, for more accurate results it needs to be trained for more
    images, and for a greater number of epochs.

-   Food detection will become a necessity in the future due to rise in
    population and hence our model will be of great help to tackle the
    situation of starvation and malnutrition due to food shortage in an
    efficient way.

## **CONCLUSION**

-   In this project our model is trying to detect a Apple and then
    > showing it on viewfinder, live as to whether Apple is rotten or
    > not. The Apple detection system is used to automate the
    > segregation of Apples from rotten ones in an efficient way.

-   The model tries to protect the shopkeepers and plant owners from
    > incurring any heavy losses.

-   The model is efficient and highly accurate and hence reduces the
    > workforce required.

-   The Apple detection model has an Training accuracy of 93.56 % and
    > Validation accuracy of 81.33 %.

-   We have trained our model using CNN(Convolutional Neural Network)
    > algorithm on 4,035 fresh and rotten apple images.The model is
    > successfully sorting and predicting apples as fresh and rotten.

    **REFERENCE**

\[1\] O.Kleynen & M.F.Destain. Detection of defects on fruit by machine
vision and unsupervised segmentation(2004).

\[2\] Lu Wang , Anyu Li & Xin Tian .Detection of fruit skin defects
using machine vision systems(2014).

\[3\] Dameshwari Sahu, Ravindra M. Potdar. Defect Identification and
Maturity Detection of Mango fruit using image analysis. American Journal
of Artificial Intelligence . Vol 1. No. 1.2017,pp 5-14 .

\[4\] Shital A. Lakare & Prof. Kapale N.D . Automatic Vision based
technology (June 2019).

## **ARTICLES**

<https://www.healthline.com/nutrition/fresh-vs-frozen-fruit-and-vegetables#TOC_TITLE_HDR_6>

https://www.vahrehvah.com/indianfood/importance-of-fruits-for-healthy-life

<https://www.medicalnewstoday.com/articles/324431#strawberries>

<https://www.newyorker.com/culture/annals-of-gastronomy/how-apples-go-bad>

<https://www.mcgill.ca/oss/article/general-science/rotten-apple-really-does-spoil-barrel>
