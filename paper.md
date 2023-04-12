---
title: 'Ethome: machine learning for animal behavior'
tags:
  - Python
  - supervised-learning
  - deeplabcut
  - boris
  - neurodata-without-borders
  - pose-tracking
  - ndx-pose
  - animal-behavior
authors:
  - name: Benjamin Lansdell
    orcid: 0000-0003-1444-1950
    equal-contrib: false
    affiliation: 1
  - name: Abbas Shirinifard
    equal-contrib: false 
    affiliation: 1
affiliations:
 - name: Developmental Neurobiology, St Jude Children's Research Hospital, Memphis, Tennessee, USA
   index: 1
date: 14 March 2023
bibliography: paper.bib

---

# Summary

`ethome` supports machine learning of animal behavior. It interprets pose-tracking files and behavior annotations to create feature tables, train behavior classifiers, interpolate pose tracking data and other common analysis tasks. 
It features:

* Read in animal pose data and corresponding behavior annotations to make supervised learning easy
* Scale data to desired physical units
* Interpolate pose data to improve low-confidence predictions
* Create generic features for analysis and downstream ML tasks
* Create features specifically for mouse resident-intruder setup
* Quickly generate a movie with behavior predictions

Together these features significantly reduce the code a user needs to write to perform animal behavior analysis -- they can focus exclusively on developing the best machine learning models for their problem, instead of dealing with data imports, unit conversations, time alignments between data sources, etc. The current version supports DeepLabCut (DLC) and SLEAP [@Pereira2022sleap] tracking data, BORIS behavioral annotations, and NWB behavioral data files, all heavily used standards. The demo notebooks provided in this repository demonstrate how few lines of code are needed to create useful ML models. 

# Statement of need

The quantitative study animal behavior is rapidly growing [@Mathis2020DeepLT], with pose-tracking tools like DeepLabCut [@Lauer2022MultianimalPE] making accurate pose estimation easily and widely applicable. Such data promises to revolutionize the study of animal behavior. 

With the vast amount of data available, there is a growing need for analysis tools that simplify post-tracking machine learning tasks. After poses have been tracked in a series of videos, both supervised learning (behavior classification) and unsupervised learning can offer valuable insights. Currently, there are two main options for conducting machine learning analysis: using a comprehensive behavior analysis software package like SimBA [@Nilsson2020-simba], or writing custom analysis code. 
 
SimBA provides a user-friendly GUI, making it an ideal solution for those who prefer a no-code approach. However, this convenience comes at the cost of flexibility. On the other hand, writing custom analysis code requires navigating numerous tedious details, such as understanding the file formats used by the software generating the behavioral data, aligning pose and behavior annotation data, converting units, interpolating or correcting inaccurate pose estimates, and creating relevant features for machine learning. This process can be quite time-consuming. 


The goal of `ethome` is to perform all the required processing steps for the user, so that both experienced and novice coders/machine learning practitioners can easily build ML models. The target user is anyone familiar with basic python and machine learning, with animal behavior and pose tracking questions that would benefit from machine learning. 

# Related work

A range of packages solve related problems: `SimBA`, as mentioned, also is built for supervised machine learning on pose tracking data, but has less flexibility -- only a few mouse setups are fully supported -- in contrast to `ethome`'s support for generic animal tracking setups; `DLCAnalyzer` [@Sturman2020-dlcanalyzer], an `R` package, also addresses some of the same data processing steps as `ethome` needed to analyze DLC tracking data, but doesn't also include support for behavioral annotations, or feature creation useful for ML; `DLC2Kinematics` [@Mathis2020-DLC2Kinematics], from the DeepLabCut group, is similar in this way -- while it does provide computation of kinematic features that may be useful for ML, it doesn't read in accompanying behavioral annotations needed for supervised learning; `BentoMAT` and `MARS` [@Segalin2021-bento] provide a nice GUI and behavioral classifiers, but are focused on mouse behavior. 

Advances in deep learning has lead to a number of interesting models: `TREBA` [@Sun2020-TREBA] learns a representation that can be used for supervised ML, and `DeepEthogram` [@Bohnslav2021-deepethogram] learns a behavior classifier directly from video frames, rather than using pose estimation as an intermediate step. Associated software is focused on implementing these specific methods. Unsupervised approaches like `B-SOID` [@Hsu2021bsoid] and `VAME` [@Luxem2022vame] provide useful clustering of behavior, but do not integrate human expert annotation. Finally a recent approach, `A-SOID`, combines unsupervised and supervised approaches [@Schweihoff2022asoid] in a web-based GUI. This approach is useful, but again provides limited flexibility and support for custom-feature sets. These packages are not aimed at addressing the specific need for a lightweight, flexible framework to do machine learning on general pose data. `ethome` makes such analyses significantly faster and more straightforward. 

# Design 

The package creates and manipulates an extended Pandas `DataFrame` to house both pose data and behavioral labels for a set of recordings. The idea is that all added functionality operates on a single object already familiar to users, enabling maximum flexibility and support for as broad a range of behavior and pose analyses as possible. Extended dataframes can be treated exactly as one would treat any other dataframe. `ethome` then adds metadata and functionality to this object. 

A key feature of this tool is the ease with which it enables users to create features for machine learning. The user can use pre-built featuresets or make their own. For instance `dataframe.features.add('distances')` will compute all distances between all body parts (both between and within animals). There are also featuresets specifically tailored for social mice studies (resident-intruder studies). A more detailed run through of features is provided in the Readme and How To guide. 

# Acknowledgements

The authors thank Ryan Ly for feedback on incorporating NWB pose data into the package.

# References
