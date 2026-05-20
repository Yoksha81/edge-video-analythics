# Edge Video Analytics Service

## Overview

Edge Video Analytics Service is an in-progress project focused on analyzing camera/video recordings and extracting useful metadata and visual indicators for event detection and future network-aware video transmission.

The long-term idea is to simulate an edge video analytics system where part of the video processing is performed close to the camera or on an edge device, instead of sending every frame to a central backend.

## Motivation

The project connects three areas I am interested in:

- computer vision and video processing,
- machine learning for practical analytics systems,
- telecom/network-aware services and edge computing.

A typical use case would be a camera that detects relevant motion or events locally, sends only metadata or selected frames to a backend, and later adapts its behavior based on network conditions or QoE-related metrics.

## Current Status

The current implementation focuses on basic video analysis and feature extraction from local video files.

Currently explored or implemented components include:

- reading videos from a local dataset folder,
- extracting filename, FPS, total frame count, duration, width and height,
- calculating average brightness,
- experimenting with sharpness estimation,
- experimenting with ROI-based analysis,
- exploring motion-related metrics using frame processing.

The project is still under active development, so the current goal is not to present a finished product, but to build a clean and understandable analytics pipeline step by step.

## Planned Dataset Collection

A planned extension of the project is to use an ESP32-CAM camera module as a simple edge/camera device for collecting a small custom dataset.

The idea is to record a controlled dataset of a parking area in front of my building and use it for testing motion/event detection in a realistic fixed-camera scenario. This would make the dataset more consistent than random videos collected from the internet, because the camera angle, scene background and environment would remain approximately stable.

The dataset could include examples such as:

- no relevant motion,
- small background motion,
- pedestrians,
- vehicles entering or leaving the scene,
- parked vehicles with no activity,
- different lighting and weather conditions.

This dataset would be used carefully, with attention to privacy. The focus would be on technical experimentation with motion detection and edge video analytics, not on identifying people or vehicles.

## Motion Detection Directions

I am currently considering two directions for motion/event detection.

### 1. Frame-processing based motion detection

This version would use classical video-processing methods, such as:

- frame differencing,
- thresholding difference images,
- calculating motion area inside a selected ROI,
- filtering out small or irrelevant motion,
- detecting whether a relevant event happened in a short time window.

This approach is useful because it is simple, explainable and potentially efficient enough for constrained devices.

### 2. ML-based motion/event detection

The second version would use machine learning to classify motion or event types from video-derived features or frame sequences.

Possible directions include:

- training a lightweight model on extracted video features,
- classifying clips into categories such as no motion, small motion and strong/relevant motion,
- later exploring compact models that could run on edge devices.

This direction is more flexible, but it requires a better dataset, careful labeling and evaluation.

## Edge AI Direction

A major future interest of this project is to explore ML models on edge devices.

The goal is to investigate whether video analysis can be performed directly on or near the camera, for example on an edge device, instead of sending full video streams to a central server. This could reduce bandwidth usage, improve latency and make the system more suitable for real-time or semi-real-time scenarios.

Possible future edge-related topics:

- lightweight computer vision models,
- model optimization for constrained devices,
- running inference on an edge device,
- sending only event metadata or selected frames to the backend,
- comparing local processing with backend processing.

## Planned Architecture

The planned system could include:

1. Edge/camera component
   - Reads frames from a camera, video stream or ESP32-CAM module.
   - Performs basic frame processing or ML inference.
   - Detects relevant events.
   - Optionally collects fixed-camera parking-lot recordings for dataset creation.

2. Backend API
   - Receives metadata, metrics or selected frames.
   - Stores results.
   - Exposes analytics endpoints.

3. Database
   - Stores video metadata, motion metrics and event records.

4. Network-aware logic
   - Uses video and network metrics to decide when to send data and at what level of detail.

## Planned Tech Stack

Current / near-term:

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- ROI masks / frame processing
- ESP32-CAM camera module for planned data collection / edge-camera experiments

Planned:

- FastAPI
- Docker
- SQL database
- lightweight ML models
- possible edge-device integration

## Project Goals

Short-term goals:

- build a clean video analysis pipeline,
- define reliable motion metrics,
- compare ROI-based frame processing approaches,
- create a small dataset of fixed-camera test videos,
- experiment with ESP32-CAM as a simple camera module for collecting parking-scene recordings.

Medium-term goals:

- add event detection logic,
- compare classical frame-processing and ML-based approaches,
- expose results through an API,
- store analysis results in a database.

Long-term goals:

- investigate edge-device deployment,
- explore lightweight ML inference on edge hardware,
- connect video analytics with network-aware or QoE-related decision logic.

## Status

This project is in progress and is being developed as a learning-oriented engineering project. The main focus is to understand the full pipeline: video input, feature extraction, event detection, backend integration and possible edge deployment.
