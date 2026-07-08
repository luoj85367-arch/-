# Heart Hand Link - Wearable Sign Language Translation System
## Project Overview
There are obvious communication barriers between hearing-impaired groups and hearing people in daily life. Traditional camera-based sign language recognition is susceptible to light changes and occlusion, while cloud translation solutions rely on network access, suffer from high latency and risk privacy leakage. This project develops a flexible wearable sign language translation device based on RDK X5. Integrating flexible sensors, IMU motion capture and edge AI technologies, the device leverages the local NPU computing power of the chip to complete offline sign language recognition. It supports bidirectional barrier-free conversion between sign language, voice and text, and runs stably with low latency without network connection, serving as a lightweight assistive interaction solution for the hearing-impaired.

## Core Functions
1. Equipped with 12-channel BNO055 9-axis IMUs to capture hand motion data, completely eliminating environmental interference issues of visual recognition.
2. Offline inference based on local TCN temporal neural network, realizing bidirectional translation: real-time conversion from sign language to voice & text, and voice to text display.
3. Extended gesture control capability: users can operate smart home devices and drones via sign language, realizing multi-functional integration.

## Core Technical Highlights
1. Adopt inertial sensors instead of cameras to enable stable recognition under all lighting conditions and occlusion scenarios, and fully protect user privacy.
2. Local edge computing powered by RDK X5, supporting offline operation without network dependence and greatly reducing interaction latency.
3. Thin & flexible wearable design with split wireless transmission modules, bringing constraint-free wearing experience and adapting to all-day usage.

## Application Scenarios
It serves hearing-impaired people for daily social communication, medical consultation and government service handling, sign language teaching in special education schools. Meanwhile, it can be used as a somatosensory control terminal for smart home systems and an embedded technical training platform for university STEM projects.

## System Hardware Composition
- Collection Terminal: ESP32-C3 main controller + flexible sensing gloves with 12 IMUs on both hands; data transmitted via Wi-Fi dual-channel TCP protocol
- Computing Terminal: RDK X5 (Xuri 5), local NPU for AI model inference
- Output Modules: Voice broadcast unit & text display peripheral

## Key Performance Parameters
- Sign language recognition accuracy: 94.65%; model inference frame rate: 40 FPS
- Supports 35 categories of daily common sign languages; near-zero packet loss rate for wireless transmission
- Fully offline independent operation, immune to network status, light intensity and hand occlusion
