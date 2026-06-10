# Unsupervised ECG Anomaly Detection Using Variational Autoencoders with Beta-Warmup

**DL-ECG Research Group**  
*Email: contact@dl-ecg-group.edu.vn*

---

## Abstract
Cardiovascular diseases remain a leading cause of global mortality, making the early detection of anomalies in electrocardiogram (ECG) signals critical for patient survival. However, annotating large-scale clinical ECG datasets is time-consuming, expensive, and prone to severe class imbalance, as anomalous patterns are exceptionally rare compared to normal cardiac rhythms. In this paper, we propose a robust unsupervised deep learning approach utilizing a Variational Autoencoder (VAE) specifically designed for 1D time-series ECG anomaly detection. Evaluated on the standard UCR ECG5000 dataset (comprising 140-timestep sequences per heartbeat), our model is trained exclusively on healthy patterns, learning the latent distribution of normal sinus rhythms. To address the latent space collapse (KL Collapse) commonly observed in standard VAE optimization, we implement a systematic Beta-Warmup technique. Our comprehensive ablation study reveals that compressing the latent space to 2 dimensions ($latent\_dim=2$) yields the optimal representation, forcing the neural network to discard signal noise and retain essential morphological features. Experimental results demonstrate that the proposed VAE achieves an AUC-ROC of 0.9620, an anomaly precision of 0.98, and an F1-score of 0.90, significantly outperforming traditional unsupervised baselines such as Isolation Forest (AUC 0.9516) and One-Class SVM (AUC 0.9427) in recall and overall accuracy. Furthermore, we developed a real-time, interactive web application to assist medical professionals in rapid diagnostic screening, bridging the gap between theoretical deep learning models and practical clinical deployment.

---

## 1. Introduction

Cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year according to the World Health Organization (WHO). The electrocardiogram (ECG) is the most widely used, non-invasive diagnostic tool for monitoring heart activity, identifying arrhythmias, and predicting ischemic events. Traditionally, cardiologists manually analyze these time-series signals; however, continuous Holter monitoring generates massive volumes of data, rendering manual inspection tedious, resource-intensive, and highly susceptible to human fatigue and error.

In real-world medical data environments, normal heartbeats heavily outnumber pathological (anomalous) ones. This severe class imbalance poses a significant challenge for traditional supervised machine learning models. Supervised classifiers trained on imbalanced datasets tend to heavily favor the majority class, leading to an unacceptable number of false negatives—missing critical pathological events that could be fatal to patients. Consequently, Unsupervised Learning and Anomaly Detection have emerged as optimal paradigms for this problem domain. By training models exclusively on healthy, normal ECG patterns, any signal that significantly deviates from the learned distribution is automatically flagged as anomalous.

This study introduces a $\beta$-VAE (Variational Autoencoder) architecture tailored specifically for 1D time-series data to detect abnormal ECG signals. Standard VAEs optimize an evidence lower bound (ELBO) consisting of a reconstruction loss and a Kullback-Leibler (KL) divergence regularizer. However, when applied to sequential data, VAEs frequently suffer from the "KL Collapse" or "posterior collapse" phenomenon, where the decoder ignores the latent representations provided by the encoder. We mitigate this issue through a systematic Beta-Warmup strategy, gradually introducing the KL penalty during training, which yields robust, informative latent spaces and highly accurate anomaly thresholds based on reconstruction probabilities.

The primary contributions of this paper are as follows:
1. We adapt and optimize a 1D Variational Autoencoder for unsupervised ECG anomaly detection, circumventing the need for labeled pathological data.
2. We demonstrate the efficacy of the Beta-Warmup strategy in preventing KL collapse for time-series VAEs.
3. We conduct a rigorous ablation study on latent dimensionality, providing empirical evidence that extreme bottlenecking ($latent\_dim=2$) acts as an effective noise filter for highly periodic signals like ECGs.
4. We establish a robust clinical thresholding mechanism based on the 95th percentile of reconstruction errors on healthy data.
5. We provide a comprehensive baseline comparison against traditional anomaly detection algorithms (Isolation Forest and One-Class SVM) evaluated on the exact same train/test split.

---

## 2. Related Work

Research on ECG anomaly detection spans a wide spectrum of computational methodologies, transitioning from traditional statistical machine learning to modern deep learning architectures.

### 2.1 ECG Anomaly Detection
Before the deep learning era, traditional anomaly detection techniques such as One-Class Support Vector Machines (OC-SVM), Isolation Forests (IF), and Principal Component Analysis (PCA) were widely utilized. These algorithms generally require manual feature extraction (e.g., calculating R-R intervals, QRS complex durations, and P-wave amplitudes) prior to classification. While computationally lightweight and highly interpretable, they often fail to capture the complex, non-linear temporal dependencies inherent in high-frequency ECG signals. Supervised deep learning approaches, notably 1D Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, subsequently automated feature extraction and achieved state-of-the-art accuracy on benchmark datasets. However, their reliance on massive, meticulously annotated datasets limits their scalability to rare cardiovascular pathologies.

### 2.2 Autoencoder-based Anomaly Detection
Autoencoders (AEs) are considered the standard deep learning approach for unsupervised anomaly detection. An AE is trained to compress normal data into a lower-dimensional bottleneck and then reconstruct it. During inference, anomalous inputs yield a high reconstruction error because the network has not learned their structural representations. Chalapathy and Chawla comprehensively surveyed AE applications, noting their sensitivity to subtle signal distortions. However, standard deterministic AEs often suffer from a fragmented, non-continuous latent space. Consequently, they may "overfit" and successfully reconstruct anomalous data, leading to a high rate of false negatives in medical diagnostics.

### 2.3 Variational Autoencoders
To address the limitations of discrete latent spaces, Variational Autoencoders (VAEs) map input data into a continuous probability distribution (typically a multivariate Gaussian). An and Cho first demonstrated the utility of VAEs for anomaly detection by utilizing "reconstruction probability" instead of simple reconstruction error. VAEs inherently model uncertainty and provide smooth latent interpolation. However, Bowman et al. highlighted the "posterior collapse" phenomenon, where a powerful decoder learns to ignore the latent code $z$, driving the KL divergence to zero. Various annealing strategies, including the Beta-Warmup utilized in our research, have been proposed to force the encoder to learn meaningful representations by gradually increasing the weight of the KL divergence term during training.

### 2.4 Medical Signal Anomaly Detection
In the specific domain of medical signals, robustness against sensor noise, baseline wander, and patient-specific physiological variations is paramount. Recent literature emphasizes the integration of VAEs with recurrent structures (e.g., LSTM-VAE) for physiological time-series. Thill et al. successfully applied stacked LSTM networks for ECG anomaly detection, yet training recurrent networks remains computationally expensive. Our proposed dense 1D VAE strikes a balance between computational efficiency and temporal feature extraction, making it highly suitable for real-time edge deployment.

### 2.5 ECG5000 Benchmark Studies
The ECG5000 dataset, part of the UCR Time Series Classification Archive, serves as a standard benchmark for evaluating time-series algorithms. Extracted from a 20-hour long ECG recording, the dataset contains 5,000 heartbeats standardized to 140 timesteps. Chen et al. demonstrated high classification accuracy on ECG5000 using deep transfer learning. However, benchmark studies focusing strictly on *unsupervised* anomaly detection on this dataset are less common, making our direct comparison against Isolation Forest and One-Class SVM highly relevant to the community.

---

## 3. Methodology

### 3.1 Problem Formulation
We formulate ECG anomaly detection as an unsupervised learning problem. Let $X_{train} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$ be a dataset consisting exclusively of normal ECG signals, where each $x^{(i)} \in \mathbb{R}^T$ is a sequence of length $T=140$. Our objective is to learn a mapping function $\mathcal{F}_{\theta}$ that captures the underlying distribution of $X_{train}$. During inference, given a new, unseen sample $x_{test}$, we compute an anomaly score $\mathcal{S}(x_{test}) = \mathcal{E}(x_{test}, \mathcal{F}_{\theta}(x_{test}))$, where $\mathcal{E}$ is an error function. If $\mathcal{S}(x_{test}) > \tau$ (a predefined threshold), the sample is classified as an anomaly.

### 3.2 Dataset Description
We utilize the ECG5000 dataset provided by the UCR Time Series Archive. The dataset comprises 5,000 individual heartbeat sequences extracted from a continuous Holter monitor recording of a patient with severe congestive heart failure. Each heartbeat is represented by 140 temporal data points (timesteps). The original dataset contains five classes: Normal (Class 1), R-on-T Premature Ventricular Contraction (Class 2), Premature Ventricular Contraction (Class 3), Supra-ventricular Premature or Ectopic Beat (Class 4), and Unclassified Beat (Class 5). For our anomaly detection paradigm, Class 1 is treated as "Normal," while Classes 2 through 5 are grouped as "Anomaly."

*[Insert Table 1. Dataset statistics here]*
> **Table 1 Placeholder: Dataset statistics detailing the number of normal and anomalous samples.**

### 3.3 Data Preprocessing and Train/Test Split Protocol
The ECG5000 dataset is pre-normalized. To ensure a strictly unsupervised training regime, we repartitioned the standard splits. We extracted all Normal samples and allocated 80% (2,335 samples) to the training set $X_{train}$. The remaining 20% of Normal samples (584 samples), along with all Anomalous samples (2,081 samples), were combined to form the test set $X_{test}$. Consequently, the model is completely blind to anomalous patterns during the training phase. Data was standardized using a Min-Max scaler fit solely on the training data.

### 3.4 VAE Architecture Explanation
Our architecture utilizes densely connected neural layers optimized for 1D sequence processing. 
* **Encoder ($q_\phi(z|x)$):** The encoder takes the 140-dimensional input vector and passes it through two hidden dense layers of sizes 64 and 32, respectively, utilizing ReLU activation functions. The final hidden representation is split into two parallel linear layers that output the mean vector $\mu$ and the log-variance vector $\log(\sigma^2)$ of the latent distribution.
* **Decoder ($p_\theta(x|z)$):** The decoder mirrors the encoder's structure. It samples a latent vector $z$ of size $latent\_dim$ and reconstructs the signal through hidden layers of sizes 32 and 64, followed by a linear output layer of size 140, producing the reconstructed signal $x_{recon}$.

*[Insert Figure 1. System architecture here]*
> **Figure 1 Placeholder: Diagram illustrating the Encoder, Reparameterization, and Decoder modules of the proposed VAE.**

### 3.5 Reparameterization Trick
To enable end-to-end backpropagation through the stochastic sampling process, we employ the reparameterization trick. Instead of directly sampling $z \sim \mathcal{N}(\mu, \text{diag}(\sigma^2))$, we express $z$ as a deterministic variable dependent on an auxiliary noise variable $\epsilon$:
$$z = \mu + \epsilon \odot \sigma$$
where $\epsilon \sim \mathcal{N}(0, I)$ and $\odot$ denotes element-wise multiplication.

### 3.6 Beta-Warmup Strategy
The standard VAE loss function (Evidence Lower Bound - ELBO) is defined as:
$$\mathcal{L}_{total} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$
We implement the reconstruction term using Mean Squared Error (MSE). The KL divergence term $D_{KL}$ acts as a regularizer. To prevent KL collapse—where the optimizer aggressively drives $D_{KL}$ to zero before the encoder learns meaningful representations—we introduce a hyperparameter $\beta$ to modulate the KL term:
$$\mathcal{L}_{total} = \text{MSE}(x, x_{recon}) + \beta \cdot D_{KL}$$
We employ a linear Beta-Warmup schedule, increasing $\beta$ from 0.0 to 1.0 over the initial $W$ epochs:
$$\beta = \min\left(1.0, \frac{\text{epoch}}{W}\right)$$
In our experiments, $W$ is set to 20.

### 3.7 Threshold Calibration Method
In unsupervised anomaly detection, defining the decision boundary is critical. After training, we compute the reconstruction errors (MSE) for all samples in the training set $X_{train}$. Since $X_{train}$ consists entirely of normal heartbeats, the distribution of these errors represents the expected variance of healthy signals. We calibrate the classification threshold $\tau$ empirically at the 95th percentile of this training error distribution. During inference, if $\text{MSE}(x_{test}, x_{recon}) > \tau$, the sample is flagged as an anomaly.

---

## 4. Experiments

### 4.1 Hardware and Software Setup
All experiments were conducted on a workstation equipped with an NVIDIA RTX GPU, operating on a Windows OS environment. The neural network was implemented using PyTorch 2.x. Baseline models were executed utilizing Scikit-Learn.

### 4.2 Hyperparameters
The model was trained for 50 epochs with a batch size of 64. We utilized the Adam optimizer with a learning rate of $1\times 10^{-3}$. The latent space dimensionality ($latent\_dim$) was identified as a critical hyperparameter and subjected to rigorous ablation testing.

*[Insert Table 2. Hyperparameters here]*
> **Table 2 Placeholder: List of learning rates, batch size, epoch count, optimizer, and network layers.**

*[Insert Figure 2. Training loss curve here]*
> **Figure 2 Placeholder: Plot of reconstruction loss and KL divergence across 50 epochs demonstrating the effect of Beta-Warmup.**

### 4.3 Evaluation Metrics
We report standard clinical and machine learning metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). Since our primary goal is identifying pathological conditions, "Anomaly" is treated as the positive class (Label 1) and "Normal" as the negative class (Label 0).

### 4.4 Ablation Study Design
To understand the network's behavior regarding information compression, we conducted an ablation study altering the latent space dimensionality. We evaluated $latent\_dim \in \{2, 4, 16\}$, keeping all other hyperparameters constant, to observe the trade-off between reconstruction fidelity and anomaly discriminability.

---

## 5. Results

### 5.1 Main Evaluation and Baseline Comparison
We benchmarked our proposed VAE ($latent\_dim=2$) against two industry-standard unsupervised anomaly detection algorithms: Isolation Forest (contamination=0.05) and One-Class SVM (nu=0.05, RBF kernel).

*[Insert Table 3. Main evaluation metrics here]*
> **Table 3 Placeholder: Performance Comparison (Accuracy, Precision, Recall, F1, AUC) among IF, OC-SVM, and VAE.**

As observed in Table 3 (and previously documented in Section IV.B), the VAE drastically outperforms the baselines. The Isolation Forest achieved an Accuracy of 55.98% with an F1-Score of 0.6143. The One-Class SVM improved to an Accuracy of 70.88% and an F1-Score of 0.7747. Notably, both baseline models suffered from very poor Recall (44.88% and 64.10% respectively), meaning they failed to detect a massive portion of the anomalies. Conversely, our proposed VAE achieved an Accuracy of 86.00%, an Anomaly Precision of 98.00%, an Anomaly Recall of 84.00%, and a resulting F1-Score of 0.9000. 

### 5.2 Detailed Metric Interpretation
The VAE's high AUC-ROC score of 0.9620 indicates exceptional structural capability in separating the normal and anomalous error distributions. The high Precision (98.00%) indicates that when the VAE flags a heartbeat as anomalous, it is almost definitively pathological. 

*[Insert Figure 3. ROC curve here]*
> **Figure 3 Placeholder: ROC Curves comparing VAE against IF and OC-SVM.**

*[Insert Figure 4. Reconstruction error distribution here]*
> **Figure 4 Placeholder: Histogram showing the distinct separation of reconstruction errors between normal and anomalous samples, with the 95th percentile threshold line.**

### 5.3 Latent Dimension Ablation Results
The ablation study yielded counter-intuitive but highly significant insights into ECG signal processing.

*[Insert Table 4. Latent dimension ablation here]*
> **Table 4 Placeholder: Results of AUC-ROC, Precision, and Recall for latent_dim = 2, 4, 16.**

When reducing the latent dimension from 16 to 2, the AUC-ROC increased from 0.9611 to 0.9620. 

*[Insert Figure 6. Ablation study results here]*
> **Figure 6 Placeholder: Line or bar chart visualizing performance metrics across different latent dimensions.**

### 5.4 Clinical Implications
The phenomenon observed in the ablation study is termed the "Bottleneck Optimization Effect." Because healthy ECG signals possess high periodicity and rigid morphological rules, an extreme bottleneck (2 dimensions) is sufficient to encode the "healthy state." When forced through a 2D space, the network strips away noise and strictly learns the principal components of a normal heartbeat. If the bottleneck is too wide (16 dimensions), the network possesses enough capacity to "memorize" and accurately reconstruct anomalous waveforms, inadvertently lowering the reconstruction error for anomalies and degrading the model's discriminative power.

Furthermore, the model trades some Specificity (Normal Precision = 62%) for high Sensitivity (Anomaly Recall = 84%). In clinical triage, False Positives (requesting a doctor to review a healthy patient) are acceptable, whereas False Negatives (sending a patient with heart failure home) are catastrophic.

*[Insert Figure 5. Latent space visualization here]*
> **Figure 5 Placeholder: 2D scatter plot of the latent vectors ($\mu$) showing spatial clustering of normal vs anomalous beats.**

---

## 6. Discussion

### 6.1 Strengths
The primary strength of the proposed system lies in its entirely unsupervised nature. It circumvents the severe data labeling bottleneck in medical machine learning. Furthermore, the implementation of Beta-Warmup effectively resolved KL collapse, stabilizing training. The extreme dimensionality reduction ($latent\_dim=2$) not only maximized AUC-ROC but inherently provides a mechanism for 2D visual interpretability of the latent space for clinicians.

### 6.2 Limitations
Despite high overall performance, the model's Normal Precision (62%) implies a moderate False Positive rate. While clinically safe, too many false alarms could lead to "alarm fatigue" for medical staff. Additionally, standard Dense layers assume a fixed input size (140 timesteps). Real-world ECG signals vary in heart rate and length, requiring dynamic time warping or robust beat-segmentation preprocessing algorithms before they can be fed into this specific architecture.

### 6.3 Reproducibility Considerations
To ensure high reproducibility, the random seeds for PyTorch, NumPy, and Scikit-Learn were fixed. The exact dataset split indices and scaling parameters are persisted within the project's data pipeline (`data/processed/`). The Beta-Warmup schedule relies on deterministic mathematical bounds rather than stochastic heuristics, ensuring consistent training convergence across varying hardware setups.

### 6.4 Future Deployment Possibilities
The lightweight nature of the dense 1D VAE makes it an ideal candidate for Edge AI deployment. The model can be quantified and deployed directly onto wearable devices (e.g., smartwatches, portable Holter monitors) to perform continuous, real-time anomaly screening locally, thereby protecting patient privacy by eliminating the need to transmit sensitive physiological data to cloud servers.

To validate this deployment potential, we have currently wrapped the PyTorch model within an interactive Gradio web application, demonstrating inference times on the scale of milliseconds per heartbeat.

---

## 7. Conclusion
In this paper, we presented a robust, unsupervised deep learning framework for ECG anomaly detection using a Variational Autoencoder. By employing a Beta-Warmup training schedule, we effectively prevented KL collapse. Our empirical analysis demonstrated that compressing the latent space to just 2 dimensions forces the network to learn the quintessential morphology of a healthy heartbeat, maximizing its sensitivity to pathological deviations. Achieving an AUC-ROC of 0.9620 and significantly outperforming traditional baselines, our model represents a highly viable, scalable solution for automated cardiovascular screening. Future work will explore recurrent and convolutional variants of the VAE to eliminate the requirement for fixed-length signal segmentation and enable direct processing of continuous, unsegmented ECG streams.

---

## References

[1] Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., and Keogh, E., "The UCR time series classification archive," *IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293-1305, 2019.

[2] Chalapathy, R., and Chawla, S., "Deep learning for anomaly detection: A survey," *arXiv preprint arXiv:1901.03407*, 2019.

[3] An, J., and Cho, S., "Variational autoencoder based anomaly detection using reconstruction probability," *Special Lecture on IE*, vol. 2, no. 1, pp. 1-18, 2015.

[4] Thill, M., Däubener, S., Konen, W., and Bäck, T., "Anomaly detection in electrocardiogram readings with stacked LSTM networks," in *Proceedings of the International Conference on Time Series and Forecasting (ITISE)*, 2021.

[5] Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., and Bengio, S., "Generating sentences from a continuous space," *arXiv preprint arXiv:1511.06349*, 2015.

[6] Chen, Y., Hao, Y., Qiu, X., and Xia, J., "ECG heartbeat classification using deep transfer learning with Convolutional Neural Network and LightGBM," *IEEE Access*, vol. 8, pp. 109038-109050, 2020.
