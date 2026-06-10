# Unsupervised ECG Anomaly Detection Using Variational Autoencoders with Beta-Warmup

**DL-ECG Research Group**  
*Email: contact@dl-ecg-group.edu.vn*

---

## Abstract
Cardiovascular diseases remain a leading cause of global mortality, making the early detection of anomalies in electrocardiogram (ECG) signals critical. However, annotating large-scale ECG datasets is time-consuming and prone to severe class imbalance, as anomalous patterns are rare compared to normal rhythms. In this paper, we propose an unsupervised deep learning approach utilizing a Variational Autoencoder (VAE) for ECG anomaly detection. Evaluated on the standard ECG5000 dataset (comprising 140-timestep sequences), our model is trained exclusively on normal heartbeats. To address the latent space collapse (KL Collapse) commonly observed in standard VAEs, we implement a Beta-Warmup technique during optimization. Our ablation study indicates that compressing the latent space to 2 dimensions yields the optimal representation, forcing the model to discard noise and retain essential morphological features. Experimental results demonstrate that the proposed VAE achieves an AUC-ROC of 0.9620, an anomaly precision of 0.98, and an F1-score of 0.90, significantly outperforming traditional baselines such as Isolation Forest and One-Class SVM. Furthermore, we developed a real-time, interactive web application to assist medical professionals in rapid diagnostic screening.

## Index Terms
Electrocardiogram (ECG), Anomaly Detection, Variational Autoencoder (VAE), Deep Learning, Unsupervised Learning, Beta-Warmup.

---

## I. INTRODUCTION
Cardiovascular diseases (CVDs) are the leading cause of death globally. The electrocardiogram (ECG) is the most widely used, non-invasive diagnostic tool for monitoring heart activity and identifying arrhythmias. Traditionally, cardiologists manually analyze these signals; however, continuous monitoring generates massive volumes of data, rendering manual inspection tedious and susceptible to human error [1].

In real-world medical data, normal heartbeats heavily outnumber pathological (anomalous) ones. This severe class imbalance poses a significant challenge for supervised machine learning models. Consequently, Unsupervised Learning and Anomaly Detection have emerged as optimal paradigms. By training models exclusively on healthy ECG patterns, any signal that significantly deviates from the learned distribution is automatically flagged as anomalous [2]. 

This study introduces a $\beta$-VAE (Variational Autoencoder) architecture tailored for 1D time-series data to detect abnormal ECG signals. We mitigate the known KL Collapse issue through a systematic Beta-Warmup strategy, yielding robust anomaly thresholds based on reconstruction probabilities.

## II. RELATED WORK
Research on ECG anomaly detection spans traditional machine learning and modern deep learning methodologies:

* **Traditional Methods:** Algorithms such as One-Class SVM and Isolation Forest have been widely used due to their simplicity. However, they rely heavily on hand-crafted features (e.g., R-R intervals, QRS complex amplitudes) and struggle to capture complex temporal dependencies.
* **Supervised Deep Learning:** Architectures like 1D-CNNs and LSTMs [4] demonstrate exceptional performance on benchmark datasets (e.g., MIT-BIH, ECG5000) [6]. Nevertheless, their reliance on vast amounts of annotated data limits their applicability in novel clinical settings where rare diseases lack sufficient labels.
* **Autoencoders (AE) and VAEs:** AEs are standard for unsupervised anomaly detection based on reconstruction error. Yet, standard AEs suffer from discrete latent spaces and often overfit, reconstructing anomalous inputs too well. VAEs resolve this by mapping inputs to a continuous probabilistic latent space [3]. Bowman et al. [5] highlighted the KL Collapse phenomenon in VAEs and proposed KL annealing (Beta-Warmup), a technique we adapt for robust ECG representation learning.

## III. METHODOLOGY

### A. Variational Autoencoder (VAE) Architecture
Unlike traditional AEs, VAEs map input data to a probability distribution (typically Gaussian) in the latent space. Our proposed architecture consists of two main components:
1. **Encoder:** Takes a 140-dimensional ECG signal ($input\_dim = 140$) as input. It utilizes two dense layers with ReLU activation to progressively reduce dimensionality (to 64, then 32), ultimately outputting two vectors: the mean ($\mu$) and the log-variance ($\log(\sigma^2)$).
2. **Reparameterization Trick:** To allow backpropagation through a random variable, we sample the latent vector $z$ as:
   $$z = \mu + \epsilon \odot \sigma$$
   where $\epsilon \sim \mathcal{N}(0, I)$ is Gaussian noise.
3. **Decoder:** Receives the latent vector $z$ and reconstructs the original signal ($x_{recon}$) through a symmetric dense structure (32 $\rightarrow$ 64 $\rightarrow$ 140).

### B. Loss Function and Beta-Warmup
The objective function of the VAE comprises two terms:
$$\mathcal{L}_{total} = \mathcal{L}_{Reconstruction} + \beta \cdot \mathcal{L}_{KL}$$

* **Reconstruction Loss ($\mathcal{L}_{Reconstruction}$):** Mean Squared Error (MSE) between the input $x$ and the reconstruction $x_{recon}$.
* **KL Divergence ($\mathcal{L}_{KL}$):** Measures the divergence between the learned latent distribution and the prior $\mathcal{N}(0, I)$, acting as a regularizer.
* **Beta-Warmup:** To prevent the decoder from ignoring the latent space during early epochs (KL Collapse), the weight $\beta$ is linearly increased from 0.0 to 1.0 over the first 20 epochs:
   $$\beta = \min\left(1.0, \frac{\text{epoch}}{20}\right)$$

## IV. EXPERIMENTS AND RESULTS

### A. Dataset and Experimental Setup
We utilized the standard ECG5000 dataset [1]. The data was divided into:
* **Train Set:** 2,335 samples (exclusively normal heartbeats).
* **Test Set:** 2,665 samples (584 normal and 2,081 anomalous samples for comprehensive evaluation).

The model was trained for 50 epochs using the Adam optimizer with a learning rate of $10^{-3}$ and a batch size of 64. Anomalies were identified using the Reconstruction Error (MSE). The detection threshold was fixed at the **95th percentile** of the training errors.

### B. Baseline Comparison
To demonstrate the effectiveness of the proposed VAE, we compared it against two traditional unsupervised baselines: Isolation Forest (IF) and One-Class SVM (OCSVM), evaluated on the exact same train/test split.

**Table I: Performance Comparison on ECG5000**

| Model | Accuracy | Precision (Anomaly) | Recall (Anomaly) | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Isolation Forest | 0.5598 | 0.9729 | 0.4488 | 0.6143 | 0.9516 |
| One-Class SVM | 0.7088 | 0.9787 | 0.6410 | 0.7747 | 0.9427 |
| **VAE (Proposed)** | **0.8600** | **0.9800** | **0.8400** | **0.9000** | **0.9620** |

As shown in Table I, while traditional models maintain high precision, they suffer from extremely low recall. The VAE significantly outperforms the baselines across Accuracy, Recall, F1-Score, and AUC-ROC.

### C. Ablation Study on Latent Dimension
We investigated the impact of the latent space dimensionality ($latent\_dim \in \{2, 4, 16\}$) on model performance.

**Table II: Ablation Study Results**

| Metric | $latent\_dim = 2$ | $latent\_dim = 4$ | $latent\_dim = 16$ |
| :--- | :---: | :---: | :---: |
| **Threshold** | 0.0374 | 0.0382 | 0.0375 |
| **AUC-ROC Score** | **0.9620** | 0.9618 | 0.9611 |
| **Normal Recall** | 0.95 | 0.95 | 0.95 |
| **Anomaly Precision**| 0.98 | 0.98 | 0.98 |
| **Anomaly Recall** | **0.84** | 0.83 | 0.83 |

### D. Discussion
1. **Bottleneck Optimization Effect:** The model achieves its highest AUC-ROC (0.9620) when the latent space is compressed to just 2 dimensions. ECG signals exhibit high periodicity; an extreme bottleneck forces the network to discard noise and retain only the core morphological features. Conversely, a larger latent space (16 dimensions) enables the network to effectively reconstruct anomalous inputs as well, marginally degrading detection performance.
2. **Clinical Trade-offs:** The model trades a lower Normal Precision (0.62) for an exceptionally high Normal Recall (0.95) and Anomaly Precision (0.98). In clinical screening, false positives (flagging a healthy heartbeat for review) are vastly preferable to false negatives (missing a critical pathological event).

## V. DEMO APPLICATION
To validate the practical applicability of our research, we developed an interactive web application using Gradio. Medical professionals can input arbitrary ECG samples to receive instantaneous diagnostic feedback. The system outputs the predicted label (Normal/Anomaly), visualizes the raw vs. reconstructed signal, plots the latent space coordinates ($\mu$), and provides a confidence score based on the relative distance of the reconstruction error to the established threshold.

## VI. CONCLUSION AND FUTURE WORK
We successfully applied an unsupervised $\beta$-VAE architecture for anomaly detection on the ECG5000 dataset. By integrating a Beta-Warmup strategy and optimizing the bottleneck dimension ($latent\_dim=2$), the model achieved an outstanding AUC-ROC of 0.9620, proving superior to traditional Isolation Forest and One-Class SVM baselines. 

**Future directions include:**
1. Replacing dense layers with 1D-CNNs or LSTMs to better capture temporal sequence dynamics.
2. Validating the proposed architecture on larger, multi-channel clinical datasets such as MIT-BIH.
3. Deploying the model on edge devices for continuous, low-latency patient monitoring.

---

## REFERENCES
[1] Dau, H. A., et al., "The UCR time series classification archive," *IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293-1305, 2019.  
[2] Chalapathy, R., & Chawla, S., "Deep learning for anomaly detection: A survey," *arXiv preprint arXiv:1901.03407*, 2019.  
[3] An, J., & Cho, S., "Variational autoencoder based anomaly detection using reconstruction probability," *Special Lecture on IE*, vol. 2, no. 1, pp. 1-18, 2015.  
[4] Thill, M., et al., "Anomaly detection in electrocardiogram readings with stacked LSTM networks," in *ITISE*, 2021.  
[5] Bowman, S. R., et al., "Generating sentences from a continuous space," *arXiv preprint arXiv:1511.06349*, 2015.  
[6] Chen, Y., et al., "ECG heartbeat classification using deep transfer learning with Convolutional Neural Network and LightGBM," *IEEE Access*, vol. 8, pp. 109038-109050, 2020.
