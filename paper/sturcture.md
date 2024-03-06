# The structure of report

## OverActivation - Verifying and testing backdoors in deep neural networks

### Abstract
With the widespread integration of deep neural networks (DNNs) into contemporary societal applications, concerns are mounting over their susceptibility to malicious attacks, one such of it is backdoor attacks. Instances such as Tesla's autonomous driving accidents and the prevalence of adversarial attacks on AI image generators underscore the urgent need for robust security measures within DNNs.

This paper proposes a novel backdoor testing approach that leverages the phenomenon of neuron overactivation within DNNs to directly detect the presence of backdoors in models. To evaluate the proposed method, extensive testing and research were conducted on popular datasets and across different models. Experimental results demonstrate the effectiveness of the approach in revealing variations between various model structures and datasets.

In conclusion, this study introduces a valuable tool to mitigate the risk of backdoor attacks in DNNs, providing a reliable testing method to enhance the overall security of these systems. The effectiveness demonstrated in discerning potential threats highlights the potential of this method in fortifying artificial intelligence systems against malicious manipulation.

### Introduction
Deep neural networks (DNNs) have revolutionized image recognition in various fields, including autonomous driving\cite{A1}, medical diagnostics\cite{B1}, and security surveillance\cite{A2}. Their remarkable success, however, is shadowed by their vulnerability to adversarial attacks, which pose significant threats to their reliability and security. Among these adversarial threats, backdoor attacks have emerged as a particular challenge.

Backdoor attacks involve the manipulation of DNN behavior by malicious actors through the injection of images with triggers into the training dataset. Once deployed, these triggers can activate specific behaviors or misclassifications, compromising the integrity and trustworthiness of the entire system. Detecting and mitigating such attacks is paramount for ensuring the robustness and trustworthiness of DNN-based image recognition systems.

### Aim
Related work: One of the existing model backdoor detection tools, NeuralCleanse\cite{NC}, operates under the assumption that smaller modifications will result in misclassification by the model. Another detection method, STRIP\cite{STRIP}, requires prior knowledge of trigger-related information to ascertain whether a model has been injected with a backdoor. This project aims to explore a novel method that offers greater flexibility in usage conditions compared to existing tools and methods. 


### Literature reviews
From Fine-Pruning/cite{FineP} and the direction given by Doctor Rossolini, I learned that pruning defense: a neuron pruning defense method based on neuron activation status has achieved outstanding results in the field of backdoor defense and mitigating the impact of backdoor injection. 


**How the Fine Pruning works...**


I think similar principles can be used in deep neural network backdoor detection., it aims to provide insights that could inspire advancements in the field of backdoor detection to some extent.

